# Copyright 2024 **AUTHORS_TODO**
# License: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Apache-2.0 license
# Modified from Mamba LayerNorm/RMSNorm https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "BLOCK_N"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    low_precision: tl.constexpr,  # if true, doesn't upcast the RMSNorm calculation to float32
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    if low_precision:
        x = tl.load(X + cols, mask=cols < N, other=0.0)
    else:
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < N
    if low_precision:
        w = tl.load(W + cols, mask=mask)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "BLOCK_N"],
)
@triton.jit
def _rms_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_dy_row,
    stride_dx_row,
    M,  # number of rows in X
    N,  # number of columns in X
    rows_per_program,
    low_precision: tl.constexpr,  # if true, doesn't upcast the RMSNorm calculation to float32
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Compute the pointers and load the weights for the current block
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if low_precision:
        w = tl.load(W + cols, mask=mask)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)

    # Initialize the partial sum of weights gradient for the current block
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Compute the end row for the current block
    row_end = min((row_block_id + 1) * rows_per_program, M)

    # Iterate over the rows in the current block
    for row in range(row_start, row_end):
        # Load the input, output gradient, and 1/std for the current row
        if low_precision:
            x = tl.load(X + cols, mask=mask, other=0)
            dy = tl.load(DY + cols, mask=mask, other=0)
        else:
            x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        # Compute the normalized input (xhat) and the element-wise product of weights and output gradient (wdy)
        xhat = x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        wdy = w * dy

        # Accumulate the partial sum of weights gradient and compute the sum of element-wise product of xhat and wdy
        dw += dy * xhat
        c1 = tl.sum(xhat * wdy, axis=0) / N

        # Compute the input gradient and store it for the current row
        dx = (wdy - xhat * c1) * rstd
        tl.store(DX + cols, dx, mask=mask)

        # Update the pointers for the next row
        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row

    # Store the partial sum of weights gradient for the current block
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


class ApplyRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6, low_precision=False):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.view(-1, x.shape[-1])
        n_rows, n_cols = x.shape

        y = torch.empty_like(x)
        rstd = torch.empty((n_rows,), dtype=x.dtype if low_precision else torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(x.shape[-1]))
        if n_cols > BLOCK_N:
            raise RuntimeError("This RMSNorm doesn't support feature dim >= 64KB.")

        # Enqueue kernel
        _rms_norm_fwd_kernel[(n_rows,)](
            x,
            y,
            weight,
            rstd,
            x.stride(0),
            y.stride(0),
            n_cols,
            eps,
            low_precision,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_N = BLOCK_N
        ctx.eps = eps
        ctx.low_precision = low_precision
        ctx.x_shape_og = x_shape_og

        return y.view(x_shape_og)

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dy = dy.view(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape

        dx = torch.empty_like(dy)

        # Enqueue kernel using Triton
        M, N = dy.shape
        sm_count = torch.cuda.get_device_properties(dy.device).multi_processor_count
        _dw = torch.empty(
            (sm_count, N), dtype=weight.dtype if ctx.low_precision else torch.float32, device=weight.device
        )
        rows_per_program = math.ceil(M / sm_count)
        _rms_norm_bwd_kernel[(sm_count,)](
            x,
            weight,
            dy,
            dx,
            _dw,
            rstd,
            x.stride(0),
            dy.stride(0),
            dx.stride(0),
            M,
            N,
            rows_per_program,
            ctx.low_precision,
            BLOCK_N=ctx.BLOCK_N,
        )

        dw = _dw.sum(0).to(weight.dtype)
        return dx.view(ctx.x_shape_og), dw, None, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        low_precision: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.low_precision = low_precision
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor):
        return ApplyRMSNorm.apply(x, self.weight, self.eps, self.low_precision)
