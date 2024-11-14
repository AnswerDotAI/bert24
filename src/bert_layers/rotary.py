# Copyright 2024 **AUTHORS_TODO**
# License: Apache-2.0

# Copyright (c) 2023, Tri Dao.
# License: Apache-2.0

# This rotary implementation is modified from the original flash attention implementation to support torch.compile using the new PyTorch 2.4 custom_op.
# It also is a simplified version of the rotary implementation in flash attention, as it only supports GPT-NeoX style rotary embeddings for variable
# length sequences and it no longer supports seqlen_offset for kvcache as bidirectional transformers do not have a kvcache


from typing import Optional

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch import Tensor


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    # Matrix dimensions
    seqlen,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
    OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)

    # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
    X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
    COS = COS + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    cos = tl.load(COS, mask=(rm[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=(rm[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    x1 = tl.load(
        X + rotary_dim_half * stride_x_headdim,
        mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        other=0.0,
    ).to(tl.float32)
    if CONJUGATE:
        sin = -sin
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    # write back result
    OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
    tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(
        OUT + rotary_dim_half * stride_out_headdim,
        o1,
        mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
    )


@torch.library.custom_op("modernbert::apply_rotary", mutates_args={"x"}, device_types="cuda")
def apply_rotary(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate: bool = False,
) -> Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        raise ValueError("This kernel only supports variable length sequences")
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert x.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    output = x

    BLOCK_K = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 8 if rotary_dim <= 64 else 4

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen,  # shapes
            rotary_dim,
            seqlen_ro,
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            conjugate,
            BLOCK_M,
        )


@apply_rotary.register_fake
def _(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate: bool = False,
) -> None:
    pass


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, 3, nheads, headdim)
        total_nnz, three, nheads, headdim = qkv.shape
        assert three == 3
        if qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
            qk = qkv[:, :2].view(total_nnz, -1, headdim)
            apply_rotary(
                qk,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            q, k = qkv[:, 0, :, :], qkv[:, 1, :, :]
            apply_rotary(
                q,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            apply_rotary(
                k,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        if do.is_contiguous():
            total_nnz, three, nheads, headdim = do.shape
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            dqk = do[:, :2].view(total_nnz, -1, headdim)
            apply_rotary(
                dqk,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                conjugate=True,
            )
        else:
            dq, dk = do[:, 0, :, :], do[:, 1, :, :]
            apply_rotary(
                dq,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                conjugate=True,
            )

        return do, None, None, None, None, None, None


def apply_rotary_emb_unpad(
    qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        qkv: (total_nnz, 3, nheads, headdim) - input tensor for packed QKV.
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        inplace: if True, apply rotary embedding in-place.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class UnpaddedRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings applied directly to unpadded sequences.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        scale_base: Optional[bool] = None,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
            up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
            the cos_sin_cache wll be recomputed during the forward pass.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        """
        qkv: (total_nnz, 3, nheads, headdim)
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen: int max seq length in the batch
        Apply rotary embedding *inplace* to qkv.
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_emb_unpad(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"
