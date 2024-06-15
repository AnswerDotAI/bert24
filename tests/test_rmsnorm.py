# Test modified from attorch
# Copyright (c) 2023 Borna Ahmadzadeh
# SPDX-License-Identifier: MIT


from typing import Tuple

import pytest
import torch
from torch.cuda.amp import autocast

from src.bert_layers.normalization import RMSNorm
from src.bert_layers.triton.rmsnorm import TritonRMSNorm


def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0, max_error_rate=None, name=""):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if max_error_rate is not None:
        total_elements = torch.prod(torch.tensor(a.shape))
        if error_count > total_elements * max_error_rate and error_count > max_error_count:
            print(f"{name}Too many values not close: assert {error_count} < {total_elements * max_error_rate}")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    elif error_count > max_error_count:
        print(f"{name}Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


is_sm8x = torch.cuda.get_device_capability("cuda") >= (8, 0)


@pytest.mark.parametrize("shape", [(128,), (1024,), (2048, 768), (2048, 2048), (16, 256, 512)])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16] if not is_sm8x else [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("low_precision", [True, False])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.triton
def test_rmsnorm_layer(
    shape: Tuple[int, ...],
    eps: float,
    dtype: bool,
    low_precision: bool,
    amp: bool,
) -> None:
    if dtype is torch.float16 and not amp:
        pytest.skip("Skipping fp16 test without autocast enabled")
    if dtype == torch.float32 and low_precision:
        pytest.skip("Skipping fp32 test with low precision enabled")

    if dtype == torch.float32:
        atol, rtol = 2e-5, 2e-5  # tests don't always pass at the usual 1e-6, 1e-5, small percentage of params are off
    elif dtype == torch.bfloat16:
        atol, rtol = (5e-3, 5e-2) if low_precision else (1e-3, 1e-2)
    elif dtype == torch.float16:
        atol, rtol = (5e-4, 5e-3) if low_precision else (1e-4, 1e-3)

    triton_input = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    pytorch_input = triton_input.detach().clone().requires_grad_(True)

    triton_rmsnorm = TritonRMSNorm(shape[-1], eps=eps, device="cuda", dtype=dtype, low_precision=low_precision)
    pytorch_rmsnorm = RMSNorm(shape[-1], eps=eps, device="cuda", dtype=dtype, low_precision=low_precision)

    with autocast(enabled=amp, dtype=dtype):
        triton_output = triton_rmsnorm(triton_input)
        pytorch_output = pytorch_rmsnorm(pytorch_input)

    assert (
        triton_output.dtype == pytorch_output.dtype
    ), f"dtypes do not match, {triton_output.dtype=}, {pytorch_output.dtype=}"
    assert triton_output.dtype == dtype, f"dtypes do not match, {triton_output.dtype=}, {dtype=}"
    torch.testing.assert_close(triton_output, pytorch_output, rtol=rtol, atol=atol)

    grad = torch.rand_like(triton_output)
    triton_output.backward(grad)
    pytorch_output.backward(grad)

    assert_most_approx_close(
        triton_rmsnorm.weight.grad,
        pytorch_rmsnorm.weight.grad,
        rtol=rtol,
        atol=atol,
        max_error_rate=0.06 if low_precision else 0,
        name=f"weight gradients do not match {shape=}, {dtype=}, {amp=}, {low_precision=}, {eps=}",
    )

    assert_most_approx_close(
        triton_input.grad,
        pytorch_input.grad,
        rtol=rtol,
        atol=atol,
        max_error_rate=0.06 if low_precision else 0,
        name=f"input gradients do not match {shape=}, {dtype=}, {amp=}, {low_precision=}, {eps=}",
    )
