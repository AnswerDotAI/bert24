# adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_rotary.py

import os
import sys

import math

import pytest
import torch
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb_torch
from flash_attn.bert_padding import pad_input, unpad_input

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bert_layers.rotary import apply_rotary_emb_unpad


is_sm8x = torch.cuda.get_device_capability("cuda") >= (8, 0)


def generate_cos_sin(seqlen, rotary_dim, device, dtype):
    assert rotary_dim % 2 == 0
    angle = torch.rand(seqlen * 2, rotary_dim // 2, device=device) * 2 * math.pi
    cos = torch.cos(angle).to(dtype=dtype)
    sin = torch.sin(angle).to(dtype=dtype)
    return cos, sin


def generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device):
    if seqlen_offsets_type == 0:
        return 0
    elif seqlen_offsets_type is int:
        return torch.randint(0, seqlen + 1, (1,)).item()
    elif seqlen_offsets_type is torch.Tensor:
        return torch.randint(0, seqlen + 1, (batch_size,), dtype=torch.int32, device=device)


def index_cos_sin(cos, sin, seqlen_offsets, seqlen):
    if isinstance(seqlen_offsets, torch.Tensor):
        batch_size = seqlen_offsets.shape[0]
        arange = rearrange(torch.arange(seqlen, device=cos.device), "s -> 1 s")
        idx = rearrange(seqlen_offsets, "b -> b 1") + arange
        cos_pt = rearrange(cos[idx.flatten()], "(b s) d -> b s d", b=batch_size)
        sin_pt = rearrange(sin[idx.flatten()], "(b s) d -> b s d", b=batch_size)
    else:
        cos_pt = cos[seqlen_offsets : seqlen_offsets + seqlen]
        sin_pt = sin[seqlen_offsets : seqlen_offsets + seqlen]
    return cos_pt, sin_pt


@pytest.mark.parametrize("dtype", ([torch.float16] if not is_sm8x else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("seqlen_offsets_type", [0, int, torch.Tensor])
@pytest.mark.parametrize("rotary_fraction", [1, 0.5])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_emb_unpad(interleaved, rotary_fraction, seqlen_offsets_type, dtype):
    rtol = 1e-3
    batch_size = 32
    nheads = 4
    seqlen = 2048
    headdim = 128
    device = "cuda"
    rotary_dim = int(rotary_fraction * headdim)
    torch.manual_seed(42)

    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    qkv_pt = qkv.detach().clone().requires_grad_()

    lengths = torch.randint(max(1, seqlen - 20), seqlen + 1, (batch_size, 1), device=device)
    padding_mask = rearrange(torch.arange(seqlen, device=device), "s -> 1 s") < lengths

    qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_input(qkv, padding_mask)
    qkv_unpad = qkv_unpad.requires_grad_()

    cos, sin = generate_cos_sin(seqlen, rotary_dim, device, dtype)
    seqlen_offsets = generate_seqlen_offsets(seqlen_offsets_type, batch_size, seqlen, device)

    qkv_unpad = qkv_unpad.view(-1, 3, nheads, headdim)
    out_unpad = apply_rotary_emb_unpad(
        qkv_unpad,
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        interleaved=interleaved,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )

    out = pad_input(out_unpad, indices, batch_size, seqlen)
    out = out.requires_grad_()

    cos_pt, sin_pt = index_cos_sin(cos, sin, seqlen_offsets, seqlen)

    q_pt = apply_rotary_emb_torch(qkv_pt[:, :, 0].float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved).to(
        dtype=dtype
    )

    k_pt = apply_rotary_emb_torch(qkv_pt[:, :, 1].float(), cos_pt.float(), sin_pt.float(), interleaved=interleaved).to(
        dtype=dtype
    )

    out_pt = torch.stack([q_pt, k_pt, qkv_pt[:, :, 2]], dim=2)
    out_pt = out_pt.masked_fill(rearrange(~padding_mask, "b s -> b s 1 1 1"), 0.0)

    print(f"Output max diff: {(out - out_pt).abs().max().item()}")
    atol = ((out_pt + 0.3 - 0.3) - out_pt).abs().max().item()
    assert torch.allclose(out, out_pt, rtol=rtol, atol=2 * atol)

    g = torch.randn_like(out)
    g_pt = g.clone()  # Since inplace=True, we modify the gradient inplace
    out.backward(g)
    out_pt.backward(g_pt)

    print(f"Grad max diff: {(qkv.grad - qkv_pt.grad).abs().max().item()}")
    atol = ((qkv_pt.grad + 0.3 - 0.3) - qkv_pt.grad).abs().max().item()
    assert torch.allclose(qkv.grad, qkv_pt.grad, rtol=rtol, atol=2 * atol)
