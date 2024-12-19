# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# RMSNorm Implementation: Copyright Meta (from their Llama RMSNorm implementation)
# License: LLAMA 2 COMMUNITY LICENSE AGREEMENT


import inspect
import torch
import torch.nn as nn
from torch.nn import init

from .configuration_bert import FlexBertConfig

try:
    from flash_attn.ops.triton.layer_norm import RMSNorm as TritonRMSNorm
    from flash_attn.ops.triton.layer_norm import layer_norm_fn

except ImportError:
    TritonRMSNorm = None
    layer_norm_fn = None


class RMSNorm(nn.Module):
    """Llama2 RMSNorm implementation"""

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        init.ones_(self.weight)


if layer_norm_fn is not None:

    class TritonLayerNorm(nn.LayerNorm):
        def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
            return layer_norm_fn(
                x,
                self.weight,
                self.bias,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )
else:
    TritonLayerNorm = None

NORM2CLS = {
    "layernorm": nn.LayerNorm,
    "triton_layernorm": TritonLayerNorm if TritonLayerNorm is not None else nn.LayerNorm,
    "rmsnorm": RMSNorm,
    "triton_rmsnorm": TritonRMSNorm if TritonRMSNorm is not None else RMSNorm,
}


def get_norm_layer(config: FlexBertConfig, compiled_norm: bool = False) -> nn.Module:
    try:
        if compiled_norm:
            # Use non-Triton norms when compiling
            if config.normalization.startswith("triton_"):
                norm = config.normalization.replace("triton_", "")
            else:
                norm = config.normalization
        else:
            norm = config.normalization
        signature = inspect.signature(NORM2CLS[norm])
        if hasattr(config, "norm_kwargs"):
            norm_kwargs = {k: v for k, v in config.norm_kwargs.items() if k in signature.parameters}
        else:
            norm_kwargs = {}
        return NORM2CLS[norm](config.hidden_size, **norm_kwargs)
    except KeyError:
        raise ValueError(f"Invalid normalization layer type: {config.normalization}, must be one of {NORM2CLS.keys()}.")
