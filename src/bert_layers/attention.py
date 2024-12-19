# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional
import importlib.metadata
import logging
import math

import bert_padding
from .configuration_bert import FlexBertConfig, maybe_add_padding
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights
import src.utils  # noqa: F401

IMPL_USE_FLASH3 = False
IMPL_USE_FLASH2 = False
try:
    from flash_attn_interface import flash_attn_varlen_func

    IMPL_USE_FLASH3 = True
except ImportError:
    pass
# Import Flash Attention 2, which supports ALiBi https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func  # type: ignore

    installed_version = importlib.metadata.version("flash_attn")  # type: ignore
    if installed_version < "2.5.7":
        raise ImportError("newer version of flash_attn required (>= 2.5.7)")
    IMPL_USE_FLASH2 = True
except ImportError:
    pass

try:
    from flash_attn.layers.rotary import RotaryEmbedding  # type: ignore
    from .rotary import UnpaddedRotaryEmbedding  # type: ignore

except ImportError:
    RotaryEmbedding = None
    UnpaddedRotaryEmbedding = None

logger = logging.getLogger(__name__)


class BertAlibiUnpadSelfAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to greatly improve throughput.
    The Flash Attention implementation used in MosaicBERT supports arbitrary attention biases (which
    we use to implement ALiBi). If either Flash Attention 2 is not installed the implementation will
    default to a math-equivalent pytorch version, which is much slower.

    See `forward` method for additional details.
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * config.hidden_size)
        self.deterministic_fa2 = getattr(config, "deterministic_fa2", False)

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2:
            warnings.warn(
                "Unable to import flash_attn; defaulting MosaicBERT attention implementation to "
                "vanilla PyTorch (this will reduce throughput when using this model)."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
        bias: torch.Tensor,
        slopes: torch.Tensor,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations: vanilla attention with ALiBi, and Flash Attention 2 with ALiBi

        The arguments are unpadded. The vanilla implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using vanilla we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)
            bias: (batch, heads, max_seqlen, max_seqlen)
            slopes: (heads) or (batch, heads)

        Returns:
            attention: (total_nnz, dim)
        """
        bs, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)

        # Option 1: Flash Attention with ALiBi
        if IMPL_USE_FLASH2:
            qkv = qkv.view(-1, 3, self.num_attention_heads, self.attention_head_size)
            assert 1 <= len(slopes.shape) <= 2, f"{slopes=}"
            assert slopes.shape[-1] == self.num_attention_heads, f"{slopes=}"

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16
                # If FA2 is supported, bfloat16 must be supported
                # as of FA2 2.4.2. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attention = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    alibi_slopes=slopes,
                )
                attention = attention.to(orig_dtype)  # type: ignore
            else:
                attention = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    alibi_slopes=slopes,
                )
        else:
            qkv = bert_padding.pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen)  # batch, max_seqlen, thd
            unpad_bs, *_ = qkv.shape
            qkv = qkv.view(unpad_bs, -1, 3, self.num_attention_heads, self.attention_head_size)
            # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
            attention_scores = torch.matmul(q, k) / math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1, 3)  # b s h d

            attention = bert_padding.unpad_input_only(attention, torch.squeeze(attn_mask) == 1)

        return attention.view(bs, dim)


# Copy of transformer's library BertSelfOutput that will not be caught by surgery methods looking for HF BERT modules.
class BertSelfOutput(nn.Module):
    """Computes the output of the attention layer.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertSelfOutput`.
    The implementation is identical. Rather than use the original module
    directly, we re-implement it here so that Mosaic BERT's modules will not
    be affected by any Composer surgery algorithm that modifies Hugging Face
    BERT modules.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = get_norm_layer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAlibiUnpadAttention(nn.Module):
    """Chains attention, Dropout, and LayerNorm for Mosaic BERT."""

    def __init__(self, config):
        super().__init__()
        self.self = BertAlibiUnpadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for scaled self-attention without padding.

        Arguments:
            input_tensor: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_s: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
            bias: None or (batch, heads, max_seqlen, max_seqlen)
            slopes: None or (batch, heads) or (heads,)
        """
        assert (bias is None) == (slopes is None), f"{bias=}, {slopes=}"
        self_output = self.self(input_tensor, cu_seqlens, max_s, indices, attn_mask, bias, slopes)
        if subset_idx is not None:
            return self.output(
                bert_padding.index_first_axis(self_output, subset_idx),
                bert_padding.index_first_axis(input_tensor, subset_idx),
            )
        else:
            return self.output(self_output, input_tensor)


class FlexBertAttentionBase(nn.Module):
    """A FlexBERT attention base class for type hints."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

    def _init_weights(self, reset_params: bool = False):
        raise NotImplementedError("This is a base class and should not be used directly.")

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")

    def extra_repr(self) -> str:
        repr = ""
        if hasattr(self, "num_attention_heads"):
            repr += f"num_attention_heads={self.num_attention_heads}"
        if hasattr(self, "attn_head_size"):
            repr += f", attn_head_size={self.attn_head_size}"
        if hasattr(self, "sliding_window"):
            repr += f", sliding_window={self.sliding_window if self.sliding_window != (-1, -1) else 'False'}"
        if hasattr(self, "use_fa2"):
            repr += f", use_fa2={self.use_fa2}"
        if hasattr(self, "deterministic_fa2"):
            repr += f", deterministic_fa2={self.deterministic_fa2}"
        return repr


class FlexBertUnpadAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional detail.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attn_head_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attn_qkv_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )
        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2 and self.use_fa2:
            logger.warn_once(
                "Unable to import flash_attn; defaulting FlexBERT attention implementation to PyTorch's"
                " SDPA kernel. This requires padding and unpadding inputs, which will add some overhead."
            )
            self.use_fa2 = False
        if not self.use_fa2:
            if not self.use_sdpa_attn_mask:
                logger.warn_once(
                    "SDPA attention is being used without an attention mask. Including padding in the "
                    " attention calculation may cause differences from the Flash Attention implementation."
                )
            else:
                logger.warn_once(
                    "SDPA attention with an attention mask doesn't use the Flash Attention kernel and will"
                    " use more memory during the backward pass. Use the FA2 backend for linear memory scaling"
                    " with sequence length."
                )
            if self.sliding_window[0] > 0:
                raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wqkv,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported: PyTorch's SDPA attention and Flash Attention 2.

        The arguments are unpadded. The SDPA implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using SDPA we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)

        Returns:
            attention: (total_nnz, dim)
        """
        bs, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)

        if self.use_fa2:
            qkv = qkv.view(-1, 3, self.num_attention_heads, self.attn_head_size)

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
            attn = attn.view(bs, dim)
        else:
            qkv = bert_padding.pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen)  # batch, max_seqlen, thd
            unpad_bs, seqlen, _ = qkv.shape

            qkv = qkv.view(unpad_bs, -1, 3, self.num_attention_heads, self.attn_head_size)
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # b h s d
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(unpad_bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            )
            attn = attn.transpose(1, 2).view(unpad_bs, -1, dim)  # b s h d
            attn = bert_padding.unpad_input_only(attn, torch.squeeze(attn_mask) == 1)

        return self.out_drop(self.Wo(attn))


class FlexBertUnpadParallelAttention(FlexBertAttentionBase):
    """Computes the output of the multi-headed self parallel attention on a batch of unpadded sequences

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional detail.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )
        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2 and self.use_fa2:
            logger.warn_once(
                "Unable to import flash_attn; defaulting FlexBERT attention implementation to PyTorch's"
                " SDPA kernel. This requires padding and unpadding inputs, which will add some overhead."
            )
            self.use_fa2 = False
        if not self.use_fa2:
            if not self.use_sdpa_attn_mask:
                logger.warn_once(
                    "SDPA attention is being used without an attention mask. Including padding in the "
                    " attention calculation may cause differences from the Flash Attention implementation."
                )
            else:
                logger.warn_once(
                    "SDPA attention with an attention mask doesn't use the Flash Attention kernel and will"
                    " use more memory during the backward pass. Use the FA2 backend for linear memory scaling"
                    " with sequence length."
                )
            if self.sliding_window[0] > 0:
                raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported: PyTorch's SDPA attention and Flash Attention 2.

        The arguments are unpadded. The SDPA implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using SDPA we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            qkv: (total_nnz, 3 * dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)

        Returns:
            attention: (total_nnz, dim)
        """
        bs = qkv.shape[0]
        dim = self.hidden_size
        if self.use_fa2:
            qkv = qkv.view(-1, 3, self.num_attention_heads, self.attn_head_size)

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
            attn = attn.view(bs, dim)
        else:
            qkv = bert_padding.pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen)  # batch, max_seqlen, thd
            unpad_bs, seqlen, _ = qkv.shape

            qkv = qkv.view(unpad_bs, -1, 3, self.num_attention_heads, self.attn_head_size)
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # b h s d
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(unpad_bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            )
            attn = attn.transpose(1, 2).view(unpad_bs, -1, dim)  # b s h d
            attn = bert_padding.unpad_input_only(attn, torch.squeeze(attn_mask) == 1)

        return self.out_drop(self.Wo(attn.view(bs, dim)))


class FlexBertPaddedAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of padded sequences.

    This module supports two attention implementations:
    1. Flash Attention 2 (if installed), which improves throughput.
    2. PyTorch's scaled_dot_product_attention.

    See `forward` method for additional detail.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attn_head_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attn_qkv_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )
        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        if not IMPL_USE_FLASH2 and self.use_fa2:
            self.use_fa2 = False
        if self.use_fa2 and self.use_sdpa_attn_mask:
            logger.warn_once(
                "Flash Attention 2 does not support attention masks. Use unpadded attention "
                "the equivalent functionality of masking out padding tokens."
            )
        if not self.use_fa2 and self.sliding_window[0] > 0:
            raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wqkv,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported:
        Flash Attention 2 and PyTorch's scaled_dot_product_attention.

        Args:
            hidden_states: (batch, seqlen, dim)
            attn_mask: (batch, seqlen)

        Returns:
            attention: (batch, seqlen, dim)
        """
        bs, seqlen, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)

        if self.use_fa2:
            qkv = qkv.view(bs, seqlen, 3, self.num_attention_heads, self.attn_head_size)

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
        else:
            qkv = qkv.view(bs, seqlen, 3, self.num_attention_heads, self.attn_head_size)

            q, k, v = qkv.transpose(3, 1).unbind(dim=2)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            ).transpose(1, 2)

        attn = attn.view(bs, seqlen, dim)
        return self.out_drop(self.Wo(attn))


class FlexBertUnpadRopeAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attn_head_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attn_qkv_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        if config.rotary_emb_dim is None:
            config.rotary_emb_dim = self.attn_head_size

        rotary_base = config.rotary_emb_base
        rotary_dim = config.rotary_emb_dim
        if self.sliding_window != (-1, -1):
            if config.local_attn_rotary_emb_base != -1:
                rotary_base = config.local_attn_rotary_emb_base
            if config.local_attn_rotary_emb_dim is not None:
                rotary_dim = config.local_attn_rotary_emb_dim

        assert UnpaddedRotaryEmbedding is not None, "rotary_emb is not installed"
        self.rotary_emb = UnpaddedRotaryEmbedding(
            dim=rotary_dim,
            base=rotary_base,
            scale_base=config.rotary_emb_scale_base,  # If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
            interleaved=config.rotary_emb_interleaved,
        )

        self.use_fa2 = config.use_fa2
        # flash attention 3 only supports global attention
        self.use_fa3 = config.use_fa2 and self.sliding_window == (-1, -1) and IMPL_USE_FLASH3
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2 and self.use_fa2:
            logger.warn_once(
                "Unable to import flash_attn; defaulting FlexBERT attention implementation to PyTorch's"
                " SDPA kernel. This requires padding and unpadding inputs, which will add some overhead."
            )
            self.use_fa2 = False
        if not self.use_fa2:
            if not self.use_sdpa_attn_mask:
                logger.warn_once(
                    "SDPA attention is being used without an attention mask. Including padding in the "
                    " attention calculation may cause differences from the Flash Attention implementation."
                )
            else:
                logger.warn_once(
                    "SDPA attention with an attention mask doesn't use the Flash Attention kernel and will"
                    " use more memory during the backward pass. Use the FA2 backend for linear memory scaling"
                    " with sequence length."
                )
            if self.sliding_window[0] > 0:
                raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wqkv,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported: PyTorch's SDPA attention and Flash Attention 2.

        The arguments are unpadded. The SDPA implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using SDPA we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)

        Returns:
            attention: (total_nnz, dim)
        """
        bs, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)

        # only needed for inference when we have KV cache
        seqlen_offset = 0

        # (total_seqlen, 3, nheads, headdim)
        qkv = qkv.view(-1, 3, self.num_attention_heads, self.attn_head_size)
        qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, seqlen_offset=seqlen_offset)

        if self.use_fa3:
            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)
                q, k, v = qkv.view(-1, 3, self.num_attention_heads, self.attn_head_size).unbind(dim=1)

                attn, _ = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    deterministic=self.deterministic_fa2,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                q, k, v = qkv.view(-1, 3, self.num_attention_heads, self.attn_head_size).unbind(dim=1)
                attn, _ = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    deterministic=self.deterministic_fa2,
                )
            attn = attn.view(bs, dim)
        elif self.use_fa2:
            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
            attn = attn.view(bs, dim)
        else:
            qkv = bert_padding.pad_input(
                qkv, indices, cu_seqlens.shape[0] - 1, attn_mask.shape[-1]
            )  # batch, max_seqlen, thd
            unpad_bs, seqlen, *_ = qkv.shape

            q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # b h s d
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(unpad_bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            )
            attn = attn.transpose(1, 2).view(unpad_bs, -1, dim)  # b s h d
            attn = bert_padding.unpad_input_only(attn, torch.squeeze(attn_mask) == 1)

        return self.out_drop(self.Wo(attn))


class FlexBertPaddedRopeAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of padded sequences.

    This module supports two attention implementations:
    1. Flash Attention 2 (if installed), which improves throughput.
    2. PyTorch's scaled_dot_product_attention.

    See `forward` method for additional details.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attn_head_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attn_qkv_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )

        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        if config.rotary_emb_dim is None:
            config.rotary_emb_dim = self.attn_head_size

        rotary_base = config.rotary_emb_base
        rotary_dim = config.rotary_emb_dim
        if self.sliding_window != (-1, -1):
            if config.local_attn_rotary_emb_base != -1:
                rotary_base = config.local_attn_rotary_emb_base
            if config.local_attn_rotary_emb_dim is not None:
                rotary_dim = config.local_attn_rotary_emb_dim

        assert RotaryEmbedding is not None, "rotary_emb is not installed"
        self.rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            base=rotary_base,
            scale_base=config.rotary_emb_scale_base,  # If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
            interleaved=config.rotary_emb_interleaved,
        )

        if not IMPL_USE_FLASH2 and self.use_fa2:
            self.use_fa2 = False
        if self.use_fa2 and self.use_sdpa_attn_mask:
            logger.warn_once(
                "Flash Attention 2 does not support attention masks. Use unpadded attention "
                "the equivalent functionality of masking out padding tokens."
            )
        if not self.use_fa2 and self.sliding_window[0] > 0:
            raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wqkv,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported:
        Flash Attention 2 and PyTorch's scaled_dot_product_attention.

        Args:
            hidden_states: (batch, seqlen, dim)
            attn_mask: (batch, seqlen)

        Returns:
            attention: (batch, seqlen, dim)
        """
        bs, seqlen, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)

        seqlen_offset = 0

        # Reshape to (batch, seqlen, 3, nheads, headdim)
        qkv = qkv.view(bs, seqlen, 3, self.num_attention_heads, self.attn_head_size)

        if IMPL_USE_FLASH2:
            # Apply RoPE
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=None)

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
        else:
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=None)
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            ).transpose(1, 2)

        attn = attn.view(bs, seqlen, dim)
        return self.out_drop(self.Wo(attn))


class FlexBertUnpadRopeParallelAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        if config.rotary_emb_dim is None:
            config.rotary_emb_dim = self.attn_head_size

        rotary_base = config.rotary_emb_base
        rotary_dim = config.rotary_emb_dim
        if self.sliding_window != (-1, -1):
            if config.local_attn_rotary_emb_base != -1:
                rotary_base = config.local_attn_rotary_emb_base
            if config.local_attn_rotary_emb_dim is not None:
                rotary_dim = config.local_attn_rotary_emb_dim

        assert UnpaddedRotaryEmbedding is not None, "rotary_emb is not installed"
        self.rotary_emb = UnpaddedRotaryEmbedding(
            dim=rotary_dim,
            base=rotary_base,
            scale_base=config.rotary_emb_scale_base,  # If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
            interleaved=config.rotary_emb_interleaved,
        )

        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2 and self.use_fa2:
            logger.warn_once(
                "Unable to import flash_attn; defaulting FlexBERT attention implementation to PyTorch's"
                " SDPA kernel. This requires padding and unpadding inputs, which will add some overhead."
            )
            self.use_fa2 = False
        if not self.use_fa2:
            if not self.use_sdpa_attn_mask:
                logger.warn_once(
                    "SDPA attention is being used without an attention mask. Including padding in the "
                    " attention calculation may cause differences from the Flash Attention implementation."
                )
            else:
                logger.warn_once(
                    "SDPA attention with an attention mask doesn't use the Flash Attention kernel and will"
                    " use more memory during the backward pass. Use the FA2 backend for linear memory scaling"
                    " with sequence length."
                )
            if self.sliding_window[0] > 0:
                raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported: PyTorch's SDPA attention and Flash Attention 2.

        The arguments are unpadded. The SDPA implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using SDPA we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            qkv: (total_nnz, 3 * dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)

        Returns:
            attention: (total_nnz, dim)
        """
        bs = qkv.shape[0]
        dim = self.hidden_size

        # only needed for inference when we have KV cache
        seqlen_offset = 0

        # (total_seqlen, 3, nheads, headdim)
        qkv = qkv.view(-1, 3, self.num_attention_heads, self.attn_head_size)
        qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, seqlen_offset=seqlen_offset)

        if self.use_fa2:
            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
            attn = attn.view(bs, dim)
        else:
            qkv = bert_padding.pad_input(
                qkv, indices, cu_seqlens.shape[0] - 1, attn_mask.shape[-1]
            )  # batch, max_seqlen, thd
            unpad_bs, seqlen, *_ = qkv.shape

            q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # b h s d
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(unpad_bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            )
            attn = attn.transpose(1, 2).view(unpad_bs, -1, dim)  # b s h d
            attn = bert_padding.unpad_input_only(attn, torch.squeeze(attn_mask) == 1)

        return self.out_drop(self.Wo(attn))


class FlexBertPaddedRopeParallelAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of padded sequences.

    This module supports two attention implementations:
    1. Flash Attention 2 (if installed), which improves throughput.
    2. PyTorch's scaled_dot_product_attention.

    See `forward` method for additional details.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )

        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask
        if not IMPL_USE_FLASH2 and self.use_fa2:
            self.use_fa2 = False

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        if config.rotary_emb_dim is None:
            config.rotary_emb_dim = self.attn_head_size

        rotary_base = config.rotary_emb_base
        rotary_dim = config.rotary_emb_dim
        if self.sliding_window != (-1, -1):
            if config.local_attn_rotary_emb_base != -1:
                rotary_base = config.local_attn_rotary_emb_base
            if config.local_attn_rotary_emb_dim is not None:
                rotary_dim = config.local_attn_rotary_emb_dim

        assert RotaryEmbedding is not None, "rotary_emb is not installed"
        self.rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            base=rotary_base,
            scale_base=config.rotary_emb_scale_base,  # If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
            interleaved=config.rotary_emb_interleaved,
        )

        if not IMPL_USE_FLASH2 and self.use_fa2:
            self.use_fa2 = False
        if self.use_fa2 and self.use_sdpa_attn_mask:
            logger.warn_once(
                "Flash Attention 2 does not support attention masks. Use unpadded attention "
                "the equivalent functionality of masking out padding tokens."
            )
        if not self.use_fa2 and self.sliding_window[0] > 0:
            raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported:
        Flash Attention 2 and PyTorch's scaled_dot_product_attention.

        Args:
            qkv: (batch, seqlen, 3 * dim)
            attn_mask: (batch, seqlen)

        Returns:
            attention: (batch, seqlen, dim)
        """
        bs, seqlen, _ = qkv.shape
        dim = self.hidden_size

        seqlen_offset = 0

        # Reshape to (batch, seqlen, 3, nheads, headdim)
        qkv = qkv.view(bs, seqlen, 3, self.num_attention_heads, self.attn_head_size)

        if self.use_fa2:
            # Apply RoPE
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=None)

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
        else:
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=None)
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            ).transpose(1, 2)

        attn = attn.view(bs, seqlen, dim)
        return self.out_drop(self.Wo(attn))


class FlexBertPaddedParallelAttention(FlexBertAttentionBase):
    """Performs multi-headed self attention on a batch of padded sequences.

    This module supports two attention implementations:
    1. Flash Attention 2 (if installed), which improves throughput.
    2. PyTorch's scaled_dot_product_attention.

    See `forward` method for additional detail.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attn_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.out_drop = (
            nn.Dropout(config.attn_out_dropout_prob) if config.attn_out_dropout_prob > 0.0 else nn.Identity()
        )
        self.use_fa2 = config.use_fa2
        self.deterministic_fa2 = config.deterministic_fa2
        self.use_sdpa_attn_mask = config.use_sdpa_attn_mask

        if config.global_attn_every_n_layers > 0:
            if config.sliding_window == -1:
                raise ValueError("global_attn_every_n_layers` requires `sliding_window` to be set")
            if layer_id % config.global_attn_every_n_layers != 0:
                self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)
            else:
                self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (config.sliding_window // 2, config.sliding_window // 2)

        if not IMPL_USE_FLASH2 and self.use_fa2:
            self.use_fa2 = False
        if self.use_fa2 and self.use_sdpa_attn_mask:
            logger.warn_once(
                "Flash Attention 2 does not support attention masks. Use unpadded attention "
                "the equivalent functionality of masking out padding tokens."
            )
        if not self.use_fa2 and self.sliding_window[0] > 0:
            raise ValueError("Sliding window is not implemented for the PyTorch SDPA path. Use the FA2 backend.")

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported:
        Flash Attention 2 and PyTorch's scaled_dot_product_attention.

        Args:
            qkv: (batch, seqlen, 3 * dim)
            attn_mask: (batch, seqlen)

        Returns:
            attention: (batch, seqlen, dim)
        """
        bs, seqlen, _ = qkv.shape
        dim = self.hidden_size

        if self.use_fa2:
            qkv = qkv.view(bs, seqlen, 3, self.num_attention_heads, self.attn_head_size)

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
                # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
                attn = attn.to(orig_dtype)  # type: ignore
            else:
                attn = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    window_size=self.sliding_window,
                )
        else:
            qkv = qkv.view(bs, seqlen, 3, self.num_attention_heads, self.attn_head_size)
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # b h s d
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                attn_mask=attn_mask[:, None, None, :seqlen].to(torch.bool).expand(bs, 1, seqlen, seqlen)
                if self.use_sdpa_attn_mask
                else None,
            ).transpose(1, 2)

        attn = attn.view(bs, seqlen, dim)
        return self.out_drop(self.Wo(attn))


ATTN2CLS = {
    "unpadded_base": FlexBertUnpadAttention,
    "padded_base": FlexBertPaddedAttention,
    "unpadded_parallel": FlexBertUnpadParallelAttention,
    "padded_parallel": FlexBertPaddedParallelAttention,
    "unpadded_rope": FlexBertUnpadRopeAttention,
    "padded_rope": FlexBertPaddedRopeAttention,
    "unpadded_rope_parallel": FlexBertUnpadRopeParallelAttention,
    "padded_rope_parallel": FlexBertPaddedRopeParallelAttention,
}


def get_attention_layer(config: FlexBertConfig, layer_id: Optional[int] = None) -> FlexBertAttentionBase:
    try:
        attention_layer = (
            config.initial_attention_layer
            if layer_id < config.num_initial_layers and getattr(config, "initial_attention_layer", None) is not None
            else config.attention_layer
        )
        return ATTN2CLS[maybe_add_padding(config, attention_layer)](config, layer_id=layer_id)
    except KeyError:
        if layer_id < config.num_initial_layers and getattr(config, "initial_attention_layer", None) is not None:
            raise ValueError(
                f"Invalid attention layer type: {config.initial_attention_layer=}, must be one of {ATTN2CLS.keys()}."
                f"{config.padding=} will be automatically prepended to `config.attention_layer` if unspecified."
            )
        else:
            raise ValueError(
                f"Invalid attention layer type: {config.attention_layer=}, must be one of {ATTN2CLS.keys()}. "
                f"{config.padding=} will be automatically prepended to `config.attention_layer` if unspecified."
            )
