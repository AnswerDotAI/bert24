# Copyright 2024 **AUTHORS_TODO**
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
import warnings
from typing import Optional
import importlib
import logging
import math

import bert_padding
from .normalization import NORM2CLS

IMPL_USE_FLASH2 = False
# Import Flash Attention 2, which supports ALiBi https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore

    installed_version = importlib.metadata.version("flash_attn")  # type: ignore
    if installed_version < "2.5.7":
        raise ImportError("newer version of flash_attn required (>= 2.5.7)")
    IMPL_USE_FLASH2 = True
    flash_attn_qkvpacked_func = None
except ImportError as e:
    warnings.warn(
        f"Failed to import flash_attn. Will use slow and memory hungry PyTorch native implementation: {e}", stacklevel=2
    )

logger = logging.getLogger(__name__)


class BertAlibiUnpadSelfAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to greatly improve throughput.
    The Flash Attention implementation used in MosaicBERT supports arbitrary attention biases (which
    we use to implement ALiBi), but does not support attention dropout. If either Flash Attention 2 is
    not installed or `config.attention_probs_dropout_prob > 0`, the implementation will default to a
    math-equivalent pytorch version, which is much slower.

    See `forward` method for additional detail.
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

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2:
            warnings.warn(
                "Unable to import flash_attn; defaulting MosaicBERT attention implementation to pytorch (this will reduce throughput when using this model)."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_in_batch: int,
        indices: torch.Tensor,
        attn_mask: torch.Tensor,
        bias: torch.Tensor,
        slopes: torch.Tensor,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are three attention implementations supported: vanilla attention with ALiBi,
        Triton Flash Attention with ALibi, and Flash Attention 2 with ALiBi

        In order to use the Triton kernel, dropout must be zero (i.e. attention_probs_dropout_prob = 0)

        The arguments are unpadded, and our implementations of attention require padded arguments,
        so we first call `pad_input`. Once we compute attention, we re-unpad our outputs for the other layers.
        The pad/unpad operations add overhead, but not sending pad tokens through ffs saves compute.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen_in_batch: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen_in_batch)
            bias: (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
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
                    max_seqlen=max_seqlen_in_batch,
                    dropout_p=self.p_dropout,
                    alibi_slopes=slopes,
                )
                attention = attention.to(orig_dtype)  # type: ignore
            else:
                attention = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen_in_batch,
                    dropout_p=self.p_dropout,
                    alibi_slopes=slopes,
                )
        else:
            qkv = bert_padding.pad_input(
                qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen_in_batch
            )  # batch, max_seqlen_in_batch, thd
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

        if not IMPL_USE_FLASH2:
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
        self.LayerNorm = NORM2CLS[config.normalization](config.hidden_size, eps=config.layer_norm_eps)
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
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
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
