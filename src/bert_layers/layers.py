# Copyright 2024 **AUTHORS_TODO**
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.


import copy
import math
import warnings
from typing import Optional, Union, List

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

import bert_padding
from .attention import BertAlibiUnpadAttention
from .mlp import BertResidualGLU
from .norm import RMSNorm


class BertAlibiLayer(nn.Module):
    """Composes the Mosaic BERT attention and FFN blocks into a single layer."""

    def __init__(self, config, use_rmsnorm: bool = True, use_silu: bool = True):
        super().__init__()
        self.attention = BertAlibiUnpadAttention(config, use_rmsnorm=use_rmsnorm)
        self.mlp = BertResidualGLU(config, use_rmsnorm=use_rmsnorm, use_silu=use_silu)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlen: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
            slopes: None or (batch, heads) or (heads,)
        """
        assert (bias is None) == (slopes is None), f"{bias=}, {slopes=}"
        attention_output = self.attention(
            hidden_states, cu_seqlens, seqlen, subset_idx, indices, attn_mask, bias, slopes
        )
        layer_output = self.mlp(attention_output)
        return layer_output


class BertAlibiEncoder(nn.Module):
    """A stack of BERT layers providing the backbone of Mosaic BERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertAlibiEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config, use_rmsnorm: bool = True, use_silu: bool = True):
        super().__init__()
        layer = BertAlibiLayer(config, use_rmsnorm=use_rmsnorm, use_silu=use_silu)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.num_attention_heads = config.num_attention_heads

        # The alibi mask will be dynamically expanded if it is too small for
        # the input the model receives. But it generally helps to initialize it
        # to a reasonably large size to help pre-allocate CUDA memory.
        # The default `alibi_starting_size` is 512.
        self._current_alibi_size = int(config.alibi_starting_size)
        self.alibi = torch.zeros((1, self.num_attention_heads, self._current_alibi_size, self._current_alibi_size))
        self.rebuild_alibi_tensor(size=config.alibi_starting_size)

    def rebuild_alibi_tensor(self, size: int, device: Optional[Union[torch.device, str]] = None):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:
            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        self.slopes = slopes
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: Optional[bool] = True,
        subset_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]
        # Unpad inputs and mask. It will remove tokens that are padded.
        # Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens.
        # Then unpadding performs the following compression of the inputs:
        # hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
        hidden_states, indices, cu_seqlens, _ = bert_padding.unpad_input(hidden_states, attention_mask_bool)

        # Add alibi matrix to extended_attention_mask
        if self._current_alibi_size < seqlen:
            # Rebuild the alibi tensor when needed
            warnings.warn(f"Increasing alibi size from {self._current_alibi_size} to {seqlen}")
            self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
        elif self.alibi.device != hidden_states.device:
            # Device catch-up
            self.alibi = self.alibi.to(hidden_states.device)
            self.slopes = self.slopes.to(hidden_states.device)  # type: ignore
        alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
        attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
        alibi_attn_mask = attn_bias + alibi_bias

        all_encoder_layers = []
        if subset_mask is None:
            for layer_module in self.layer:
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    seqlen,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    bias=alibi_attn_mask,
                    slopes=self.slopes,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # Pad inputs and mask. It will insert back zero-padded tokens.
            # Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens.
            # Then padding performs the following de-compression:
            #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
            hidden_states = bert_padding.pad_input(hidden_states, indices, batch, seqlen)
        else:
            for i in range(len(self.layer) - 1):
                layer_module = self.layer[i]
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    seqlen,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    bias=alibi_attn_mask,
                    slopes=self.slopes,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool], as_tuple=False).flatten()
            hidden_states = self.layer[-1](
                hidden_states,
                cu_seqlens,
                seqlen,
                subset_idx=subset_idx,
                indices=indices,
                attn_mask=attention_mask,
                bias=alibi_attn_mask,
                slopes=self.slopes,
            )

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, pool: Optional[bool] = True) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config, use_rmsnorm: bool = True):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = (
            RMSNorm(config.hidden_size, eps=1e-12) if use_rmsnorm else nn.LayerNorm(config.hidden_size, eps=1e-12)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states