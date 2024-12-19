# Copyright 2024 onwards Answer.AI, LightOn, and contributors
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

import bert_padding

from .activation import get_act_fn
from .attention import FlexBertAttentionBase, BertAlibiUnpadAttention, get_attention_layer
from .mlp import FlexBertMLPBase, BertResidualGLU, get_mlp_layer
from .configuration_bert import FlexBertConfig, maybe_add_padding
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights


class BertAlibiLayer(nn.Module):
    """Composes the Mosaic BERT attention and FFN blocks into a single layer."""

    def __init__(self, config):
        super().__init__()
        self.attention = BertAlibiUnpadAttention(config)
        self.mlp = BertResidualGLU(config)

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

    def __init__(self, config):
        super().__init__()
        layer = BertAlibiLayer(config)
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
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_act_fn(config.head_pred_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = get_norm_layer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FlexBertLayerBase(nn.Module):
    """A FlexBERT Layer base class for type hints."""

    attn: FlexBertAttentionBase
    mlp: FlexBertMLPBase

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

    def _init_weights(self, reset_params: bool = False):
        if hasattr(self, "attn"):
            self.attn._init_weights(reset_params)
        if hasattr(self, "mlp"):
            self.mlp._init_weights(reset_params)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertCompileUnpadPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config, compiled_norm=config.compile_model)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.compile_model = config.compile_model

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), cu_seqlens, max_seqlen, indices, attn_mask)
        return attn_out + self.compiled_mlp(attn_out)


class FlexBertUnpadPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), cu_seqlens, max_seqlen, indices, attn_mask)
        return attn_out + self.mlp(self.mlp_norm(attn_out))


class FlexBertUnpadParallelPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT parallel attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn_size = config.hidden_size * 3
        self.mlp_size = config.intermediate_size * 2
        # Compute QKV and FF outputs at once
        self.Wqkvff = nn.Linear(config.hidden_size, self.attn_size + self.mlp_size, bias=config.attn_qkv_bias)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params and hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

        init_weights(
            self.config,
            self.Wqkvff,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        # Compute QKV and FF outputs at once and split them
        qkv, intermediate_ff = self.Wqkvff(self.norm(hidden_states)).split([self.attn_size, self.mlp_size], dim=1)
        return hidden_states + self.attn(qkv, cu_seqlens, max_seqlen, indices, attn_mask) + self.mlp(intermediate_ff)


class FlexBertPaddedPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (batch, max_seqlen, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), attn_mask)
        return attn_out + self.mlp(self.mlp_norm(attn_out))


class FlexBertPaddedParallelPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn_size = config.hidden_size * 3
        self.mlp_size = config.intermediate_size * 2
        # Compute QKV and FF outputs at once
        self.Wqkvff = nn.Linear(config.hidden_size, self.attn_size + self.mlp_size, bias=config.attn_qkv_bias)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.norm.reset_parameters()

        init_weights(
            self.config,
            self.Wqkvff,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (batch, max_seqlen, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        # Compute QKV and FF outputs at once and split them
        qkv, intermediate_ff = self.Wqkvff(self.norm(hidden_states)).split([self.attn_size, self.mlp_size], dim=2)
        return hidden_states + self.attn(qkv, attn_mask) + self.mlp(intermediate_ff)


class FlexBertUnpadPostNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using post-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.attn_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = self.attn_norm(hidden_states + self.attn(hidden_states, cu_seqlens, max_seqlen, indices, attn_mask))
        return self.mlp_norm(attn_out + self.mlp(attn_out))


class FlexBertPaddedPostNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using post-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.attn_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (batch, max_seqlen, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = self.attn_norm(hidden_states + self.attn(hidden_states, attn_mask))
        return self.mlp_norm(attn_out + self.mlp(attn_out))


LAYER2CLS = {
    "unpadded_prenorm": FlexBertUnpadPreNormLayer,
    "unpadded_compile_prenorm": FlexBertCompileUnpadPreNormLayer,
    "unpadded_parallel_prenorm": FlexBertUnpadParallelPreNormLayer,
    "unpadded_postnorm": FlexBertUnpadPostNormLayer,
    "padded_prenorm": FlexBertPaddedPreNormLayer,
    "padded_parallel_prenorm": FlexBertPaddedParallelPreNormLayer,
    "padded_postnorm": FlexBertPaddedPostNormLayer,
}


def get_bert_layer(config: FlexBertConfig, layer_id: Optional[int] = None) -> FlexBertLayerBase:
    try:
        bert_layer = (
            config.initial_bert_layer
            if layer_id < config.num_initial_layers and getattr(config, "initial_bert_layer", None) is not None
            else config.bert_layer
        )
        bert_layer = maybe_add_padding(config, bert_layer)
        if config.compile_model and bert_layer == "unpadded_prenorm":
            bert_layer = "unpadded_compile_prenorm"
        return LAYER2CLS[bert_layer](config, layer_id=layer_id)
    except KeyError:
        if layer_id < config.num_initial_layers and getattr(config, "initial_bert_layer", None) is not None:
            raise ValueError(
                f"Invalid BERT layer type: {config.initial_bert_layer=}, must be one of {LAYER2CLS.keys()}."
                f"{config.padding=} will be automatically prepended to `config.bert_layer` if unspecified."
            )
        else:
            raise ValueError(
                f"Invalid BERT layer type: {config.bert_layer=}, must be one of {LAYER2CLS.keys()}. "
                f"{config.padding=} will be automatically prepended to `config.bert_layer` if unspecified."
            )


class FlexBertEncoderBase(nn.Module):
    """A FlexBERT base class for type hints."""

    layers: nn.ModuleList

    def _init_weights(self, reset_params: bool = False):
        if hasattr(self, "layers"):
            for layer in self.layers:
                layer._init_weights(reset_params=reset_params)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertUnpadEncoder(FlexBertEncoderBase):
    """A stack of BERT layers providing the backbone of FlexBERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertAlibiEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.layers = nn.ModuleList([get_bert_layer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.num_attention_heads = config.num_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        if indices is None and cu_seqlens is None and max_seqlen is None:
            attention_mask_bool = attention_mask.bool()
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen = bert_padding.unpad_input(
                hidden_states, attention_mask_bool
            )

            for layer_module in self.layers:
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    indices,
                    attn_mask=attention_mask,
                )

            return bert_padding.pad_input(hidden_states, indices, batch, seqlen)
        else:
            for layer_module in self.layers:
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    indices,
                    attn_mask=attention_mask,
                )
            return hidden_states


class FlexBertPaddedEncoder(FlexBertEncoderBase):
    """A stack of BERT layers providing the backbone of FlexBERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertAlibiEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.layers = nn.ModuleList([get_bert_layer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.num_attention_heads = config.num_attention_heads

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attn_mask=attention_mask)

        return hidden_states


ENC2CLS = {
    "unpadded_base": FlexBertUnpadEncoder,
    "padded_base": FlexBertPaddedEncoder,
}


def get_encoder_layer(config: FlexBertConfig) -> FlexBertEncoderBase:
    try:
        return ENC2CLS[maybe_add_padding(config, config.encoder_layer)](config)
    except KeyError:
        raise ValueError(
            f"Invalid encoder layer type: {config.encoder_layer=}, must be one of {ENC2CLS.keys()}. "
            f"{config.padding=} will be automatically prepended to `config.encoder_layer` if unspecified."
        )
