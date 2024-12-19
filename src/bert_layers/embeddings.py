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
from typing import Optional

from .configuration_bert import FlexBertConfig
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights


class BertAlibiEmbeddings(nn.Module):
    """Construct the embeddings for words, ignoring position.

    There are no positional embeddings since we use ALiBi and token_type
    embeddings.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertEmbeddings`, but is
    modified as part of Mosaic BERT's ALiBi implementation. The key change is
    that position embeddings are removed. Position information instead comes
    from attention biases that scale linearly with the position distance
    between query and key tokens.

    This module ignores the `position_ids` input to the `forward` method.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # ALiBi doesn't use position embeddings
        if getattr(config, "token_type_embeddings", True):
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.use_token_type_embeddings = True
        else:
            self.use_token_type_embeddings = False

        self.LayerNorm = get_norm_layer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_token_type_embeddings:
            self.register_buffer(
                "token_type_ids", torch.zeros(config.max_position_embeddings, dtype=torch.long), persistent=False
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError("Must specify either input_ids or input_embeds!")
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert inputs_embeds is not None  # just for type checking
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # great! ALiBi
            pass

        # Setting the token_type_ids to the registered buffer in constructor
        # where it is all zeros, which usually occurs when it's auto-generated;
        # registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if self.use_token_type_embeddings and token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.use_token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds

        # no position embeddings! ALiBi
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FlexBertEmbeddingsBase(nn.Module):
    """A FlexBERT embeddings base class for type hints."""

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, reset_params: bool = False):
        raise NotImplementedError("This is a base class and should not be used directly.")

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertAbsoluteEmbeddings(FlexBertEmbeddingsBase):
    """Construct the embeddings with absolute positional embeddings."""

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.norm = get_norm_layer(config) if config.embed_norm else nn.Identity()
        self.drop = nn.Dropout(config.embed_dropout_prob) if config.embed_dropout_prob > 0.0 else nn.Identity()

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.tok_embeddings, type_of_module=ModuleType.emb)
        init_weights(self.config, self.position_embeddings, type_of_module=ModuleType.emb)

        if reset_params:
            if self.config.embed_norm:
                self.norm.reset_parameters()  # type: ignore

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = self.position_ids[:, 0 : input_ids.shape[1]]

        embeddings = self.tok_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.norm(embeddings + position_embeddings)
        return self.drop(embeddings)


class FlexBertCompiledSansPositionEmbeddings(FlexBertEmbeddingsBase):
    """Construct the embeddings from token embeddings without any positional embeddings."""

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.norm = get_norm_layer(config, compiled_norm=config.compile_model) if config.embed_norm else nn.Identity()
        self.drop = nn.Dropout(config.embed_dropout_prob) if config.embed_dropout_prob > 0.0 else nn.Identity()

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.tok_embeddings, type_of_module=ModuleType.emb)

        if reset_params:
            if self.config.embed_norm:
                self.norm.reset_parameters()  # type: ignore

    @torch.compile(dynamic=True)
    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class FlexBertSansPositionEmbeddings(FlexBertEmbeddingsBase):
    """Construct the embeddings from token embeddings without any positional embeddings."""

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.norm = get_norm_layer(config) if config.embed_norm else nn.Identity()
        self.drop = nn.Dropout(config.embed_dropout_prob) if config.embed_dropout_prob > 0.0 else nn.Identity()

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.tok_embeddings, type_of_module=ModuleType.emb)

        if reset_params:
            if self.config.embed_norm:
                self.norm.reset_parameters()  # type: ignore

    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


EBB2CLS = {
    "absolute_pos": FlexBertAbsoluteEmbeddings,
    "sans_pos": FlexBertSansPositionEmbeddings,
}


def get_embedding_layer(config: FlexBertConfig) -> FlexBertEmbeddingsBase:
    try:
        if config.compile_model and config.embedding_layer == "sans_pos":
            return FlexBertCompiledSansPositionEmbeddings(config)
        elif config.compile_model:
            raise ValueError(f"{config.compile_model=} only supports sans_pos embeddings.")
        return EBB2CLS[config.embedding_layer](config)
    except KeyError:
        raise ValueError(f"Invalid embeddings layer type: {config.embedding_layer=}, must be one of {EBB2CLS.keys()}.")
