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
from typing import Optional

from .normalization import NORM2CLS


class BertAlibiEmbeddings(nn.Module):
    """Construct the embeddings for words, ignoring position.

    There are no positional embeddings since we use ALiBi and token_type
    embeddings.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertAlibiEmbeddings`, but is
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
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = NORM2CLS[config.normalization](config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                assert isinstance(self.token_type_ids, torch.LongTensor)
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded  # type: ignore
            else:
                token_type_ids = torch.zeros(
                    input_shape,  # type: ignore
                    dtype=torch.long,
                    device=self.word_embeddings.device,
                )  # type: ignore  # yapf: disable

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # no position embeddings! ALiBi
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
