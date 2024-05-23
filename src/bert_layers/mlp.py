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

from .activation import ACT2FN
from .normalization import NORM2CLS


class BertResidualGLU(nn.Module):
    """Applies the FFN at the end of each Mosaic BERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.

    Note: Mosaic BERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a Mosaic BERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, Mosaic BERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        # self.gated_layers = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.layernorm = NORM2CLS[config.normalization](config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, intermediate_ff: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        # Activation is already computed in the attention layer
        # hidden_states = self.gated_layers(hidden_states)
        gated = intermediate_ff[:, : self.config.intermediate_size]
        non_gated = intermediate_ff[:, self.config.intermediate_size :]
        ff = self.act(gated) * non_gated
        ff = self.dropout(ff)
        # multiply by the second matrix
        ff = self.wo(ff)
        # For parralel attention, we do pre norm
        # ff = self.layernorm(ff + residual_connection)
        return ff
