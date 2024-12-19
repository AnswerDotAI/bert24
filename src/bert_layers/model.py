# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# RMSNorm Implementation: Copyright Meta (from their Llama RMSNorm implementation)
# License: LLAMA 2 COMMUNITY LICENSE AGREEMENT

# Copyright 2022 Jonas Geiping
# License: MIT

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.

"""Implements Mosaic BERT, with an eye towards the Hugging Face API.

Mosaic BERT improves performance over Hugging Face BERT through the following:

1. ALiBi. This architectural change removes positional embeddings and instead encodes positional
information through attention biases based on query-key position distance. It improves the effectiveness
of training with shorter sequence lengths by enabling extrapolation to longer sequences.

2. Gated Linear Units (GLU). This architectural change replaces the FFN component of the BERT layer
to improve overall expressiveness, providing better convergence properties.

3. Flash Attention. The MosaicBERT's self-attention layer makes use of Flash Attention, which dramatically
improves the speed of self-attention. Our implementation utilizes a bleeding edge implementation that
supports attention biases, which allows us to use Flash Attention with ALiBi.

4. Unpadding. Padding is often used to simplify batching across sequences of different lengths. Standard BERT
implementations waste computation on padded tokens. MosaicBERT internally unpads to reduce unnecessary computation
and improve speed. It does this without changing how the user interfaces with the model, thereby
preserving the simple API of standard implementations.


Currently, MosaicBERT is available for masked language modeling :class:`BertForMaskedLM` and sequence
classification :class:`BertForSequenceClassification`. We aim to expand this catalogue in future releases.

See :file:`./mosaic_bert.py` for utilities to simplify working with MosaicBERT in Composer, and for example usage
of the core Mosaic BERT classes.
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers.modeling_outputs import (
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from bert_padding import index_put_first_axis

from src.bert_layers.activation import get_act_fn
from src.bert_layers.attention import (
    FlexBertPaddedAttention,
    FlexBertPaddedParallelAttention,
    FlexBertPaddedRopeAttention,
    FlexBertPaddedRopeParallelAttention,
    FlexBertUnpadAttention,
    FlexBertUnpadParallelAttention,
    FlexBertUnpadRopeAttention,
    FlexBertUnpadRopeParallelAttention,
)
from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.embeddings import (
    BertAlibiEmbeddings,
    FlexBertAbsoluteEmbeddings,
    FlexBertCompiledSansPositionEmbeddings,
    FlexBertSansPositionEmbeddings,
    get_embedding_layer,
)
from src.bert_layers.initialization import (
    ModuleType,
    TileLinear,
    TileMode,
    init_weights,
    tile_embedding,
    tile_linear,
    tile_norm,
)
from src.bert_layers.layers import (
    BertAlibiEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    FlexBertCompileUnpadPreNormLayer,
    FlexBertPaddedEncoder,
    FlexBertPaddedParallelPreNormLayer,
    FlexBertPaddedPostNormLayer,
    FlexBertPaddedPreNormLayer,
    FlexBertUnpadEncoder,
    FlexBertUnpadParallelPreNormLayer,
    FlexBertUnpadPostNormLayer,
    FlexBertUnpadPreNormLayer,
    get_encoder_layer,
)
from src.bert_layers.loss import get_loss_fn
from src.bert_layers.mlp import FlexBertGLU, FlexBertMLP, FlexBertParallelGLU
from src.bert_layers.normalization import get_norm_layer
from src.bert_layers.padding import pad_input, unpad_input

logger = logging.getLogger(__name__)


def _count_parameters(model: nn.Module, trainable: bool = True) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class BertModel(BertPreTrainedModel):
    """Overall BERT model.

    Args:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(
        self,
        config,
        add_pooling_layer: bool = True,
    ):
        super(BertModel, self).__init__(config)
        self.embeddings = BertAlibiEmbeddings(config)
        self.encoder = BertAlibiEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_all_encoded_layers: Optional[bool] = False,
        masked_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=subset_mask,
        )

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][masked_tokens_mask[attention_mask_bool][subset_idx]]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][first_col_mask[attention_mask_bool][subset_idx]]
                pooled_output = self.pooler(pool_input, pool=False)
            else:
                pooled_output = None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


###################
# Bert Heads
###################
class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0))
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


#####################
# Various Bert models
#####################


class BertForPreTraining(BertPreTrainedModel):
    # TBD: Coming in Future Commit
    pass


class BertLMHeadModel(BertPreTrainedModel):
    # TBD: Coming in Future Commit
    pass


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            warnings.warn(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        config=None,
        *inputs,
        **kwargs,
    ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # labels should be a `torch.LongTensor` of shape
        # `(batch_size, sequence_length)`. These are used for computing the
        #  masked language modeling loss.
        #
        # Indices should be in `[-100, 0, ..., config.vocab_size]` (see
        # `input_ids` docstring) Tokens with indices set to `-100` are ignored
        # (masked), the loss is only computed for the tokens with labels in `[0,
        # ..., config.vocab_size]`
        #
        # Prediction scores are only computed for masked tokens and the (bs,
        # seqlen) dimensions are flattened
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError("Must specify either input_ids or input_embeds!")

        if labels is None:
            masked_tokens_mask = None
        else:
            masked_tokens_mask = labels > 0

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            masked_tokens_mask=masked_tokens_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            loss = loss_fct(prediction_scores, labels.flatten()[masked_token_idx])

            assert input_ids is not None, "Coding error; please open an issue"
            batch, seqlen = input_ids.shape[:2]
            prediction_scores = rearrange(
                index_put_first_axis(prediction_scores, masked_token_idx, batch * seqlen),
                "(b s) d -> b s d",
                b=batch,
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertForNextSentencePrediction(BertPreTrainedModel):
    # TBD: Push in future commit
    pass


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        config=None,
        *inputs,
        **kwargs,
    ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class BertForMultipleChoice(BertPreTrainedModel):
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # In multiple choice tasks, all choices are submitted in a batch, and
        # we compute a logit for each option independently. The logits are then
        # normalized in the forward pass to get a probability distribution over
        # the choices.
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        config=None,
        *inputs,
        **kwargs,
    ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )


class BertForTokenClassification(BertPreTrainedModel):
    # TBD: Push in future commit
    pass


class BertForQuestionAnswering(BertPreTrainedModel):
    """Bert Model with a span classification head.

    This is used for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden states' output to compute `span start logits`
    and `span end logits`).
    """

    # TBD: Push in future commit


###################
# FlexBert Heads
###################


class FlexBertPredictionHead(nn.Module):
    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.head_pred_bias)
        self.act = get_act_fn(config.head_pred_act) if config.head_pred_act else nn.Identity()
        self.norm = (
            get_norm_layer(config, compiled_norm=config.compile_model) if config.head_pred_norm else nn.Identity()
        )

    def _init_weights(self, reset_params: bool = False):
        if reset_params:
            self.norm.reset_parameters()
        init_weights(self.config, self.dense, layer_dim=self.config.hidden_size, type_of_module=ModuleType.in_module)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class FlexBertPoolingHead(nn.Module):
    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.head_class_bias)
        self.act = get_act_fn(config.head_class_act) if config.head_class_act else nn.Identity()
        self.norm = get_norm_layer(config) if config.head_class_norm else nn.Identity()
        self.drop = torch.nn.Dropout(config.head_class_dropout) if config.head_class_dropout > 0 else nn.Identity()
        self.pooling_type = config.pooling_type

    def forward(self, hidden_states: torch.Tensor, pool: Optional[bool] = True) -> torch.Tensor:
        if pool:
            if self.pooling_type == "cls":
                output = hidden_states[:, 0]
            elif self.pooling_type == "mean":
                output = hidden_states.mean(dim=1)
            elif self.pooling_type == "max":
                output = hidden_states.max(dim=1)[0]
        else:
            output = hidden_states

        return self.drop(self.norm(self.act(self.dense(output))))

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.dense, self.config.hidden_size, type_of_module=ModuleType.out_module)
        if reset_params and hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    def reset_parameters(self):
        self._init_weights(reset_params=True)


###################
# FlexBert Models
###################


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    indices: Optional[torch.LongTensor] = None
    cu_seqlens: Optional[torch.LongTensor] = None
    max_seqlen: Optional[int] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    labels: Optional[torch.LongTensor] = None


@dataclass
class MaskedLMOutputZLoss(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        ce_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Cross entropy loss.
        z_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Z loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        indices (`torch.LongTensor` of shape `(batch_size,)`):
            Indices of the tokens to be masked.
    """

    loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    z_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    indices: Optional[torch.LongTensor] = None
    cu_seqlens: Optional[torch.LongTensor] = None
    max_seqlen: Optional[int] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    labels: Optional[torch.LongTensor] = None


class FlexBertPreTrainedModel(BertPreTrainedModel):
    """
    An abstract class to handle custom weights initialization of modules
    """

    def _init_module_weights(self, module: nn.Module):
        """
        Custom weight init of modules using src.bert_layers.initialization.init_weights
        Currently only supports init of embedding modules
        """
        assert isinstance(module, nn.Module)
        if isinstance(module, nn.Embedding):
            init_weights(self.config, module, type_of_module=ModuleType.emb)
        else:
            raise NotImplementedError("Custom weight init for the given module is not supported")


class FlexBertModel(FlexBertPreTrainedModel):
    """Overall BERT model.

    Args:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.embeddings = get_embedding_layer(config)
        self.encoder = get_encoder_layer(config)
        if config.final_norm:
            # if we use prenorm attention we need to add a final norm
            self.final_norm = get_norm_layer(config)
        else:
            self.final_norm = None
        self.unpad_embeddings = config.unpad_embeddings

    def post_init(self):
        self._init_weights(reset_params=False)
        self._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(input_ids, position_ids)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.final_norm is not None:
            encoder_outputs = self.final_norm(encoder_outputs)
        return encoder_outputs

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.embeddings._init_weights(reset_params=reset_params)
            self.encoder._init_weights(reset_params=reset_params)

            if reset_params and self.config.final_norm:
                self.final_norm.reset_parameters()

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def get_number_parameters(self, count_embeddings: bool = True, trainable: bool = True) -> int:
        """Returns the number of parameters in the model.

        Args:
            count_embeddings: count the parameters in the embeddings layer, excluding position embeddings.
            trainable: only count trainable parameters.
        """
        params = sum([_count_parameters(layer, trainable) for layer in self.encoder.layers])
        if count_embeddings:
            params += _count_parameters(self.embeddings, trainable)
            if hasattr(self.embeddings, "position_embeddings"):
                params -= _count_parameters(self.embeddings.position_embeddings, trainable)
        return params


class FlexBertForMaskedLM(FlexBertPreTrainedModel):
    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.bert = FlexBertModel(config)
        self.head = FlexBertPredictionHead(config)

        if config.tie_word_embeddings:
            decoder_weights = self.bert.embeddings.tok_embeddings.weight
        else:
            decoder_weights = nn.Linear(config.hidden_size, config.vocab_size, bias=False).weight
        self.decoder = nn.Linear(decoder_weights.size(1), decoder_weights.size(0), bias=config.decoder_bias)
        self.decoder.weight = decoder_weights

        self.loss_fn = nn.CrossEntropyLoss() if not hasattr(config, "loss_function") else get_loss_fn(config)
        self.fa_ce = getattr(config, "loss_function", "cross_entropy") == "fa_cross_entropy"
        self.return_z_loss = config.loss_kwargs.get("return_z_loss", False)
        self.unpad_embeddings = config.unpad_embeddings
        self.pad_logits = config.pad_logits
        self.compile_model = config.compile_model
        self.masked_prediction = config.masked_prediction

        # Initialize weights and apply final processing
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.bert._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)

            # Output weights.
            if not self.config.tie_word_embeddings:
                init_weights(self.config, self.decoder, self.config.hidden_size, type_of_module=ModuleType.final_out)

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        config=None,
        *inputs,
        **kwargs,
    ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("FlexBERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    @torch.no_grad()
    def unpad_inputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor, labels: torch.Tensor
    ):
        return unpad_input(input_ids, attention_mask, position_ids, labels)

    @torch.no_grad()
    def pad_inputs(
        self,
        inputs: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        return pad_input(
            inputs=inputs, indices=indices, batch=batch_size, seqlen=seqlen, labels=labels, ignore_index=ignore_index
        )

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # labels should be a `torch.LongTensor` of shape
        # `(batch_size, sequence_length)`. These are used for computing the
        #  masked language modeling loss.
        #
        # Indices should be in `[-100, 0, ..., config.vocab_size]` (see
        # `input_ids` docstring) Tokens with indices set to `-100` are ignored
        # (masked), the loss is only computed for the tokens with labels in `[0,
        # ..., config.vocab_size]`
        #
        # Prediction scores are only computed for masked tokens and the (bs,
        # seqlen) dimensions are flattened

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.unpad_embeddings and (indices is None and cu_seqlens is None and max_seqlen is None):
            batch_size, seq_len = input_ids.shape[:2]
            input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = self.unpad_inputs(
                input_ids, attention_mask, position_ids, labels
            )

        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.masked_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            output = output.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.loss_fn.ignore_index
            output = output[mask_tokens]
            labels = labels[mask_tokens]

        if self.compile_model:
            logits = self.compiled_head(output)
        else:
            logits = self.decoder(self.head(output))

        loss = None
        if labels is not None:
            if not self.masked_prediction:
                labels = labels.view(-1)
                logits = logits.view(labels.shape[0], -1)

            if self.return_z_loss:
                loss, z_loss = self.loss_fn(logits, labels)
                if self.pad_logits:
                    return MaskedLMOutputZLoss(
                        loss=loss,
                        ce_loss=loss.detach().clone() - z_loss,
                        z_loss=z_loss,
                        logits=self.pad_inputs(logits, indices, batch_size, seq_len)[0],
                        hidden_states=None,
                        attentions=None,
                    )
                else:
                    return MaskedLMOutputZLoss(
                        loss=loss,
                        ce_loss=loss.detach().clone() - z_loss,
                        z_loss=z_loss,
                        logits=logits,
                        hidden_states=None,
                        attentions=None,
                        indices=indices,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        labels=labels,
                    )
            else:
                loss = self.loss_fn(logits, labels)

        if self.pad_logits:
            return MaskedLMOutput(
                loss=loss,
                logits=self.pad_inputs(logits, indices, batch_size, seq_len)[0],
                hidden_states=None,
                attentions=None,
            )
        else:
            return MaskedLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                seq_len=seq_len,
                labels=labels,
            )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_number_parameters(
        self, count_embeddings: bool = True, count_decoder: bool = False, trainable: bool = True
    ) -> int:
        """Returns the number of parameters in the model.

        Args:
            count_embeddings: count the parameters in the embeddings layer, excluding position embeddings.
            count_decoder: count the parameters in the decoder layer if weights are not tied.
            trainable: only count trainable parameters.
        """
        params = self.bert.get_number_parameters(count_embeddings, trainable)
        params += _count_parameters(self.head, trainable)
        if count_decoder and not self.config.tie_word_embeddings:
            params += _count_parameters(self.decoder, trainable)
        return params


class FlexBertForSequenceClassification(FlexBertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = FlexBertModel(config)
        self.head = FlexBertPoolingHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.bert._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)
            init_weights(self.config, self.classifier, self.config.hidden_size, type_of_module=ModuleType.final_out)

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        config=None,
        *inputs,
        **kwargs,
    ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        pooled_output = self.head(output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + output
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def get_number_parameters(self, count_embeddings: bool = True, trainable: bool = True) -> int:
        """Returns the number of parameters in the model.

        Args:
            count_embeddings: count the parameters in the embeddings layer, excluding position embeddings.
            trainable: only count trainable parameters.
        """
        params = self.bert.get_number_parameters(count_embeddings, trainable)
        params += _count_parameters(self.head, trainable)
        params += _count_parameters(self.classifier, trainable)
        return params


class FlexBertForMultipleChoice(FlexBertPreTrainedModel):
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = FlexBertModel(config)
        self.head = FlexBertPoolingHead(config)

        # In multiple choice tasks, all choices are submitted in a batch, and
        # we compute a logit for each option independently. The logits are then
        # normalized in the forward pass to get a probability distribution over
        # the choices.
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.bert._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)
            init_weights(self.config, self.classifier, self.config.hidden_size, type_of_module=ModuleType.final_out)

    @classmethod
    def from_composer(
        cls,
        pretrained_checkpoint,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        config=None,
        *inputs,
        **kwargs,
    ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        pooled_output = self.head(output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + output
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

    def get_number_parameters(self, count_embeddings: bool = True, trainable: bool = True) -> int:
        """Returns the number of parameters in the model.

        Args:
            count_embeddings: count the parameters in the embeddings layer, excluding position embeddings.
            trainable: only count trainable parameters.
        """
        params = self.bert.get_number_parameters(count_embeddings, trainable)
        params += _count_parameters(self.head, trainable)
        params += _count_parameters(self.classifier, trainable)
        return params


def init_model_from_pretrained(
    pretrained_model: FlexBertModel,
    new_model: FlexBertModel,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Initialize the new model from the pretrained model.

    This method uses Gopher layer scaling and Phi-style weight tiling as selected by `mode`.
    The new model must have the same or more layers and the same or larger dimensions than the pretrained model.

    Args:
        pretrained_model (FlexBertModel): The smaller, pre-trained model
        new_model (FlexBertModel): The larger model to be initialized
        mode (Union[str, TileMode]): The Phi-style weight tiling mode to use

    This function assumes that the new_model has more layers and a larger hidden size
    than the pretrained_model, but the same vocabulary size.
    """

    # Tile embeddings
    assert isinstance(
        new_model.embeddings, type(pretrained_model.embeddings)
    ), f"Pretrained and new_model layers must be the same type, got {type(new_model.embeddings)} and {type(pretrained_model.embeddings)}"
    assert isinstance(
        new_model.embeddings,
        (FlexBertAbsoluteEmbeddings, FlexBertSansPositionEmbeddings, FlexBertCompiledSansPositionEmbeddings),
    ), f"Unsupported embedding layer type: {type(new_model.embeddings)}"

    tile_embedding(pretrained_model.embeddings.tok_embeddings, new_model.embeddings.tok_embeddings, mode=mode)
    if isinstance(pretrained_model.embeddings, FlexBertAbsoluteEmbeddings):
        tile_embedding(pretrained_model.embeddings.pos_embeddings, new_model.embeddings.pos_embeddings, mode=mode)

    if hasattr(pretrained_model.embeddings, "norm"):
        tile_norm(pretrained_model.embeddings.norm, new_model.embeddings.norm, mode=mode)

    # Tile encoder layers
    assert isinstance(
        pretrained_model.encoder, (FlexBertUnpadEncoder, FlexBertPaddedEncoder)
    ), f"Unsupported encoder layer type: {type(pretrained_model.encoder)}"
    assert isinstance(
        new_model.encoder, type(pretrained_model.encoder)
    ), f"Pretrained and new_model encoder layers must be the same type, got {type(new_model.encoder)} and {type(pretrained_model.encoder)}"

    # Calculate the layer mapping
    pretrained_layers = len(pretrained_model.encoder.layers)
    new_layers = len(new_model.encoder.layers)
    layer_mapping = [round(i * pretrained_layers / new_layers) for i in range(new_layers)]

    # Initialize layers
    for new_model_idx, pretrained_idx in enumerate(layer_mapping):
        new_model_layer = new_model.encoder.layers[new_model_idx]
        pretrained_layer = pretrained_model.encoder.layers[pretrained_idx]

        # first tile the PreNorm/PostNorm layers
        assert isinstance(
            new_model_layer, type(pretrained_layer)
        ), f"Pretrained and new_model prenorm/postnorm layers must be the same type, got {type(new_model_layer)} and {type(pretrained_layer)}"
        assert isinstance(
            new_model_layer,
            (
                FlexBertUnpadPreNormLayer,
                FlexBertCompileUnpadPreNormLayer,
                FlexBertUnpadParallelPreNormLayer,
                FlexBertUnpadPostNormLayer,
                FlexBertPaddedPreNormLayer,
                FlexBertPaddedParallelPreNormLayer,
                FlexBertPaddedPostNormLayer,
            ),
        ), f"Unsupported prenorm/postnorm layer type: {type(new_model_layer)}"

        # First tile the normalization layers
        if hasattr(pretrained_layer, "attn_norm"):
            tile_norm(pretrained_layer.attn_norm, new_model_layer.attn_norm, mode=mode)
        if hasattr(pretrained_layer, "norm"):
            tile_norm(pretrained_layer.norm, new_model_layer.norm, mode=mode)
        if hasattr(pretrained_layer, "mlp_norm"):
            tile_norm(pretrained_layer.mlp_norm, new_model_layer.mlp_norm, mode=mode)

        # Then tile the attention & mlp layers
        assert isinstance(
            new_model_layer.attn, type(pretrained_layer.attn)
        ), f"Pretrained and new_model attention layers must be the same type, got {type(new_model_layer.attn)} and {type(pretrained_layer.attn)}"

        # first try the parallel attention layers
        if isinstance(pretrained_layer, (FlexBertUnpadParallelPreNormLayer, FlexBertPaddedParallelPreNormLayer)):
            assert isinstance(
                pretrained_layer.attn,
                (
                    FlexBertUnpadParallelAttention,
                    FlexBertPaddedParallelAttention,
                    FlexBertUnpadRopeParallelAttention,
                    FlexBertPaddedRopeParallelAttention,
                ),
            ), f"Parallel prenorm layer must have parallel attention layer: {type(pretrained_layer.attn)}"
            if not isinstance(pretrained_layer.mlp, (FlexBertParallelGLU)):
                raise ValueError(f"Parallel prenorm layer must have parallel MLP layer: {type(pretrained_layer.mlp)}")
            tile_linear(
                pretrained_layer.Wqkvff,
                new_model_layer.Wqkvff,
                linear_type=TileLinear.wqkvff,
                mode=mode,
                pretrained_attn_size=pretrained_layer.attn_size,
                pretrained_mlp_size=pretrained_layer.mlp_size,
                new_attn_size=new_model_layer.attn_size,
                new_mlp_size=new_model_layer.mlp_size,
                wqkvff_is_glu=True,
            )

        # then try the fused attention layers
        elif isinstance(
            pretrained_layer.attn,
            (
                FlexBertUnpadAttention,
                FlexBertPaddedAttention,
                FlexBertUnpadRopeAttention,
                FlexBertPaddedRopeAttention,
            ),
        ):
            tile_linear(pretrained_layer.attn.Wqkv, new_model_layer.attn.Wqkv, linear_type=TileLinear.wqkv, mode=mode)
        else:
            raise ValueError(f"Unsupported attention layer type: {type(pretrained_layer.attn)}")

        # finally, tile the attention output layer
        tile_linear(pretrained_layer.attn.Wo, new_model_layer.attn.Wo, linear_type=TileLinear.default, mode=mode)

        # tile the mlp layer if the model is not using parallel attention layers
        if not isinstance(pretrained_layer.mlp, (FlexBertMLP, FlexBertGLU, FlexBertParallelGLU)):
            raise ValueError(f"Unsupported MLP layer type: {type(pretrained_layer.mlp)}")
        assert isinstance(
            new_model_layer.mlp, type(pretrained_layer.mlp)
        ), f"Pretrained and new_model mlp layers must be the same type, got {type(new_model_layer.mlp)} and {type(pretrained_layer.mlp)}"

        # already tiled the parallel glu layer if it exists, so only need to handle mlp & glu Wi
        if isinstance(pretrained_layer.mlp, FlexBertGLU):
            tile_linear(pretrained_layer.mlp.Wi, new_model_layer.mlp.Wi, linear_type=TileLinear.glu, mode=mode)
        elif isinstance(pretrained_layer.mlp, FlexBertMLP):
            tile_linear(pretrained_layer.mlp.Wi, new_model_layer.mlp.Wi, linear_type=TileLinear.default, mode=mode)
        # tile the output for both ParallelGLU and MLP/GLU
        tile_linear(pretrained_layer.mlp.Wo, new_model_layer.mlp.Wo, linear_type=TileLinear.default, mode=mode)


def init_mlm_model_from_pretrained(
    config: FlexBertConfig,
    pretrained_model: FlexBertForMaskedLM,
    new_model: FlexBertForMaskedLM,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Initialize the new model from the pretrained model.

    This method uses Gopher layer scaling and Phi-style weight tiling as selected by `mode`.
    The new model must have the same or more layers and the same or larger dimensions than the pretrained model.

    Args:
        config (FlexBertConfig): The configuration of the new_model
        pretrained_model (FlexBertForMaskedLM): The smaller, pre-trained model
        new_model (FlexBertForMaskedLM): The larger model to be initialized from the pretrained model
        mode (Union[str, TileMode]): The Phi-style weight tiling mode to use

    This function assumes that the new_model has more layers and a larger hidden size
    than the pretrained_model, but the same vocabulary size.
    """
    init_model_from_pretrained(pretrained_model.bert, new_model.bert, mode=mode)

    # TODO: uncomment this when the repo is turned into a pip installable package
    # if not isinstance(pretrained_model.head, FlexBertPredictionHead):
    #     raise ValueError(f"Pretrained model must have a prediction head: {type(pretrained_model.head)}")
    # if not isinstance(new_model.head, FlexBertPredictionHead):
    #     raise ValueError(f"New model must have a prediction head: {type(new_model.head)}")

    # tile the prediction head
    tile_linear(pretrained_model.head.dense, new_model.head.dense, linear_type=TileLinear.default, mode=mode)
    tile_norm(pretrained_model.head.norm, new_model.head.norm, mode=mode)

    # setup weight tying
    if config.tie_word_embeddings:
        new_model.decoder.weight = new_model.bert.embeddings.tok_embeddings.weight
        tile_linear(
            pretrained_model.decoder, new_model.decoder, linear_type=TileLinear.default, mode=mode, bias_only=True
        )
    else:
        tile_linear(pretrained_model.decoder, new_model.decoder, linear_type=TileLinear.default, mode=mode)
