# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a MosaicBERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

import os
import sys
from typing import Optional

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bert_layers as bert_layers_module
import configuration_bert as configuration_bert_module
import transformers
from composer.metrics.nlp import (BinaryF1Score, LanguageCrossEntropy,
                                  MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

all = ['create_mosaic_bert_mlm', 'create_mosaic_bert_classification']


def create_mosaic_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
                           model_config: Optional[dict] = None,
                           tokenizer_name: Optional[str] = None,
                           gradient_checkpointing: Optional[bool] = False,
                           pretrained_checkpoint: Optional[str] = None):
    """Mosaic BERT masked language model based on |:hugging_face:| Transformers.

    For more information, see
    `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a MosaicBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided, the state dictionary
            stored at `pretrained_checkpoint` will be loaded into the model
            after initialization. Default: ``None``.

    .. code-block::

        {
        "_name_or_path": "bert-base-uncased",
        "alibi_starting_size": 512,
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.16.0",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
        }

    To create a MosaicBERT model for Masked Language Model pretraining:

     .. testcode::

         from src.mosaic import create_mosaic_bert_mlm
         model = create_mosaic_bert_mlm()
    """
    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    config = configuration_bert_module.BertConfig.from_pretrained(
        pretrained_model_name, **model_config)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = bert_layers_module.BertForMaskedLM.from_composer(
            pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = bert_layers_module.BertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name)

    metrics = [
        # vocab size no longer arg in composer
        LanguageCrossEntropy(ignore_index=-100),
        MaskedAccuracy(ignore_index=-100)
    ]

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model


def create_mosaic_bert_classification(
        num_labels: int,
        pretrained_model_name: str = 'bert-base-uncased',
        model_config: Optional[dict] = None,
        tokenizer_name: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False,
        pretrained_checkpoint: Optional[str] = None):
    """Mosaic BERT classification model based on |:hugging_face:| Transformers.

    For more information, see `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a MosaicBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        num_labels (int): The number of classes in the classification task.
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided,
            the state dictionary stored at `pretrained_checkpoint` will be
            loaded into the model after initialization. Default: ``None``.

    .. code-block::
        {
            "_name_or_path": "bert-base-uncased",
            "alibi_starting_size": 512,
            "architectures": [
            "BertForSequenceClassification
            ],
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1",
            "2": "LABEL_2"
            },
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2
            },
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.16.0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }

    To create a MosaicBERT model for classification:

     .. testcode::
        from mosaic_bert import create_mosaic_bert_classification
        model = create_mosaic_bert_classification(num_labels=3) # if the task has three classes.

    Note:
        This function can be used to construct a BERT model for regression by
        setting ``num_labels == 1``. This will have two noteworthy effects.
        First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics
        will be :class:`~torchmetrics.MeanSquaredError` and
        :class:`~torchmetrics.SpearmanCorrCoef`. For the classifcation case
        (when ``num_labels > 1``), the training loss is
        :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.MulticlassAccuracy` and
        :class:`~torchmetrics.MatthewsCorrCoef`, as well as
        :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    if not model_config:
        model_config = {}

    # By default, turn off attention dropout in MosaicBERT
    # Flash Attention 2 supports dropout in the attention module
    # while our previous Triton Flash Attention layer only works with
    # attention_probs_dropout_prob = 0.
    if 'attention_probs_dropout_prob' not in model_config:
        model_config['attention_probs_dropout_prob'] = 0.0

    # Use `alibi_starting_size` to determine how large of an alibi tensor to
    # create when initializing the model. You should be able to ignore
    # this parameter in most cases.
    if 'alibi_starting_size' not in model_config:
        model_config['alibi_starting_size'] = 512

    model_config['num_labels'] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    config, unused_kwargs = transformers.AutoConfig.from_pretrained(
        pretrained_model_name, return_unused_kwargs=True, **model_config)
    # This lets us use non-standard config fields (e.g. `starting_alibi_size`)
    config.update(unused_kwargs)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = bert_layers_module.BertForSequenceClassification.from_composer(
            pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = bert_layers_module.BertForSequenceClassification(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name)

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            MulticlassAccuracy(num_classes=num_labels, average='micro'),
            MatthewsCorrCoef(task='multiclass',
                             num_classes=model.config.num_labels)
        ]
        if num_labels == 2:
            metrics.append(BinaryF1Score())

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
