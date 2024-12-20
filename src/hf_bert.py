# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face BERT wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Optional

from composer.metrics.nlp import BinaryF1Score, LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError
from torchmetrics import MeanSquaredError
from torchmetrics.classification.accuracy import MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

__all__ = ["create_hf_bert_mlm", "create_hf_bert_classification"]


def create_hf_bert_mlm(
    pretrained_model_name: str = "bert-base-uncased",
    use_pretrained: Optional[bool] = False,
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'bert-base-uncased'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": ["BertForMaskedLM"],
              "attention_probs_dropout_prob": 0.1,
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

    To create a |:hugging_face:| BERT model for Masked Language Model pretraining:

     .. testcode::

         from src.hf_bert import create_hf_bert_mlm
         model = create_hf_bert_mlm()
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group="nlp", conda_package="transformers"
        ) from e

    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    if use_pretrained:
        assert (
            transformers.AutoModelForMaskedLM.from_pretrained is not None
        ), "AutoModelForMaskedLM has from_pretrained method"
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config
        )
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config
        )
        assert (
            transformers.AutoModelForMaskedLM.from_config is not None
        ), "AutoModelForMaskedLM has from_config method"
        model = transformers.AutoModelForMaskedLM.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        # vocab_size no longer in composer
        LanguageCrossEntropy(ignore_index=-100),
        MaskedAccuracy(ignore_index=-100),
    ]
    return HuggingFaceModel(
        model=model, tokenizer=tokenizer, use_logits=True, metrics=metrics
    )


def create_hf_bert_classification(
    num_labels: int,
    pretrained_model_name: str = "bert-base-uncased",
    use_pretrained: Optional[bool] = False,
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    custom_eval_metrics: Optional[list] = [],
    multiple_choice: Optional[bool] = False,
):
    """BERT model based on |:hugging_face:| Transformers.

    For more information, see `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        num_labels (int): The number of classes in the task (``1`` indicates regression). Default: ``2``.
        pretrained_model_name (str): Name of the Hugging Face model to instantiate. Default: ``'bert-base-uncased'``.
        use_pretrained (bool, optional): Whether to initialize the model with the pretrained weights. Default: ``False``.
        model_config (dict, optional): The settings used to create a Hugging Face BertConfig. BertConfig is used to specify the
            architecture of a Hugging Face model.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing. Default: ``False``.
        custom_eval_metrics (list, optional): Classes of custom metrics to evaluate the model. Default: ``[]``.
        multiple_choice (bool, optional): Whether the model is used for multiple choice tasks. Default: ``False``.

        .. code-block::

            {
              "_name_or_path": "bert-base-uncased",
              "architectures": [
                "BertForSequenceClassification"
              ],
              "attention_probs_dropout_prob": 0.1,
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

    Note:
        This function can be used to construct a BERT model for regression by setting ``num_labels == 1``.
        This will have two noteworthy effects. First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics will be :class:`~torchmetrics.MeanSquaredError` and :class:`~torchmetrics.SpearmanCorrCoef`.

        For the classification case (when ``num_labels > 1``), the training loss is :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.MulticlassAccuracy` and :class:`~torchmetrics.MatthewsCorrCoef`, as well as :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group="nlp", conda_package="transformers"
        ) from e

    if not model_config:
        model_config = {}

    model_config["num_labels"] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    auto_model_cls = transformers.AutoModelForSequenceClassification

    if multiple_choice:
        auto_model_cls = transformers.AutoModelForMultipleChoice

    if use_pretrained:
        assert (
            auto_model_cls.from_pretrained is not None
        ), f"{auto_model_cls.__name__} has from_pretrained method"
        model = auto_model_cls.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, **model_config
        )
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config
        )
        assert (
            auto_model_cls.from_config is not None
        ), f"{auto_model_cls.__name__} has from_config method"
        model = auto_model_cls.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            MulticlassAccuracy(num_classes=num_labels, average="micro"),
            MatthewsCorrCoef(task="multiclass", num_classes=model.config.num_labels),
        ]
        if num_labels == 2:
            metrics.append(BinaryF1Score())
            
    if model_config.get('problem_type', '') == 'multi_label_classification':
        metrics = [
            MultilabelAccuracy(num_labels=num_labels, average="micro"),
        ]

    return HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=[
            *metrics,
            *[metric_cls() for metric_cls in custom_eval_metrics],
        ],
    )
