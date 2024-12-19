# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import inspect
import torch.nn as nn
from .configuration_bert import FlexBertConfig

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

LOSS2CLS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "binary_cross_entropy": nn.BCEWithLogitsLoss,
    "mean_squared_error": nn.MSELoss,
}

if CrossEntropyLoss is not None:
    LOSS2CLS["fa_cross_entropy"] = CrossEntropyLoss


def get_loss_fn(config: FlexBertConfig) -> nn.Module:
    try:
        loss_class = LOSS2CLS[config.loss_function]
        signature = inspect.signature(loss_class)
        loss_kwargs = {k: v for k, v in config.loss_kwargs.items() if k in signature.parameters}
        return loss_class(**loss_kwargs)
    except KeyError:
        raise ValueError(f"Invalid loss function type: {config.loss_function}, must be one of {LOSS2CLS.keys()}.")
