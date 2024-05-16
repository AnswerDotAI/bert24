# Copyright 2024 **AUTHORS_TODO**
# License: Apache-2.0

# Copyright 2020 The HuggingFace Team.
# License: Apache-2.0

from collections import OrderedDict
import torch.nn as nn


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "celu": nn.CELU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "gelu_tanh": (nn.GELU, {"approximate": "tanh"}),
    "hardtanh": nn.Hardtanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hardshrink": nn.Hardshrink,
    "hardswish": nn.Hardswish,
    "leaky_relu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "mish": nn.Mish,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "softmin": nn.Softmin,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": nn.Threshold,
}
ACT2FN = ClassInstantier(ACT2CLS)
