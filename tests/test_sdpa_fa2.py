# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import random
import time

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
from test_utils import SynthTextDirectory
from src.bert_layers.initialization import InitFnType

IMPL_USE_FLASH2 = False
try:
    import flash_attn

    IMPL_USE_FLASH2 = True
except ImportError:
    pass


layer_combinations = [
    ("prenorm", "absolute_pos", "base", "mlp"),
    ("postnorm", "absolute_pos", "base", "glu"),
    ("prenorm", "sans_pos", "rope", "mlp"),
    ("postnorm", "sans_pos", "rope", "glu"),
    ("parallel_prenorm", "absolute_pos", "parallel", "parallel_glu"),
    ("parallel_prenorm", "sans_pos", "rope_parallel", "parallel_glu"),
]


@pytest.mark.skipif(not IMPL_USE_FLASH2, reason="Flash Attention is not installed")
@pytest.mark.parametrize("padding", ["padded", "unpadded"])
@pytest.mark.parametrize("layer,embedding,attention,mlp", layer_combinations)
@pytest.mark.parametrize("different_first_layer", [False, True])
def test_trainer(padding: str, layer: str, embedding: str, attention: str, mlp: str, different_first_layer: bool):
    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open("yamls/models/flex_bert.yaml") as f:
        model_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_sdpa_fa2.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, model_cfg, test_config)
    assert isinstance(config, DictConfig)
    config.model.name = "flex_bert"
    config.seed = 42
    config.model.model_config.padding = padding
    config.model.model_config.bert_layer = layer
    config.model.model_config.embedding_layer = embedding
    config.model.model_config.attention_layer = attention
    config.model.model_config.mlp_layer = mlp
    if layer == "postnorm":
        config.model.model_config.final_norm = False

    if different_first_layer:
        if layer != "parallel_prenorm":
            pytest.skip("Only parallel_prenorm needs a different first layer")
        config.model.model_config.first_attention_layer = "rope" if attention == "rope_parallel" else "base"
        config.model.model_config.first_bert_layer = "prenorm"
        config.model.model_config.first_mlp_layer = "glu"

    if config.model.model_config.bert_layer in ["parallel_prenorm", "prenorm"] and random.random() < 0.5:
        config.model.model_config.skip_first_prenorm = True
        config.model.model_config.embed_norm = True

    # pick a random init type for testing
    config.model.model_config.init_fn = random.choice([member.value for member in InitFnType])
    if config.model.model_config.init_fn != InitFnType.full_megatron:
        config.model.model_config.init_small_embedding = random.choice([True, False])

    with SynthTextDirectory() as tmp_datadir:
        config.model.model_config.use_fa2 = False
        if padding == "unpadded":
            config.model.model_config.use_sdpa_attn_mask = True
        else:
            config.model.model_config.use_sdpa_attn_mask = False
        config.train_loader.dataset.remote = tmp_datadir
        config.train_loader.dataset.local = os.path.join(tmp_datadir, "tr-local1")
        config.eval_loader.dataset.remote = tmp_datadir
        config.eval_loader.dataset.local = os.path.join(tmp_datadir, "ev-local1")

        # Train with SDPA
        trainer1 = main(config, return_trainer=True)
        assert trainer1 is not None
        model1 = trainer1.state.model.model

        config.model.model_config.use_fa2 = True
        config.train_loader.dataset.local = os.path.join(tmp_datadir, "tr-local2")
        config.eval_loader.dataset.local = os.path.join(tmp_datadir, "ev-local2")

        # Train with FA2
        trainer2 = main(config, return_trainer=True)
        assert trainer2 is not None
        model2 = trainer2.state.model.model

    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(param1, param2, rtol=1e-2, atol=1e-3)
