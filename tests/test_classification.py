# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import DictConfig, OmegaConf
from ..sequence_classification import train


@pytest.mark.parametrize("model_name", ["mosaic_bert", "hf_bert", "flex_bert"])
def test_classification_script(model_name):
    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open(f"yamls/models/{model_name}.yaml") as f:
        model_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_classification.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, model_cfg, test_config)
    assert isinstance(config, DictConfig)

    # The test is that `main` runs successfully
    train(config)
