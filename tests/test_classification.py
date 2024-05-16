# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig, OmegaConf
from ..sequence_classification import main


def test_classification_script():
    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_classification.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, test_config)
    assert isinstance(config, DictConfig)

    # The test is that `main` runs successfully
    main(config)
