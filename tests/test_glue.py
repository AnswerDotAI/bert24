# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from typing import Any

import pytest

from ..glue import train
from omegaconf import DictConfig, OmegaConf


class GlueDirContext(object):
    def __init__(self):
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        del exc_type, exc_value, traceback  # unused
        if self.path is not None:
            shutil.rmtree(self.path)


@pytest.mark.parametrize("model_name", ["mosaic_bert", "hf_bert", "flex_bert"])
def test_glue_script(model_name: str):
    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open(f"yamls/models/{model_name}.yaml") as f:
        model_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_glue.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, model_cfg, test_config)
    assert isinstance(config, DictConfig)
    config.model.name = model_name

    # The test is that `train` runs successfully
    with GlueDirContext() as local_save_dir:
        config.save_finetune_checkpoint_prefix = local_save_dir
        train(config)
