# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import shutil
import tempfile
from typing import Any

import pytest

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glue import train
from omegaconf import DictConfig, OmegaConf


class SuperGlueDirContext(object):
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
def test_superglue_script(model_name: str):
    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open(f"yamls/models/{model_name}.yaml") as f:
        model_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_superglue.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, model_cfg, test_config)
    assert isinstance(config, DictConfig)
    config.model.name = model_name

    if (
        model_name == "flex_bert"
        and not config.model.model_config.use_fa2
        and config.model.model_config.padding == "unpadded"
    ):
        pytest.skip("SDPA call currently errors with SuperGlue test on unpadded inputs")

    # The test is that `train` runs successfully
    with SuperGlueDirContext() as local_save_dir:
        config.save_finetune_checkpoint_prefix = local_save_dir
        train(config)