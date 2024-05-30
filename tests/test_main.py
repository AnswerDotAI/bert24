# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
from test_utils import SynthTextDirectory


@pytest.mark.parametrize("model_name,seed", [("mosaic_bert", 17), ("hf_bert", 18), ("flex_bert", 42)])
def test_trainer(model_name: str, seed: int):
    with open("yamls/defaults.yaml") as f:
        default_cfg = OmegaConf.load(f)
    with open(f"yamls/models/{model_name}.yaml") as f:
        model_cfg = OmegaConf.load(f)
    with open("tests/smoketest_config_main.yaml") as f:
        test_config = OmegaConf.load(f)
    config = OmegaConf.merge(default_cfg, model_cfg, test_config)
    assert isinstance(config, DictConfig)
    config.model.name = model_name
    config.seed = seed

    with SynthTextDirectory() as tmp_datadir:
        config.train_loader.dataset.remote = tmp_datadir
        config.train_loader.dataset.local = os.path.join(tmp_datadir, "tr-local1")
        config.eval_loader.dataset.remote = tmp_datadir
        config.eval_loader.dataset.local = os.path.join(tmp_datadir, "ev-local1")
        # Also save checkpoints in the temporary directory
        config.save_folder = tmp_datadir

        # Train
        trainer1 = main(config, return_trainer=True)
        assert trainer1 is not None
        model1 = trainer1.state.model.model

        # Check that the checkpoint was saved
        chkpt_path = os.path.join(tmp_datadir, "latest-rank0.pt")
        assert os.path.isfile(chkpt_path), f"{os.listdir(tmp_datadir)}"

        # Check that the checkpoint was loaded by comparing model weights (with no weight changes)
        config.load_path = chkpt_path
        config.seed += 10  # change seed
        config.train_loader.dataset.local = os.path.join(tmp_datadir, "tr-local2")
        config.eval_loader.dataset.local = os.path.join(tmp_datadir, "ev-local2")
        trainer2 = main(config, return_trainer=True, do_train=False)
        assert trainer2 is not None
        model2 = trainer2.state.model.model

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(param1, param2)
