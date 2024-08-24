# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import logging
import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

try:
    from src.optimizer import StableAdamW
except ImportError:
    StableAdamW = None

__all__ = ["LogGradNorm"]


log = logging.getLogger(__name__)


class LogGradNorm(Callback):
    """Logs the precomputed L1 and L2 gradient norms from StableAdamW"""

    def __init__(self, log_optimizer_metrics: bool = True, batch_log_interval: int = 10):
        self.log_optimizer_metrics = log_optimizer_metrics
        self.batch_log_interval = batch_log_interval
        if StableAdamW is None:
            raise ImportError("Install `pip install torch-optimi` to use the StableAdamW optimizer.")

    def epoch_start(self, state: State, logger: Logger):
        if state.fsdp_enabled and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            raise ValueError("Logging grad_norms is currently incompatible with FSDP.")
        if not isinstance(state.optimizers[0], StableAdamW):
            self.log_optimizer_metrics = False
            log.warn("Disabling `LogGradNorm` as it requires the internal `StableAdamW` optimizer")

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_log_interval != 0 or not self.log_optimizer_metrics:
            return

        optimizer_metrics = getattr(state.optimizers[0], "grad_norms", None)
        if optimizer_metrics is not None:
            logged_metrics = {}
            for metric, value in optimizer_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                logged_metrics[f"gradient_norms/{metric}"] = value
            logger.log_metrics(logged_metrics)
