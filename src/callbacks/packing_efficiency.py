# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ["PackingEfficency"]


class PackingEfficency(Callback):
    """Records the packing efficiency for each batch."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval

    def after_dataloader(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval != 0:
            return
        logger.log_metrics(
            {
                "trainer/packing_efficiency": self._packing_efficiency(state),
            }
        )

    def _packing_efficiency(self, state: State) -> float:
        return state.batch["attention_mask"].sum().item() / state.batch["attention_mask"].numel()
