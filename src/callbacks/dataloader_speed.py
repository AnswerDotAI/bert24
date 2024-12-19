# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import time
from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ["DataloaderSpeedMonitor"]


class DataloaderSpeedMonitor(Callback):
    """Measure how long it takes to return a batch from the dataloader."""

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_time = time.time_ns()

    def after_dataloader(self, state: State, logger: Logger) -> None:
        self.batch_serve_time = time.time_ns() - self.batch_start_time
        logger.log_metrics(
            {
                "throughput/batch_serve_time_ns": self.batch_serve_time,
                "throughput/batch_serve_time_ms": self.batch_serve_time / 1e6,
            }
        )
