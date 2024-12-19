# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from __future__ import annotations

import logging
from typing import Union

from composer import Time
import torch.nn as nn

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

from src.bert_layers.attention import FlexBertAttentionBase
from src.bert_layers.model import FlexBertPreTrainedModel

try:
    from flash_attn.layers.rotary import RotaryEmbedding  # type: ignore
    from src.bert_layers.rotary import UnpaddedRotaryEmbedding  # type: ignore

except ImportError:
    RotaryEmbedding = None
    UnpaddedRotaryEmbedding = None


log = logging.getLogger(__name__)

__all__ = ["FlexBertRopeSchedule"]


class FlexBertRopeSchedule(Algorithm):
    def __init__(
        self,
        min_rope_theta: int,
        max_rope_theta: int,
        warmup_tokens: Union[str, Time, int],
        rope_theta_increment: int = 10_000,
        target_layer: nn.Module = UnpaddedRotaryEmbedding,
        ignore_sliding_window: bool = True,
        batch_log_interval: int = 10,
        increment_theta_immediately: bool = False,
    ):
        if isinstance(warmup_tokens, str):
            warmup_tokens = Time.from_timestring(warmup_tokens).value
        elif isinstance(warmup_tokens, Time):
            warmup_tokens = warmup_tokens.value
        self.min_rope_theta = min_rope_theta
        self.max_rope_theta = max_rope_theta
        self.rope_theta_increment = rope_theta_increment
        self.target_layer = target_layer
        self.ignore_sliding_window = ignore_sliding_window
        self.batch_log_interval = batch_log_interval
        self.increment_theta_immediately = increment_theta_immediately
        self._rotary_layers = []
        self.warmup_tokens = warmup_tokens  # Store warmup_tokens for recalculations
        self._min_theta = self.min_rope_theta
        self._calculate_increase_every_tokens()

    def _calculate_increase_every_tokens(self):
        self._increase_every_tokens = self.warmup_tokens // (
            (self.max_rope_theta - self._min_theta) / self.rope_theta_increment
        )

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.INIT, Event.FIT_START, Event.BATCH_START, Event.BATCH_END]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.FIT_START:
            flexbert = False
            self._current_theta = self._min_theta
            for layer in state.model.modules():
                if isinstance(layer, FlexBertPreTrainedModel):
                    flexbert = True
                if isinstance(layer, FlexBertAttentionBase):
                    if hasattr(layer, "rotary_emb") and isinstance(layer.rotary_emb, self.target_layer):
                        if (
                            not self.ignore_sliding_window
                            or (hasattr(layer, "sliding_window") and layer.sliding_window == (-1, -1))
                            or not hasattr(layer, "sliding_window")
                        ):
                            self._rotary_layers.append(layer.rotary_emb)
                        if layer.rotary_emb.base != self.min_rope_theta:
                            raise ValueError(f"{self.min_rope_theta=} does not match the Rotary Embedding's RoPE theta {layer.rotary_emb.base}")  # fmt: skip
                        if self.increment_theta_immediately:
                            # Increase the RoPE theta by rope_theta_increment
                            layer.rotary_emb.base += self.rope_theta_increment
            if self.increment_theta_immediately:
                self._min_theta += self.rope_theta_increment
                self._current_theta = self._min_theta
                self._calculate_increase_every_tokens()
            if not flexbert:
                raise ValueError("Rope Schedule only works with a FlexBertPreTrainedModel")
            assert len(self._rotary_layers) > 0, "No layers found to apply Rope Schedule to."

        if event == Event.BATCH_START and state.timestamp.batch.value % self.batch_log_interval == 0:
            logger.log_metrics({"trainer/rope_theta": self._current_theta})

        if event == Event.BATCH_END and self._current_theta != self.max_rope_theta:
            tokens = state.timestamp.token.value

            # Calculate the expected number of increments
            increments = int(tokens // self._increase_every_tokens)
            desired_theta = min(self.max_rope_theta, self._min_theta + increments * self.rope_theta_increment)

            # Check if we need to update the RoPE theta value
            if desired_theta > self._current_theta:
                self._current_theta = desired_theta
                for rotary_emb in self._rotary_layers:
                    rotary_emb.base = self._current_theta
