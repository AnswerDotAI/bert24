import textwrap
import warnings
from typing import Union

from composer.core import State, Time
from composer.optim.scheduler import (
    ComposerScheduler,
    LinearScheduler,
    _convert_time,
)


class WarmupStableDecayScheduler(ComposerScheduler):
    r"""
    Args:
        t_warmup (str | Time): Warmup time.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
        scale_warmup (float): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(
        self,
        t_warmup: Union[str, Time],
        t_decay: Union[str, Time] = "0.1dur",
        t_max: Union[str, Time] = "1dur",
        alpha_f: float = 0.1,
        scale_warmup: bool = False,
    ):
        self.t_warmup = t_warmup
        self.t_decay = t_decay
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=1e-10, alpha_f=1.0, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):
        assert state.max_duration is not None, "max_duration should be set whenever schedulers are invoked"
        t_warmup = _convert_time(self.t_warmup, state)
        t_decay = _convert_time(self.t_decay, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent(
                    """\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""
                )
            )

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)
        elif state.timestamp < t_max - t_decay:
            return 1.0
        else:
            current_time = state.timestamp.get(t_warmup.unit)
            current_decay_time = current_time - (t_max - t_decay)
            current_decay_fraction = current_decay_time / t_decay
            lr_scale = 1 - current_decay_fraction.value * (1 - self.alpha_f)
            return max(0.0, lr_scale)  # prevent negative lr_scale
