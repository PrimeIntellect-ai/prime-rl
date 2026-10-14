import math
from typing import Callable

from prime_rl.configs.pretrain import ScheduleConfig, resolve_steps
from prime_rl.trainer.optim import OptimizerLike

DECAY_SHAPES: dict[str, Callable[[float], float]] = {
    "linear": lambda progress: 1 - progress,
    "cosine": lambda progress: 0.5 * (1 + math.cos(math.pi * progress)),
    "1-sqrt": lambda progress: 1 - math.sqrt(progress),
}


def wsd(step: int, total_steps: int, warmup: int, decay: int, end_ratio: float, decay_shape: str) -> float:
    """Linear warmup from 0 at step 0 to 1 at step ``warmup - 1``, stable at 1, then ``decay`` updates
    that reach ``end_ratio`` at the last step ``total_steps - 1``."""
    peak = max(warmup - 1, 0)
    if step < peak:
        return step / peak
    if decay == 0:
        return 1.0
    last_stable = max(total_steps - 1 - decay, peak)
    if step <= last_stable:
        return 1.0
    progress = (step - last_stable) / (total_steps - 1 - last_stable)
    return end_ratio + (1 - end_ratio) * DECAY_SHAPES[decay_shape](progress)


class LRSchedule:
    """Sets the learning rate of each update. It is a pure function of the step, so nothing needs to be
    checkpointed: on resume it is recomputed from the step.

    The base learning rates are read from the optimizer's param groups at construction, before any
    checkpoint load. Each param group keeps its own base, so ratios between groups hold at every step.
    """

    def __init__(self, config: ScheduleConfig, optimizer: OptimizerLike, total_steps: int, step_offset: int = 0):
        self.config = config
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.step_offset = step_offset
        # Bases by group index: loading an optimizer state dict replaces the group dicts.
        self.bases = [group["lr"] for group in optimizer.param_groups]

    def multiplier(self, step: int) -> float:
        config = self.config
        return wsd(
            step - self.step_offset,
            self.total_steps,
            resolve_steps(config.warmup, self.total_steps),
            resolve_steps(config.decay, self.total_steps),
            config.end_ratio,
            config.decay_shape,
        )

    def apply(self, step: int) -> float:
        """Write the learning rates of update ``step`` (updates completed so far) and return the multiplier."""
        multiplier = self.multiplier(step)
        for group, base in zip(self.optimizer.param_groups, self.bases):
            group["lr"] = base * multiplier
        return multiplier
