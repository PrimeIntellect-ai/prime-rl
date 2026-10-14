import math
from functools import partial
from typing import Callable

from prime_rl.configs.pretrain import ScheduleConfig, SchedulesConfig, resolve_steps
from prime_rl.configs.trainer import OptimizerConfig, ParamGroupKey
from prime_rl.trainer.optim import OptimizerLike
from prime_rl.utils.utils import import_object

Schedule = Callable[[int], float]
"""Multiplier for an update, given the number of updates completed before it (0 for the first)."""

DECAY_SHAPES: dict[str, Callable[[float], float]] = {
    "linear": lambda progress: 1 - progress,
    "cosine": lambda progress: 0.5 * (1 + math.cos(math.pi * progress)),
    "1-sqrt": lambda progress: 1 - math.sqrt(progress),
}


def wsd(
    step: int,
    total_steps: int | None,
    warmup: int,
    decay: int,
    start_ratio: float,
    end_ratio: float,
    decay_shape: str,
) -> float:
    """Linear warmup from ``start_ratio`` at step 0 to 1 at step ``warmup - 1``, stable at 1, then
    ``decay`` updates that reach ``end_ratio`` at the last step ``total_steps - 1``."""
    peak = max(warmup - 1, 0)
    if step < peak:
        return start_ratio + (1 - start_ratio) * step / peak
    if decay == 0:
        return 1.0
    last_stable = max(total_steps - 1 - decay, peak)
    if step <= last_stable:
        return 1.0
    progress = (step - last_stable) / (total_steps - 1 - last_stable)
    return end_ratio + (1 - end_ratio) * DECAY_SHAPES[decay_shape](progress)


def cosine(step: int, total_steps: int, warmup: int, start_ratio: float, end_ratio: float) -> float:
    """Linear warmup as in ``wsd``, then cosine decay from 1 to ``end_ratio`` at the last step."""
    decay = total_steps - max(warmup, 1)
    return wsd(step, total_steps, warmup, decay, start_ratio, end_ratio, "cosine")


def constant(step: int, total_steps: int | None) -> float:
    return 1.0


def build_schedule(config: ScheduleConfig, total_steps: int | None) -> Schedule:
    """Bind a schedule config to the number of updates it spans. Fractional phases resolve against it."""
    match config.type:
        case "wsd":
            return partial(
                wsd,
                total_steps=total_steps,
                warmup=resolve_steps(config.warmup, total_steps),
                decay=resolve_steps(config.decay, total_steps),
                start_ratio=config.start_ratio,
                end_ratio=config.end_ratio,
                decay_shape=config.decay_shape,
            )
        case "cosine":
            return partial(
                cosine,
                total_steps=total_steps,
                warmup=resolve_steps(config.warmup, total_steps),
                start_ratio=config.start_ratio,
                end_ratio=config.end_ratio,
            )
        case "constant":
            return partial(constant, total_steps=total_steps)
        case "custom":
            return partial(import_object(config.import_path), total_steps=total_steps, **config.kwargs)


class Schedules:
    """Sets the scheduled hyperparameters of each update. The values are a pure function of the step,
    so nothing needs to be checkpointed: on resume they are recomputed from the step.

    The base values are read from the optimizer's param groups at construction, before any checkpoint
    load. Each param group keeps its own base, so ratios between groups hold at every step.
    """

    def __init__(
        self,
        config: SchedulesConfig,
        optim_config: OptimizerConfig,
        optimizer: OptimizerLike,
        total_steps: int | None,
        step_offset: int = 0,
    ):
        self.optimizer = optimizer
        self.step_offset = step_offset
        # Group indices, not group dicts: loading an optimizer state dict replaces the group dicts.
        self.optim: dict[str, tuple[Schedule, ParamGroupKey, list[tuple[int, float]]]] = {}
        for name, schedule in config.optim.items():
            target = optim_config.schedulable[name]
            bases = [
                (i, group[target.key] if target.index is None else group[target.key][target.index])
                for i, group in enumerate(optimizer.param_groups)
                if target.algorithm is None or group["algorithm"] == target.algorithm
            ]
            self.optim[name] = (build_schedule(schedule, total_steps), target, bases)
        self.max_norm = optim_config.max_norm
        self.clip_norm = build_schedule(config.clip_norm, total_steps) if config.clip_norm is not None else None

    def apply(self, step: int) -> tuple[float | None, dict[str, float]]:
        """Write the values of update ``step`` (updates completed so far) into the optimizer.
        Returns the gradient clipping norm and metrics of every scheduled value."""
        step -= self.step_offset
        metrics = {}
        for name, (schedule, target, bases) in self.optim.items():
            multiplier = schedule(step)
            for i, base in bases:
                group = self.optimizer.param_groups[i]
                if target.index is None:
                    group[target.key] = base * multiplier
                else:
                    values = list(group[target.key])
                    values[target.index] = base * multiplier
                    group[target.key] = tuple(values)
            metrics[f"schedule/{name}"] = multiplier
            metrics[f"optim/{name}"] = bases[0][1] * multiplier
        max_norm = self.max_norm
        if self.clip_norm is not None and max_norm is not None:
            multiplier = self.clip_norm(step)
            max_norm *= multiplier
            metrics["schedule/clip_norm"] = multiplier
            metrics["optim/max_norm"] = max_norm
        return max_norm, metrics
