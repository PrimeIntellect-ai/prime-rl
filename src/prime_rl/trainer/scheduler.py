from __future__ import annotations

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from prime_rl.configs.trainer import SchedulerConfig
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike


def _get_base_optimizer(optimizer: OptimizerLike) -> Optimizer:
    """Extract the base optimizer from a potentially wrapped optimizer."""
    if isinstance(optimizer, OffloadOptimizer):
        return optimizer.base_optimizer
    return optimizer


def _min_lr_factor(min_lr: float, lr: float) -> float:
    """``min_lr`` relative to ``lr``: the factor a warmup starts at and a linear decay ends at, 1e-8 for 0."""
    return min_lr / lr if min_lr > 0 else 1e-8


def _constant_factor(epoch: int) -> float:
    return 1.0


def _linear_factor(epoch: int, horizon: int | None, warmup_steps: int, decay_steps: int, min_factor: float) -> float:
    """The linear (WSD) schedule's factor of the peak at ``epoch``: a linear warmup from ``min_factor`` over
    ``warmup_steps``, a constant phase, and a linear decay over the last ``decay_steps`` of ``horizon`` (from the peak at
    the first decay step to ``min_factor`` at the last one)."""
    if epoch < warmup_steps:
        return min_factor + (1 - min_factor) * epoch / warmup_steps
    if decay_steps <= 1:
        return 1.0
    decay_epoch = min(max(epoch - (horizon - decay_steps), 0), decay_steps - 1)
    return min_factor + (1 - min_factor) * (decay_steps - 1 - decay_epoch) / (decay_steps - 1)


def _cosine_factor(epoch: int, horizon: int, warmup_steps: int, start_factor: float, min_factor: float) -> float:
    """The cosine schedule's factor of the peak at ``epoch``: a linear warmup from ``start_factor`` over
    ``warmup_steps``, then a cosine decay from the peak to ``min_factor`` over the rest of ``horizon``."""
    if epoch < warmup_steps:
        return start_factor + (1 - start_factor) * epoch / warmup_steps
    decay_steps = horizon - warmup_steps
    decay_epoch = min(epoch - warmup_steps, decay_steps)
    return min_factor + (1 - min_factor) * (1 + math.cos(math.pi * decay_epoch / decay_steps)) / 2


def setup_scheduler(
    optimizer: OptimizerLike,
    scheduler_config: SchedulerConfig,
    max_steps: int | None,
    lr: float,
    checkpoint_step: int | None,
) -> LambdaLR:
    """Create the learning rate scheduler, positioned at the step after ``checkpoint_step`` (the first step on a fresh
    start). The learning rate is a function of the step: each param group's lr at setup (its peak, kept as
    ``initial_lr``) times the schedule's factor. ``min_lr`` is relative to ``lr``, so every group warms up from and
    decays to the same fraction of its peak.

    Handles CPUOffloadOptimizer by extracting the base optimizer for the scheduler.
    """
    start_step = scheduler_config.start_step
    if checkpoint_step is None and start_step > 0:
        raise ValueError(f"scheduler.start_step ({start_step}) needs a resume from a checkpoint at or after it")
    if checkpoint_step is not None and checkpoint_step < start_step:
        raise ValueError(f"Resuming from step {checkpoint_step}, before scheduler.start_step ({start_step})")
    horizon = max_steps - start_step if max_steps is not None else None

    match scheduler_config.type:
        case "constant":
            factor = _constant_factor
        case "linear":
            factor = partial(
                _linear_factor,
                horizon=horizon,
                warmup_steps=scheduler_config.warmup_steps,
                decay_steps=scheduler_config.decay_steps,
                min_factor=_min_lr_factor(scheduler_config.min_lr, lr),
            )
        case "cosine":
            factor = partial(
                _cosine_factor,
                horizon=horizon,
                warmup_steps=scheduler_config.warmup_steps,
                start_factor=_min_lr_factor(scheduler_config.min_lr, lr),
                min_factor=scheduler_config.min_lr / lr if scheduler_config.min_lr > 0 else 0.0,
            )
        case _:
            raise ValueError(f"Invalid scheduler type: {scheduler_config.type}")

    base_optimizer = _get_base_optimizer(optimizer)
    for group in base_optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    last_epoch = checkpoint_step - 1 if checkpoint_step is not None else -1
    return LambdaLR(base_optimizer, lambda epoch: factor(epoch - start_step), last_epoch=last_epoch)
