from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.configs.trainer import SchedulerConfig
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike


def _get_base_optimizer(optimizer: OptimizerLike) -> Optimizer:
    """Extract the base optimizer from a potentially wrapped optimizer."""
    if isinstance(optimizer, OffloadOptimizer):
        return optimizer.base_optimizer
    return optimizer


def setup_constant_scheduler(optimizer: Optimizer) -> LRScheduler:
    """Create a constant learning rate scheduler."""
    return ConstantLR(optimizer, factor=1.0)


def setup_linear_scheduler(
    optimizer: Optimizer, max_steps: int | None, warmup_steps: int, decay_steps: int, lr: float, min_lr: float
) -> LRScheduler:
    """Create a linear (WSD) learning rate scheduler: a linear warmup over ``warmup_steps``, a constant phase, and a
    linear decay over the last ``decay_steps`` (from the peak at the first decay step to ``min_lr`` at the last one).
    Phases of length 0 are left out: a LinearLR without iterations divides by zero when SequentialLR switches to it."""
    assert warmup_steps > 0 or decay_steps > 0, (
        "Either warmup steps or decay steps must be specified for a linear scheduler"
    )
    min_lr_factor = min_lr / lr if min_lr > 0 else 1e-8
    # Every phase and the step it starts at.
    phases: list[tuple[LRScheduler, int]] = []
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=min_lr_factor, end_factor=1.0, total_iters=warmup_steps)
        phases.append((warmup, 0))
    if decay_steps > 0:
        assert max_steps is not None, "max_steps must be specified when specifying decay_steps"
        decay_start_step = max_steps - decay_steps
        assert decay_start_step >= warmup_steps, (
            f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) must not exceed max_steps ({max_steps})"
        )
        if decay_start_step > warmup_steps:
            constant_steps = decay_start_step - warmup_steps
            constant = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)
            phases.append((constant, warmup_steps))
        if decay_steps > 1:
            decay = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr_factor, total_iters=decay_steps - 1)
            phases.append((decay, decay_start_step))
    if not phases:  # a single decay step and no other step: the peak learning rate
        return setup_constant_scheduler(optimizer)
    if len(phases) == 1:
        return phases[0][0]
    return SequentialLR(
        optimizer, [scheduler for scheduler, _ in phases], milestones=[start for _, start in phases[1:]]
    )


def setup_cosine_scheduler(
    optimizer: Optimizer, max_steps: int | None, warmup_steps: int, lr: float, min_lr: float
) -> LRScheduler:
    """Create a cosine learning rate scheduler."""
    # Create schedulers for each phase
    schedulers, milestones = [], []

    assert max_steps is not None, "max_steps must be specified when specifying decay_steps"

    # Add warmup (if any)
    if warmup_steps > 0:
        min_lr_factor = min_lr / lr if min_lr > 0 else 1e-8
        warmup_scheduler = LinearLR(optimizer, start_factor=min_lr_factor, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    decay_steps = max_steps - warmup_steps
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=min_lr)
    schedulers.append(cosine_scheduler)

    # Return single scheduler if only one phase, otherwise combine with SequentialLR
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)


def setup_scheduler(
    optimizer: OptimizerLike,
    scheduler_config: SchedulerConfig,
    max_steps: int | None,
    lr: float,
) -> LRScheduler:
    """Create learning rate scheduler based on config.

    Handles CPUOffloadOptimizer by extracting the base optimizer for the scheduler.
    """
    base_optimizer = _get_base_optimizer(optimizer)

    match scheduler_config.type:
        case "constant":
            return setup_constant_scheduler(base_optimizer)
        case "linear":
            return setup_linear_scheduler(
                base_optimizer,
                max_steps=max_steps,
                warmup_steps=scheduler_config.warmup_steps,
                decay_steps=scheduler_config.decay_steps,
                lr=lr,
                min_lr=scheduler_config.min_lr,
            )
        case "cosine":
            return setup_cosine_scheduler(
                base_optimizer,
                max_steps=max_steps,
                warmup_steps=scheduler_config.warmup_steps,
                lr=lr,
                min_lr=scheduler_config.min_lr,
            )
        case _:
            raise ValueError(f"Invalid scheduler type: {scheduler_config.type}")
