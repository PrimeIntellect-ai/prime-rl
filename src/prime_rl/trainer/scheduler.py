from __future__ import annotations

from typing import Any

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
    """Create a linear (WSD) learning rate scheduler."""
    # Create schedulers for each phase
    schedulers, milestones = [], []

    assert warmup_steps > 0 or decay_steps > 0, (
        "Either warmup steps or decay steps must be specified for a linear scheduler"
    )

    # Add warmup (if any)
    min_lr_factor = min_lr / lr if min_lr > 0 else 1e-8
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=min_lr_factor, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    # Add decay (if any)
    if decay_steps > 0:
        assert max_steps is not None, "max_steps must be specified when specifying decay_steps"
        decay_start_step = max_steps - decay_steps
        assert decay_start_step >= warmup_steps, (
            f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) must not exceed max_steps ({max_steps})"
        )
        constant_steps = decay_start_step - warmup_steps
        constant_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)
        decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr_factor, total_iters=decay_steps - 1)
        schedulers.append(constant_scheduler)
        schedulers.append(decay_scheduler)
        milestones.append(decay_start_step)

    # Return single scheduler if only one phase, otherwise combine with SequentialLR
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)


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


def restore_configured_lrs(optimizer: OptimizerLike, lrs: list[float]) -> None:
    """For a fresh schedule after a resume (``ckpt.skip_scheduler``): put back each param group's configured learning
    rate, ``lrs`` as read from the groups when the optimizer was built, for the fresh schedule to start from. The
    scheduler built before the load and the optimizer's checkpoint have changed the groups' rates since."""
    for group, lr in zip(optimizer.param_groups, lrs, strict=True):
        group["lr"] = lr
        group.pop("initial_lr", None)


def load_scheduler_state(scheduler: LRScheduler, state: dict[str, Any], optimizer_state: bool) -> None:
    """Load a checkpoint's scheduler state into ``scheduler``, built from the config: the schedule continues from the
    step it reached, at the configured learning rates.

    Without the optimizer's state (``ckpt.skip_optimizer``), the param groups first get the loaded schedule's last
    learning rates: the new scheduler set the fresh groups to its start, which the loaded schedule's next steps would
    scale. Then each group's rate and the schedule's base rates scale by the group's configured / loaded learning
    rate. The rest of the schedule is the checkpoint's: a linear decay's floor scales with the rates, a cosine's
    ``min_lr`` stays."""
    phases = scheduler._schedulers if isinstance(scheduler, SequentialLR) else [scheduler]
    configured_lrs = list(phases[0].base_lrs)
    scheduler.load_state_dict(state)
    loaded_lrs = phases[0].base_lrs
    groups = scheduler.optimizer.param_groups
    if not optimizer_state:
        last_lrs = scheduler.get_last_lr()
        if len(groups) != len(last_lrs):
            raise ValueError(
                f"Resuming with ckpt.skip_optimizer: the optimizer has {len(groups)} param groups and the "
                f"checkpoint's scheduler has learning rates for {len(last_lrs)}, so the loaded schedule cannot be "
                "matched to the groups. Resume with ckpt.skip_scheduler too to start a fresh schedule."
            )
        for group, lr in zip(groups, last_lrs):
            group["lr"] = lr
    for group, configured, loaded in zip(groups, configured_lrs, loaded_lrs, strict=True):
        if configured != loaded:
            group["lr"] *= configured / loaded
        group["initial_lr"] = configured
    for phase in phases:
        phase.base_lrs = list(configured_lrs)
    scheduler._last_lr = [group["lr"] for group in groups]
