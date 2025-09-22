from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.trainer.config import SchedulerConfigType


def setup_constant_scheduler(optimizer: Optimizer) -> LRScheduler:
    """Create a constant learning rate scheduler."""
    return ConstantLR(optimizer, factor=1.0)


def setup_linear_scheduler(
    optimizer: Optimizer, max_steps: int | None, warmup_steps: int, decay_steps: int, lr: float, min_lr: float
) -> LRScheduler:
    """Create a linear (WSD) learning rate scheduler."""
    # Create schedulers for each phase
    schedulers, milestones = [], []

    # Add warmup (if any)
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=min_lr / lr, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    # Add decay (if any)
    if decay_steps > 0:
        assert max_steps is not None, "max_steps must be specified when specifying decay_steps"
        decay_start_step = max_steps - decay_steps
        assert decay_start_step >= warmup_steps
        constant_steps = decay_start_step - warmup_steps
        assert constant_steps >= 0
        constant_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)
        decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr / lr, total_iters=decay_steps)
        schedulers.append(constant_scheduler)
        schedulers.append(decay_scheduler)
        milestones.append(decay_start_step)

    # Return single scheduler if only one phase, otherwise combine with SequentialLR
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)


def setup_cosine_scheduler(
    optimizer: Optimizer, max_steps: int | None, warmup_steps: int, decay_steps: int, lr: float, min_lr: float
) -> LRScheduler:
    """Create a cosine learning rate scheduler."""
    # Create schedulers for each phase
    schedulers, milestones = [], []

    # Add warmup (if any)
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=min_lr / lr, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    assert max_steps is not None, "max_steps must be specified when specifying decay_steps"
    decay_start_step = max_steps - decay_steps
    assert decay_start_step >= warmup_steps
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=min_lr)
    schedulers.append(cosine_scheduler)
    milestones.append(decay_start_step)

    # Return single scheduler if only one phase, otherwise combine with SequentialLR
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfigType,
    max_steps: int | None,
    lr: float,
) -> LRScheduler:
    """Create learning rate scheduler based on config."""
    match scheduler_config.type:
        case "constant":
            return setup_constant_scheduler(optimizer)
        case "linear":
            return setup_linear_scheduler(
                optimizer,
                max_steps=max_steps,
                warmup_steps=scheduler_config.warmup_steps,
                decay_steps=scheduler_config.decay_steps,
                lr=lr,
                min_lr=scheduler_config.min_lr,
            )
        case "cosine":
            return setup_cosine_scheduler(
                optimizer,
                max_steps=max_steps,
                warmup_steps=scheduler_config.warmup_steps,
                decay_steps=scheduler_config.decay_steps,
                lr=lr,
                min_lr=scheduler_config.min_lr,
            )
        case _:
            raise ValueError(f"Invalid scheduler type: {scheduler_config.type}")
