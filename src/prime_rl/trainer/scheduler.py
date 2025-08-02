from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.trainer.config import SchedulerConfig


class ConstantLRScheduler(LRScheduler):
    """A scheduler that keeps the learning rate constant."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def validate_scheduler_config(config: SchedulerConfig, max_steps: int | None) -> None:
    """Validate scheduler configuration against max_steps."""
    if max_steps is None:
        raise ValueError("Must specify max_steps when using a scheduler")

    if config.type == "constant":
        return

    warmup_steps = config.warmup_steps
    decay_steps = config.decay_steps

    if decay_steps is None:
        # Will use remaining steps after warmup
        decay_steps = max_steps - warmup_steps

    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be positive, got {decay_steps}")

    if warmup_steps + decay_steps > max_steps:
        raise ValueError(f"Warmup steps ({warmup_steps}) + decay steps ({decay_steps}) exceeds max_steps ({max_steps})")


def create_lr_scheduler(optimizer: Optimizer, config: SchedulerConfig, max_steps: int | None) -> LRScheduler:
    """Create learning rate scheduler based on config."""
    # Validate configuration
    validate_scheduler_config(config, max_steps)

    if config.type == "constant":
        return ConstantLRScheduler(optimizer)

    warmup_steps = config.warmup_steps
    decay_steps = config.decay_steps

    if decay_steps is None:
        # Fallback: decay for remaining steps after warmup
        decay_steps = max_steps - warmup_steps

    # Calculate when final decay starts
    decay_start_step = max_steps - decay_steps
    constant_steps = decay_start_step - warmup_steps

    # Create schedulers for each phase
    schedulers = []
    milestones = []

    # Phase 1: Warmup (if any)
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    # Phase 2: Constant (if any)
    if constant_steps > 0:
        constant_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)
        schedulers.append(constant_scheduler)
        milestones.append(decay_start_step)

    # Phase 3: Final decay
    if config.type == "linear":
        decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)
    elif config.type == "cosine":
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0.0)

    schedulers.append(decay_scheduler)

    # Return single scheduler if only one phase, otherwise combine with SequentialLR
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)
