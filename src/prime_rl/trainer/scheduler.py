from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.trainer.config import SchedulerConfigType
from prime_rl.trainer.optim import CPUOffloadOptimizer, MultiLoRAOptimizer
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.utils.logger import get_logger


def _get_base_optimizer(optimizer: Optimizer | CPUOffloadOptimizer) -> Optimizer:
    """Extract the base optimizer from a potentially wrapped optimizer."""
    if isinstance(optimizer, CPUOffloadOptimizer):
        return optimizer.optimizer
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
        assert decay_start_step >= warmup_steps
        constant_steps = decay_start_step - warmup_steps
        assert constant_steps >= 0
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
    optimizer: Optimizer | CPUOffloadOptimizer,
    scheduler_config: SchedulerConfigType,
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


class MultiLoRAScheduler:
    """Manages multiple schedulers, one per run.

    Each run has its own independent scheduler that is created when
    the run's optimizer is created via the optimizer creation hook.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfigType,
        max_steps: int | None,
    ):
        self.scheduler_config = scheduler_config
        self.max_steps = max_steps
        self.multi_run_manager = get_multi_run_manager()
        self.logger = get_logger()

        self.schedulers: list[LRScheduler | None] = [None] * self.multi_run_manager.max_runs

    def scheduler_creation_hook(self, optimizer: Optimizer, idx: int) -> None:
        """Create a scheduler for a newly created optimizer.

        This should be called after an optimizer is created for a run.
        """
        lr = self.multi_run_manager.config[idx].optim.lr
        self.schedulers[idx] = setup_scheduler(
            optimizer,
            self.scheduler_config,
            self.max_steps,
            lr,
        )

    def step(self) -> None:
        """Step all active schedulers."""
        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.schedulers[idx].step()

    def get_last_lr(self, idx: int) -> list[float]:
        """Get the last learning rate for a specific run."""
        if self.schedulers[idx] is not None:
            return self.schedulers[idx].get_last_lr()
        return []

    def state_dict(self) -> dict:
        return {
            "schedulers": [scheduler.state_dict() if scheduler is not None else None for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for idx, scheduler_state in enumerate(state_dict["schedulers"]):
            if scheduler_state is not None and self.schedulers[idx] is not None:
                self.schedulers[idx].load_state_dict(scheduler_state)


def setup_multi_scheduler(
    optimizer: MultiLoRAOptimizer,
    scheduler_config: SchedulerConfigType,
    max_steps: int | None,
) -> MultiLoRAScheduler:
    """Create a MultiLoRAScheduler for managing per-run schedulers."""
    scheduler = MultiLoRAScheduler(scheduler_config, max_steps)
    # Register callback at index 0 so schedulers are created before other callbacks
    optimizer.register_post_creation_callback(scheduler.scheduler_creation_hook, index=0)
    return scheduler
