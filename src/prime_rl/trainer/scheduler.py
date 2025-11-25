from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.trainer.config import SchedulerConfigType
from prime_rl.trainer.optim import MultiOptimizer
from prime_rl.trainer.runs import get_runs
from prime_rl.utils.logger import get_logger


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


def _setup_scheduler(
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
                lr=lr,
                min_lr=scheduler_config.min_lr,
            )
        case _:
            raise ValueError(f"Invalid scheduler type: {scheduler_config.type}")


class MultiScheduler:
    def __init__(
        self, optimizer: MultiOptimizer, scheduler_config: SchedulerConfigType, max_steps: int | None, lr: float
    ):
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.max_steps = max_steps
        self.lr = lr
        self.runs = get_runs()
        self.logger = get_logger()
        self.runs.register_creation_hook(self.scheduler_creation_hook)
        self.schedulers: list[LRScheduler | None] = [None] * self.runs.max_runs

    def scheduler_creation_hook(self, idx: int, run_id: str) -> None:
        self.schedulers[idx] = _setup_scheduler(
            self.optimizer.optimizers[idx],
            self.scheduler_config,
            self.max_steps,
            self.lr,
        )

    def step(self):
        for idx in self.runs.used_idxs:
            try:
                self.schedulers[idx].step()
            except Exception as e:
                self.logger.error(f"Error stepping scheduler for run {idx}: {e}")

    def state_dict(self):
        return {
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: dict):
        for scheduler, scheduler_state in zip(self.schedulers, state_dict["schedulers"]):
            scheduler.load_state_dict(scheduler_state)


def setup_multi_scheduler(
    optimizer: MultiOptimizer, scheduler_config: SchedulerConfigType, max_steps: int | None, lr: float
) -> MultiScheduler:
    return MultiScheduler(optimizer, scheduler_config, max_steps, lr)


def setup_scheduler(
    optimizer: Optimizer | MultiOptimizer, scheduler_config: SchedulerConfigType, max_steps: int | None, lr: float
) -> LRScheduler | MultiScheduler:
    if isinstance(optimizer, MultiOptimizer):
        return setup_multi_scheduler(optimizer, scheduler_config, max_steps, lr)
    else:
        return _setup_scheduler(optimizer, scheduler_config, max_steps, lr)
