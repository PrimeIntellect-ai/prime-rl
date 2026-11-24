import math

import pytest
import torch

from prime_rl.configs.trainer import ConstantSchedulerConfig, CosineSchedulerConfig, LinearSchedulerConfig
from prime_rl.trainer.scheduler import setup_scheduler

MAX_STEPS = 16
LR = 1e-3
SCHEDULES = {
    "constant": ConstantSchedulerConfig(),
    "linear": LinearSchedulerConfig(warmup_steps=4, decay_steps=8),
    "cosine": CosineSchedulerConfig(warmup_steps=4),
}


def make_optimizer(lr: float) -> torch.optim.Optimizer:
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    return torch.optim.SGD([{"params": params[:1]}, {"params": params[1:], "lr": lr / 10}], lr=lr)


def learning_rates(optimizer: torch.optim.Optimizer, scheduler, steps: int) -> list[list[float]]:
    """Each param group's learning rate at every step, read before the scheduler's step as the trainer does."""
    lrs = []
    for _ in range(steps):
        lrs.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return lrs


def run(config, checkpoint_step: int | None = None, max_steps: int = MAX_STEPS) -> list[list[float]]:
    """The learning rates of a run from scratch, or resumed after ``checkpoint_step``, up to ``max_steps``."""
    optimizer = make_optimizer(LR)
    scheduler = setup_scheduler(optimizer, config, max_steps, LR, checkpoint_step)
    return learning_rates(optimizer, scheduler, max_steps - (checkpoint_step or 0))


@pytest.mark.parametrize(
    "config, max_steps, factors",
    [
        (LinearSchedulerConfig(warmup_steps=2, decay_steps=3), 6, [1e-8, 0.5, 1, 1, 0.5, 1e-8]),
        (LinearSchedulerConfig(warmup_steps=2, decay_steps=3, min_lr=1e-4), 6, [0.1, 0.55, 1, 1, 0.55, 0.1]),
        (CosineSchedulerConfig(warmup_steps=1), 5, [1e-8, 1, (1 + math.sqrt(0.5)) / 2, 0.5, (1 - math.sqrt(0.5)) / 2]),
        (
            CosineSchedulerConfig(warmup_steps=1, min_lr=1e-4),
            5,
            [0.1, 1, 0.55 + 0.45 * math.sqrt(0.5), 0.55, 0.55 - 0.45 * math.sqrt(0.5)],
        ),
    ],
    ids=["linear", "linear-min_lr", "cosine", "cosine-min_lr"],
)
def test_schedule_learning_rates(config, max_steps, factors):
    assert [lrs[0] for lrs in run(config, max_steps=max_steps)] == pytest.approx([LR * f for f in factors], rel=1e-6)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_resume_continues_the_uninterrupted_run(schedule):
    uninterrupted = run(SCHEDULES[schedule])
    for checkpoint_step in range(1, MAX_STEPS):
        assert run(SCHEDULES[schedule], checkpoint_step) == uninterrupted[checkpoint_step:]


@pytest.mark.parametrize(
    "config",
    [
        ConstantSchedulerConfig(start_step=6),
        LinearSchedulerConfig(warmup_steps=2, decay_steps=4, start_step=6),
        CosineSchedulerConfig(warmup_steps=2, start_step=6),
    ],
    ids=["constant", "linear", "cosine"],
)
def test_start_step_starts_a_fresh_schedule_there(config):
    """A schedule started at step 6 is the schedule over the remaining steps, at this resume and every later one."""
    fresh = run(config.model_copy(update={"start_step": 0}), max_steps=MAX_STEPS - 6)
    for checkpoint_step in range(6, MAX_STEPS):
        assert run(config, checkpoint_step) == fresh[checkpoint_step - 6 :]


@pytest.mark.parametrize("checkpoint_step, error", [(None, "needs a resume"), (4, "before scheduler.start_step")])
def test_start_step_after_the_resume_step_raises(checkpoint_step, error):
    with pytest.raises(ValueError, match=error):
        run(ConstantSchedulerConfig(start_step=5), checkpoint_step)
