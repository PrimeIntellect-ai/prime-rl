import copy

import pytest
import torch

from prime_rl.configs.trainer import ConstantSchedulerConfig, CosineSchedulerConfig, LinearSchedulerConfig
from prime_rl.trainer.scheduler import load_scheduler_state, restore_configured_lrs, setup_scheduler

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


def step(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> list[float]:
    lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    return lrs


def uninterrupted_run(config) -> tuple[list[list[float]], list[tuple[dict, dict]]]:
    """Each step's learning rates, and the optimizer's and the scheduler's state after each step."""
    optimizer = make_optimizer(LR)
    scheduler = setup_scheduler(optimizer, config, MAX_STEPS, LR)
    lrs, states = [], []
    for _ in range(MAX_STEPS):
        lrs.append(step(optimizer, scheduler))
        states.append((copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())))
    return lrs, states


@pytest.mark.parametrize("lr", [LR, LR / 2])
@pytest.mark.parametrize("optimizer_state", [True, False])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_resume_continues_the_schedule_at_the_configured_lr(schedule, optimizer_state, lr):
    """A resume, with or without the optimizer's state (``ckpt.skip_optimizer``), continues the uninterrupted run's
    learning rates from any step, scaled by a changed optim.lr."""
    config = SCHEDULES[schedule]
    reference, states = uninterrupted_run(config)
    for saved_step in range(MAX_STEPS - 1):
        optimizer = make_optimizer(lr)
        scheduler = setup_scheduler(optimizer, config, MAX_STEPS, lr)
        if optimizer_state:
            optimizer.load_state_dict(states[saved_step][0])
        load_scheduler_state(scheduler, states[saved_step][1], optimizer_state)
        resumed = [step(optimizer, scheduler) for _ in range(saved_step + 1, MAX_STEPS)]
        expected = [[value * (lr / LR) for value in lrs] for lrs in reference[saved_step + 1 :]]
        assert resumed == expected, f"resumed after step {saved_step}"


@pytest.mark.parametrize("optimizer_state", [True, False])
def test_skip_scheduler_starts_the_configured_schedule(optimizer_state):
    """A resume with ckpt.skip_scheduler, with or without the optimizer's state, runs a fresh schedule from a changed
    optim.lr, as the trainers rebuild it: the schedule built before the load had already scaled a warmup's start."""
    config = LinearSchedulerConfig(warmup_steps=4, decay_steps=0)
    _, states = uninterrupted_run(config)
    fresh_optimizer = make_optimizer(LR / 2)
    fresh_scheduler = setup_scheduler(fresh_optimizer, config, MAX_STEPS, LR / 2)
    expected = [step(fresh_optimizer, fresh_scheduler) for _ in range(MAX_STEPS)]

    optimizer = make_optimizer(LR / 2)
    configured_lrs = [group["lr"] for group in optimizer.param_groups]
    setup_scheduler(optimizer, config, MAX_STEPS, LR / 2)
    if optimizer_state:
        optimizer.load_state_dict(states[8][0])
    restore_configured_lrs(optimizer, configured_lrs)
    scheduler = setup_scheduler(optimizer, config, MAX_STEPS, LR / 2)
    assert [step(optimizer, scheduler) for _ in range(MAX_STEPS)] == expected
