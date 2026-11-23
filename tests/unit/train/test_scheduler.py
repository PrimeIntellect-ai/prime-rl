import copy

import torch

from prime_rl.configs.trainer import LinearSchedulerConfig
from prime_rl.trainer.scheduler import restore_scheduled_lrs, setup_scheduler

MAX_STEPS = 16


def optimizer_and_scheduler() -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    optimizer = torch.optim.SGD([{"params": params[:1]}, {"params": params[1:], "lr": 1e-4}], lr=1e-3)
    scheduler = setup_scheduler(optimizer, LinearSchedulerConfig(warmup_steps=4, decay_steps=8), MAX_STEPS, lr=1e-3)
    return optimizer, scheduler


def step(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> list[float]:
    lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    return lrs


def test_schedule_loaded_without_the_optimizer_state_continues_the_run():
    """A schedule loaded into a fresh optimizer, as on a resume with ckpt.skip_optimizer, continues with the
    uninterrupted run's learning rates, from a step in the warmup, the constant phase or the decay."""
    optimizer, scheduler = optimizer_and_scheduler()
    reference, states = [], []
    for _ in range(MAX_STEPS):
        reference.append(step(optimizer, scheduler))
        states.append(copy.deepcopy(scheduler.state_dict()))

    for saved_step in range(MAX_STEPS - 1):
        optimizer, scheduler = optimizer_and_scheduler()
        scheduler.load_state_dict(states[saved_step])
        restore_scheduled_lrs(scheduler)
        resumed = [step(optimizer, scheduler) for _ in range(saved_step + 1, MAX_STEPS)]
        assert resumed == reference[saved_step + 1 :], f"resumed after step {saved_step}"
