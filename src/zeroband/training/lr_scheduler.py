import math
from functools import partial
from typing import Callable

from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def _get_linear_schedule_with_wsd_sqrt_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_stable_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_stable_steps:
        return 1.0
    else:
        return max(0.0, 1 - math.sqrt(float(current_step - num_stable_steps) / float(num_training_steps - num_stable_steps)))


def get_linear_schedule_with_wsd_sqrt(
    optimizer, num_warmup_steps: int, num_stable_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LRScheduler:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_wsd_sqrt_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


SCHED_MAP: dict[str, Callable[..., LRScheduler]] = {
    "cosine": get_cosine_schedule_with_warmup,
    "wsd-sqrt": get_linear_schedule_with_wsd_sqrt,
    "linear": get_linear_schedule_with_warmup,
}


def get_scheduler(sched_type: str, optimizer, num_warmup_steps: int, num_stable_steps: int, num_training_steps: int) -> LRScheduler:
    if "wsd" in sched_type:
        return SCHED_MAP[sched_type](
            optimizer, num_warmup_steps=num_warmup_steps, num_stable_steps=num_stable_steps, num_training_steps=num_training_steps
        )
    else:
        return SCHED_MAP[sched_type](optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
