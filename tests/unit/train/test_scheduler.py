import pytest
import torch

from prime_rl.trainer.scheduler import setup_linear_scheduler


def test_linear_schedule_with_empty_constant_and_decay_phases():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = setup_linear_scheduler(optimizer, max_steps=2, warmup_steps=1, decay_steps=1, lr=1.0, min_lr=0.1)
    rates = []
    for _ in range(2):
        optimizer.step()
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert rates == pytest.approx([0.1, 1.0])
