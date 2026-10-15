import pytest
import torch

from prime_rl.configs.trainer import LinearSchedulerConfig
from prime_rl.trainer.scheduler import setup_scheduler


def lrs(config: LinearSchedulerConfig, max_steps: int) -> list[list[float]]:
    params = [torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))]
    optimizer = torch.optim.AdamW([{"params": params[:1], "lr": 1.0}, {"params": params[1:], "lr": 0.5}])
    scheduler = setup_scheduler(optimizer, config, max_steps, lr=1.0)
    values = []
    for _ in range(max_steps):
        values.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return values


def test_linear_scheduler_fractions_and_decay_shapes():
    linear = lrs(LinearSchedulerConfig(warmup_steps=0.1, decay_steps=0.9), max_steps=20)
    assert linear[2] == [1.0, 0.5] and linear[-1][0] < 1e-7
    shaped = lrs(LinearSchedulerConfig(warmup_steps=2, decay_steps=0.5, decay_shape="1-sqrt"), max_steps=20)
    # The decay runs over the last 10 updates: 1 at its start, 1 - sqrt(5/9) five updates in, ~0 at the end.
    assert shaped[10] == [1.0, 0.5]
    assert shaped[15][0] == pytest.approx(1 - (5 / 9) ** 0.5, abs=1e-6)
    assert shaped[15][1] == pytest.approx(shaped[15][0] / 2)
    assert shaped[-1][0] < 1e-7
