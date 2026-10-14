import math

import pytest
import torch
from pydantic import ValidationError

from prime_rl.configs.pretrain import PretrainConfig, ScheduleConfig
from prime_rl.trainer.pretrain.schedules import LRSchedule


def multipliers(config: ScheduleConfig, total_steps: int) -> list[float]:
    params = [torch.nn.Parameter(torch.zeros(1))]
    schedule = LRSchedule(config, torch.optim.AdamW(params, lr=1.0), total_steps)
    return [schedule.multiplier(step) for step in range(total_steps)]


def test_wsd_endpoints_and_shapes():
    f = multipliers(ScheduleConfig(warmup=0.1, decay=0.2, end_ratio=0.05), total_steps=100)
    assert f[0] == 0.0
    assert f[9] == 1.0
    assert f[79] == 1.0
    assert f[89] == pytest.approx(0.05 + 0.95 * 0.5)
    assert f[99] == pytest.approx(0.05)

    for shape, midpoint in [("cosine", 0.5), ("1-sqrt", 1 - math.sqrt(0.5))]:
        f = multipliers(ScheduleConfig(warmup=0, decay=20, decay_shape=shape), total_steps=100)
        assert f[0] == 1.0
        assert f[89] == pytest.approx(midpoint)
        assert f[99] == 0.0


def test_schedule_phases_must_fit_max_steps():
    config = PretrainConfig(
        model={"arch": {"model_type": "llama"}},
        tokenizer={"name": "t"},
        data={"type": "fake", "seq_len": 8, "batch_tokens": 8, "micro_batch_tokens": 8},
        optim={"lr": 1e-3},
        max_steps=100,
    )
    with pytest.raises(ValidationError, match="exceed max_steps"):
        PretrainConfig(**{**config.model_dump(), "schedule": {"warmup": 50, "decay": 51}})


def test_apply_keeps_group_ratios():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    optimizer = torch.optim.AdamW([{"params": params[:1], "lr": 1.0}, {"params": params[1:], "lr": 0.1}])
    schedule = LRSchedule(ScheduleConfig(warmup=0, decay=2, end_ratio=0.5), optimizer, total_steps=4)
    assert schedule.apply(0) == 1.0
    assert [group["lr"] for group in optimizer.param_groups] == [1.0, 0.1]

    # Loading a state dict replaces the param group dicts.
    optimizer.load_state_dict(optimizer.state_dict())
    assert schedule.apply(3) == 0.5
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.5, 0.05])
