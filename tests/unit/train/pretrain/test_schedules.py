import math

import pytest
import torch
from pydantic import ValidationError

from prime_rl.configs.pretrain import (
    CosineScheduleConfig,
    CustomScheduleConfig,
    PretrainConfig,
    SchedulesConfig,
    WSDScheduleConfig,
)
from prime_rl.configs.trainer import AdamWConfig
from prime_rl.trainer.pretrain.schedules import Schedules, build_schedule


def test_wsd_endpoints_and_shapes():
    config = WSDScheduleConfig(warmup=0.1, decay=0.2, start_ratio=0.1, end_ratio=0.05)
    f = build_schedule(config, total_steps=100)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == 1.0
    assert f(79) == 1.0
    assert f(89) == pytest.approx(0.05 + 0.95 * 0.5)
    assert f(99) == pytest.approx(0.05)

    for shape, midpoint in [("cosine", 0.5), ("1-sqrt", 1 - math.sqrt(0.5))]:
        f = build_schedule(WSDScheduleConfig(warmup=0, decay=20, decay_shape=shape), total_steps=100)
        assert f(0) == 1.0
        assert f(89) == pytest.approx(midpoint)
        assert f(99) == 0.0


def test_cosine_endpoints():
    f = build_schedule(CosineScheduleConfig(warmup=10, end_ratio=0.1), total_steps=100)
    assert f(0) == 0.0
    assert f(9) == 1.0
    assert f(54) == pytest.approx(0.55)
    assert f(99) == pytest.approx(0.1)
    assert build_schedule(CosineScheduleConfig(warmup=0), total_steps=10)(0) == 1.0


def inverse_sqrt(step: int, total_steps: int, scale: float) -> float:
    return scale / math.sqrt(step + 1)


def test_custom_schedule():
    config = CustomScheduleConfig(import_path=f"{__name__}.inverse_sqrt", kwargs={"scale": 2.0})
    assert build_schedule(config, total_steps=10)(3) == 1.0


def test_schedules_config_validation():
    PretrainConfig(max_steps=100, optim={"type": "muon"}, schedules={"optim": {"mu": {"type": "constant"}}})
    for schedules in [{"optim": {"momentum": {"type": "constant"}}}, {"optim": {"mu": {"type": "constant"}}}]:
        with pytest.raises(ValidationError, match="cannot be scheduled"):
            PretrainConfig(max_steps=100, schedules=schedules)
    with pytest.raises(ValidationError, match="needs max_steps"):
        PretrainConfig()
    with pytest.raises(ValidationError, match="exceed max_steps"):
        PretrainConfig(max_steps=100, schedules={"optim": {"lr": {"type": "wsd", "warmup": 50, "decay": 51}}})


def test_apply_keeps_group_ratios():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    optimizer = torch.optim.AdamW(
        [{"params": params[:1], "lr": 1.0}, {"params": params[1:], "lr": 0.1}], betas=(0.9, 0.95)
    )
    config = SchedulesConfig(
        optim={
            "lr": WSDScheduleConfig(warmup=0, decay=2, end_ratio=0.5),
            "betas2": WSDScheduleConfig(warmup=0, decay=1, end_ratio=0.5),
        },
        clip_norm=WSDScheduleConfig(warmup=2, decay=0),
    )
    schedules = Schedules(config, AdamWConfig(max_norm=2.0), optimizer, total_steps=4)
    max_norm, metrics = schedules.apply(0)
    assert max_norm == 0.0
    assert [group["lr"] for group in optimizer.param_groups] == [1.0, 0.1]

    # Loading a state dict replaces the param group dicts.
    optimizer.load_state_dict(optimizer.state_dict())
    max_norm, metrics = schedules.apply(3)
    assert max_norm == 2.0
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.5, 0.05])
    assert [group["betas"] for group in optimizer.param_groups] == [(0.9, 0.475)] * 2
    assert metrics["schedule/lr"] == 0.5
    assert metrics["optim/lr"] == 0.5
