"""Weight broadcast memory configuration tests."""

import pytest

from prime_rl.configs.rl import RLConfig

MODEL = "Qwen/Qwen3-0.6B"


def rl_config(**broadcast):
    return RLConfig.model_validate(
        {
            "model": {"name": MODEL},
            "trainer": {"model": {"name": MODEL}},
            "orchestrator": {"model": {"name": MODEL}},
            "inference": {},
            "weight_broadcast": broadcast,
        }
    )


@pytest.mark.parametrize("transport", ["mx_refit", "nccl", "nixl", "filesystem"])
def test_reclaim_setting_reaches_trainer(transport):
    config = rl_config(type=transport, reclaim_memory="if_needed", reclaim_headroom_gb=72.0)

    assert config.trainer.weight_broadcast.reclaim_memory == "if_needed"
    assert config.trainer.weight_broadcast.reclaim_headroom_gb == 72.0
    assert not hasattr(config.orchestrator.weight_broadcast, "reclaim_memory")
    assert not hasattr(config.orchestrator.weight_broadcast, "reclaim_headroom_gb")


def test_reclaim_default_reaches_trainer():
    config = rl_config(type="mx_refit")

    assert config.trainer.weight_broadcast.reclaim_memory == "always"


def test_unknown_reclaim_mode_is_rejected():
    with pytest.raises(Exception):
        rl_config(type="mx_refit", reclaim_memory="occasionally")


def test_headroom_must_be_positive():
    with pytest.raises(Exception):
        rl_config(type="mx_refit", reclaim_headroom_gb=0)
