from types import SimpleNamespace

import pytest

from prime_rl.configs.orchestrator import CheckpointConfig, ConcurrencyConfig, OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.control.controller import validate_update
from prime_rl.control.types import ConfigUpdate, InvalidConfigError
from prime_rl.orchestrator.ckpt import CheckpointManager, build_state
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.types import Progress
from prime_rl.utils.config import cli, dump_resolved_config


@pytest.fixture
def example():
    return cli(RLConfig, args=["@", "examples/basic/reverse-text/rl.toml"]).orchestrator


@pytest.mark.parametrize(
    ("target", "bounds", "expected_limits"),
    [(None, (40, 64), [40]), (None, (1, 8), [8]), (16, (1, 8), [])],
    ids=["raise_floor", "lower_ceiling", "manual_target"],
)
def test_concurrency_reconfigure_clamps_into_new_band(target, bounds, expected_limits):
    controller = ConcurrencyController(ConcurrencyConfig(initial_inflight=32, max_inflight=64), fallback_cost=1)
    limits = []
    controller.bind(set_limit=limits.append, get_inflight=lambda: 0)
    controller.set_target(target)
    limits.clear()
    min_inflight, max_inflight = bounds
    config = OrchestratorConfig(concurrency=ConcurrencyConfig(min_inflight=min_inflight, max_inflight=max_inflight))

    controller.reconfigure(ConfigUpdate(revision=1, step=1, config=config, changes=()))

    assert limits == expected_limits


def test_example_config_live_changes_are_accepted(example):
    raw = dump_resolved_config(example)
    raw["max_staleness"] = 2
    raw["concurrency"]["max_inflight"] = 256
    raw["train"]["source"][0]["sampling"]["temperature"] = 0.8

    _, changes = validate_update(example, raw)

    assert [change.path for change in changes] == [
        "train.source[0].sampling.temperature",
        "concurrency.max_inflight",
        "max_staleness",
    ]


def test_example_config_frozen_changes_are_rejected(example):
    raw = dump_resolved_config(example)
    raw["seq_len"] += 1
    raw["train"]["source"][0]["sampling"]["logprobs"] = 5
    raw["train"]["source"][0]["group_size"] *= 2
    raw["train"]["sampling"]["temperature"] = 0.8

    with pytest.raises(InvalidConfigError) as error:
        validate_update(example, raw)

    assert error.value.detail["frozen"] == [
        "train.source[0].sampling.logprobs",
        "train.source[0].group_size",
        "train.sampling.temperature",
        "seq_len",
    ]


def test_example_config_length_penalty_is_live(example):
    raw = dump_resolved_config(example)
    raw["train"]["source"][0]["algo"]["length_penalty"] = {"num_output_tokens_weight": 0.2}

    config, changes = validate_update(example, raw)

    assert [change.path for change in changes] == ["train.source[0].algo.length_penalty"]
    assert config.train.source[0].algo.length_penalty.num_output_tokens_weight == 0.2


def test_example_config_rest_of_algo_is_frozen(example):
    raw = dump_resolved_config(example)
    raw["train"]["source"][0]["algo"]["length_weighted_baseline"] = True
    raw["train"]["algo"]["length_penalty"] = {"num_output_tokens_weight": 0.2}

    with pytest.raises(InvalidConfigError) as error:
        validate_update(example, raw)

    assert error.value.detail["frozen"] == [
        "train.source[0].algo.length_weighted_baseline",
        "train.algo.length_penalty",
    ]


def test_example_config_frozen_change_that_fails_validation_is_reported_frozen(example):
    raw = dump_resolved_config(example)
    raw["train"]["source"][0]["sampling"]["top_p"] = 0.9

    with pytest.raises(InvalidConfigError) as error:
        validate_update(example, raw)

    assert error.value.detail["frozen"] == ["train.source[0].sampling.top_p"]


def test_checkpoint_keeps_config_state(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointConfig())
    config_state = {"base": {"batch_size": 64}, "applied": {"batch_size": 128}, "revision": 2}
    train_source = SimpleNamespace(state_dict=lambda: {"rng": None, "envs": {}})

    manager.save_state(build_state(Progress(), train_source, config_state, {}), step=3)

    assert manager.read_config_state(3) == config_state
