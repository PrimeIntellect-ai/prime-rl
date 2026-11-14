from types import SimpleNamespace

import pytest

from prime_rl.configs.diff import merge_resumed_config
from prime_rl.configs.orchestrator import CheckpointConfig, ConcurrencyConfig, OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.control.controller import validate_update
from prime_rl.control.types import ConfigUpdate, InvalidConfigError
from prime_rl.orchestrator.ckpt import CheckpointManager
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.eval_source import EvalSource
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
    raw["train"]["source"][0]["group_size"] = 8

    _, changes = validate_update(example, raw)

    assert [change.path for change in changes] == [
        "train.source[0].sampling.temperature",
        "train.source[0].group_size",
        "concurrency.max_inflight",
        "max_staleness",
    ]


def test_example_config_frozen_changes_are_rejected(example):
    raw = dump_resolved_config(example)
    raw["seq_len"] += 1
    raw["train"]["source"][0]["sampling"]["logprobs"] = 5
    raw["train"]["group_size"] *= 2
    raw["train"]["sampling"]["temperature"] = 0.8

    with pytest.raises(InvalidConfigError) as error:
        validate_update(example, raw)

    assert error.value.detail["frozen"] == [
        "train.source[0].sampling.logprobs",
        "train.sampling.temperature",
        "train.group_size",
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

    manager.save(Progress(), train_source, config_state, step=3)

    assert manager.read_config_state(3) == config_state


def eval_config(intervals: dict[str, int]) -> OrchestratorConfig:
    sources = [
        {"name": name, "env": {"taskset": {"id": "reverse-text"}}, "interval": interval}
        for name, interval in intervals.items()
    ]
    return OrchestratorConfig.model_validate(
        {"train": {"source": [{"env": {"taskset": {"id": "reverse-text"}}}]}, "eval": {"source": sources}}
    )


def test_resume_with_changed_eval_sources():
    base = dump_resolved_config(eval_config({"kept": 10, "removed": 20}))
    applied = dump_resolved_config(eval_config({"kept": 5, "removed": 20}))
    launch = dump_resolved_config(eval_config({"kept": 10, "added": 7}))

    config = OrchestratorConfig.model_validate(merge_resumed_config(base, applied, launch))
    assert config.eval.intervals == {"kept": 5, "added": 7}

    envs = [
        SimpleNamespace(
            name=name, examples=[SimpleNamespace(key=name, hash=name)], config=SimpleNamespace(group_size=1)
        )
        for name in config.eval.intervals
    ]
    source = EvalSource(envs, intervals=config.eval.intervals, is_resumed=True)
    assert [source.trigger(step)[0] for step in (13, 14, 15, 35)] == [[], ["added"], ["kept"], ["kept", "added"]]
