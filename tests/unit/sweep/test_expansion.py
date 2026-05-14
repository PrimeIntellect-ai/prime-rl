import math
import re

from prime_rl.configs.sweep import (
    ChoiceParameterConfig,
    IntUniformParameterConfig,
    LogUniformParameterConfig,
    UniformParameterConfig,
)
from prime_rl.sweep.materialize import build_nested_overrides, set_dotted_path
from prime_rl.sweep.search import expand_grid, parameters_hash, sample_random


def test_expand_grid_uses_deterministic_insertion_order() -> None:
    trials = expand_grid(
        {
            "a": ChoiceParameterConfig(values=[1, 2]),
            "b": ChoiceParameterConfig(values=["x", "y"]),
        }
    )

    assert [trial.parameters for trial in trials] == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    pattern = re.compile(r"^\d{4}-[0-9a-f]{8}$")
    assert [trial.id[:4] for trial in trials] == ["0000", "0001", "0002", "0003"]
    assert all(pattern.match(trial.id) for trial in trials)


def test_expand_grid_hash_suffix_is_stable() -> None:
    parameters = {"trainer.optim.lr": 1e-5, "orchestrator.train.sampling.temperature": 0.7}
    assert parameters_hash(parameters) == parameters_hash(dict(reversed(parameters.items())))


def test_expand_grid_hash_changes_with_values() -> None:
    a = parameters_hash({"trainer.optim.lr": 1e-5})
    b = parameters_hash({"trainer.optim.lr": 3e-5})
    assert a != b


def test_build_nested_overrides_from_dotted_paths() -> None:
    overrides = build_nested_overrides(
        {
            "trainer.optim.lr": 1e-5,
            "orchestrator.train.sampling.temperature": 0.7,
            "output_dir": "outputs/run",
        }
    )

    assert overrides == {
        "trainer": {"optim": {"lr": 1e-5}},
        "orchestrator": {"train": {"sampling": {"temperature": 0.7}}},
        "output_dir": "outputs/run",
    }


def test_set_dotted_path_rejects_table_conflict() -> None:
    data = {"trainer": 1}

    try:
        set_dotted_path(data, "trainer.optim.lr", 1e-5)
    except ValueError as exc:
        assert "non-table" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_sample_random_is_deterministic_under_seed() -> None:
    parameters = {
        "optim.lr": LogUniformParameterConfig(distribution="log_uniform", min=1e-6, max=1e-4),
        "optim.warmup": IntUniformParameterConfig(distribution="int_uniform", min=0, max=10, step=2),
        "data.temperature": UniformParameterConfig(distribution="uniform", min=0.6, max=1.2),
        "label": ChoiceParameterConfig(values=["a", "b", "c"]),
    }

    first = sample_random(parameters, num_trials=8, seed=1234)
    second = sample_random(parameters, num_trials=8, seed=1234)
    third = sample_random(parameters, num_trials=8, seed=4321)

    assert [trial.parameters for trial in first] == [trial.parameters for trial in second]
    assert [trial.parameters for trial in first] != [trial.parameters for trial in third]


def test_sample_random_respects_distribution_bounds() -> None:
    parameters = {
        "optim.lr": LogUniformParameterConfig(distribution="log_uniform", min=1e-6, max=1e-4),
        "optim.warmup": IntUniformParameterConfig(distribution="int_uniform", min=0, max=10, step=2),
        "data.temperature": UniformParameterConfig(distribution="uniform", min=0.6, max=1.2),
    }

    trials = sample_random(parameters, num_trials=64, seed=0)
    for trial in trials:
        lr = trial.parameters["optim.lr"]
        warmup = trial.parameters["optim.warmup"]
        temperature = trial.parameters["data.temperature"]
        assert math.log(1e-6) <= math.log(lr) <= math.log(1e-4)
        assert warmup in {0, 2, 4, 6, 8, 10}
        assert 0.6 <= temperature <= 1.2


def test_sample_random_assigns_unique_indexed_ids() -> None:
    parameters = {"label": ChoiceParameterConfig(values=["a", "b", "c"])}
    trials = sample_random(parameters, num_trials=5, seed=0)

    pattern = re.compile(r"^\d{4}-[0-9a-f]{8}$")
    assert [trial.id[:4] for trial in trials] == ["0000", "0001", "0002", "0003", "0004"]
    assert all(pattern.match(trial.id) for trial in trials)
