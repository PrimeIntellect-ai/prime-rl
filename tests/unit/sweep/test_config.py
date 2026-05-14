from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from prime_rl.configs.sweep import (
    ChoiceParameterConfig,
    IntUniformParameterConfig,
    LogUniformParameterConfig,
    OptunaStrategyConfig,
    RandomStrategyConfig,
    SweepConfig,
    UniformParameterConfig,
)
from prime_rl.utils.config import cli


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_sweep_config_defaults(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    assert config.entrypoint == "rl"
    assert config.strategy.type == "grid"
    assert config.scheduler.type == "local"
    assert config.scheduler.max_parallel == 1
    assert isinstance(config.parameters["optim.lr"], ChoiceParameterConfig)


def test_sweep_config_loads_from_cli_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "sweep.toml"
    write_toml(
        config_path,
        {
            "entrypoint": "sft",
            "base": ["base.toml"],
            "output_dir": "outputs/study",
            "scheduler": {"type": "slurm"},
            "parameters": {"optim.lr": {"values": [1e-5]}},
        },
    )

    config = cli(SweepConfig, args=["@", config_path.as_posix()])

    assert config.entrypoint == "sft"
    assert config.scheduler.type == "slurm"
    assert config.parameters["optim.lr"].values == [1e-5]


def test_slurm_scheduler_rejects_max_parallel(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "slurm", "max_parallel": 4},
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_sweep_config_requires_base(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="base"):
        SweepConfig(
            base=[],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_sweep_config_requires_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="parameters"):
        SweepConfig(base=[tmp_path / "base.toml"], output_dir=tmp_path / "study", parameters={})


def test_sweep_config_rejects_parent_and_child_parameter_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="parent path and one of its sub-paths"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={
                "optim": {"values": [{"lr": 1e-5}]},
                "optim.lr": {"values": [3e-5]},
            },
        )


@pytest.mark.parametrize("path", ["", ".optim.lr", "optim..lr", "optim.lr."])
def test_sweep_config_rejects_empty_parameter_path_segments(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValidationError, match="non-empty dot-separated"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={path: {"values": [1e-5]}},
        )


def test_sweep_config_rejects_empty_structured_choice_path_segments(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="non-empty dot-separated"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim": {"values": [{"": 1e-5}]}},
        )


@pytest.mark.parametrize(
    "path",
    [
        "output_dir",
        "output_dir.name",
        "trainer.output_dir",
        "orchestrator.output_dir",
        "trainer.ckpt.output_dir",
        "orchestrator.output_dir.name",
    ],
)
def test_sweep_config_rejects_output_dir_parameters(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValidationError, match="cannot set output_dir"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={path: {"values": ["custom"]}},
        )


def test_sweep_config_rejects_output_dir_inside_structured_choice(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="cannot set output_dir"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"trainer": {"values": [{"ckpt": {"output_dir": "custom"}}]}},
        )


def test_sweep_config_rejects_output_dir_inside_nested_structured_choice_list(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="cannot set output_dir"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"trainer": {"values": [[[{"ckpt": {"output_dir": "custom"}}]]]}},
        )


@pytest.mark.parametrize(
    "path",
    [
        "wandb",
        "wandb.group",
        "wandb.name",
        "wandb.tags",
        "trainer.wandb",
        "trainer.wandb.project",
        "trainer.wandb.name",
        "trainer.wandb.tags.foo",
        "orchestrator.wandb",
        "orchestrator.wandb.project",
        "orchestrator.wandb.name",
        "orchestrator.wandb.tags.foo",
    ],
)
def test_sweep_config_rejects_wandb_identity_parameters_when_injection_enabled(
    tmp_path: Path, path: str
) -> None:
    with pytest.raises(ValidationError, match="sweep-managed W&B identity"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={path: {"values": ["custom"]}},
        )


def test_sweep_config_rejects_wandb_identity_inside_structured_choice(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="sweep-managed W&B identity"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"trainer": {"values": [{"wandb": {"name": "custom"}}]}},
        )


def test_sweep_config_allows_wandb_parameters_when_injection_disabled(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"wandb.group": {"values": ["custom"]}},
        wandb=None,
    )

    assert "wandb.group" in config.parameters


def test_sweep_config_allows_unmanaged_wandb_parameters_when_injection_enabled(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"wandb.project": {"values": ["custom"]}},
    )

    assert "wandb.project" in config.parameters


def test_sweep_config_allows_unmanaged_nested_wandb_extras_when_injection_enabled(
    tmp_path: Path,
) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"orchestrator.wandb.log_extras.interval": {"values": [20]}},
    )

    assert "orchestrator.wandb.log_extras.interval" in config.parameters


def test_sweep_parameter_requires_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="values"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": []}},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_choice_parameter_rejects_non_finite_values(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [value]}},
        )


def test_choice_parameter_rejects_nested_non_finite_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="finite"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim": {"values": [{"lr": float("nan")}]}},
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {1: "not-a-toml-key"},
        {"nested": None},
        {1, 2},
        Path("not-toml"),
    ],
)
def test_choice_parameter_rejects_non_toml_serializable_values(tmp_path: Path, value) -> None:
    with pytest.raises(ValidationError, match="TOML|None|non-string"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [value]}},
        )


def test_choice_parameter_allows_none_string_for_nullable_fields(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"model.revision": {"values": ["None"]}},
    )

    assert config.parameters["model.revision"].values == ["None"]


@pytest.mark.parametrize(
    ("distribution", "bounds"),
    [
        ("uniform", {"min": False, "max": 1.0}),
        ("uniform", {"min": 0.0, "max": True}),
        ("log_uniform", {"min": True, "max": 2.0}),
        ("int_uniform", {"min": True, "max": 10}),
        ("int_uniform", {"min": 0, "max": True}),
        ("int_uniform", {"min": 0, "max": 10, "step": True}),
    ],
)
def test_distribution_parameters_reject_bool_numeric_fields(
    tmp_path: Path, distribution: str, bounds: dict
) -> None:
    with pytest.raises(ValidationError, match="cannot be boolean"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 1},
            parameters={"optim.lr": {"distribution": distribution, **bounds}},
        )


def test_uniform_parameter_rejects_overflowing_range(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="range must be finite"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 1},
            parameters={
                "optim.lr": {
                    "distribution": "uniform",
                    "min": -1e308,
                    "max": 1e308,
                }
            },
        )


def test_local_max_parallel_requires_gpu_assignment(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="gpu_assignment"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "local", "max_parallel": 2},
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_max_parallel_requires_enough_gpu_groups(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="requires at least"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 4,
                "gpu_assignment": {"visible_devices": [[0], [1]]},
            },
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_gpu_assignment_rejects_overlapping_groups(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="only appear in one"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 2,
                "gpu_assignment": {"visible_devices": [[0, 1], [1, 2]]},
            },
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_gpu_assignment_rejects_empty_group(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="at least one device"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 2,
                "gpu_assignment": {"visible_devices": [[0], []]},
            },
            parameters={"optim.lr": {"values": [1e-5]}},
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"strategy": {"type": "random", "num_trials": True}},
        {"strategy": {"type": "random", "num_trials": 1, "seed": True}},
        {"scheduler": {"type": "local", "max_parallel": True}},
        {"scheduler": {"type": "local", "gpu_assignment": {"visible_devices": [[True]]}}},
        {"retry_budget": True},
        {
            "strategy": {"type": "optuna", "num_trials": True},
            "objective": {"metric": "reward", "direction": "maximize"},
        },
        {
            "strategy": {"type": "optuna", "num_trials": 1, "seed": True},
            "objective": {"metric": "reward", "direction": "maximize"},
        },
        {
            "strategy": {
                "type": "optuna",
                "num_trials": 1,
                "pruner": {"type": "median", "n_startup_trials": True},
            },
            "objective": {"metric": "reward", "direction": "maximize"},
        },
        {
            "strategy": {
                "type": "optuna",
                "num_trials": 1,
                "pruner": {"type": "asha", "min_resource": True},
            },
            "objective": {"metric": "reward", "direction": "maximize"},
        },
        {
            "strategy": {
                "type": "optuna",
                "num_trials": 1,
                "pruner": {"type": "hyperband", "max_resource": True},
            },
            "objective": {"metric": "reward", "direction": "maximize"},
        },
        {
            "objective": {"metric": "reward", "direction": "maximize"},
            "early_stopping": {"type": "patience", "patience": True},
        },
        {
            "objective": {"metric": "reward", "direction": "maximize"},
            "early_stopping": {"type": "threshold", "threshold": 0.1, "min_trials": True},
        },
    ],
)
def test_sweep_numeric_controls_reject_bool_values(tmp_path: Path, extra: dict) -> None:
    with pytest.raises(ValidationError, match="boolean"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            **extra,
        )


def test_multi_run_lora_max_concurrent_runs_rejects_bool(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="boolean"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": True,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        )


def test_local_max_parallel_with_gpu_assignment_validates(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "local",
            "max_parallel": 2,
            "gpu_assignment": {"visible_devices": [[0, 1], [2, 3]]},
        },
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )
    assert config.scheduler.max_parallel == 2
    assert config.scheduler.gpu_assignment.mode == "static"
    assert config.scheduler.gpu_assignment.visible_devices == [[0, 1], [2, 3]]


def test_local_gpu_assignment_accepts_documented_static_mode(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "local",
            "max_parallel": 2,
            "gpu_assignment": {"mode": "static", "visible_devices": [[0, 1], [2, 3]]},
        },
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )
    assert config.scheduler.gpu_assignment.mode == "static"


def test_random_strategy_accepts_distribution_parameters(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 4, "seed": 7},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
            "data.temperature": {"distribution": "uniform", "min": 0.6, "max": 1.2},
        },
    )

    assert isinstance(config.strategy, RandomStrategyConfig)
    assert config.strategy.num_trials == 4
    assert config.strategy.seed == 7
    assert isinstance(config.parameters["optim.lr"], LogUniformParameterConfig)
    assert isinstance(config.parameters["optim.warmup"], IntUniformParameterConfig)
    assert isinstance(config.parameters["data.temperature"], UniformParameterConfig)


def test_grid_strategy_rejects_distribution_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Grid strategy"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
        )


def test_log_uniform_requires_positive_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="positive"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 1},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 0.0, "max": 1e-4}},
        )


@pytest.mark.parametrize(
    ("distribution", "bounds"),
    [
        ("uniform", {"min": float("nan"), "max": 1.0}),
        ("uniform", {"min": 0.0, "max": float("inf")}),
        ("log_uniform", {"min": float("nan"), "max": 1e-4}),
        ("log_uniform", {"min": 1e-6, "max": float("inf")}),
    ],
)
def test_float_distributions_reject_non_finite_bounds(
    tmp_path: Path, distribution: str, bounds: dict[str, float]
) -> None:
    with pytest.raises(ValidationError, match="finite"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 1},
            parameters={"optim.lr": {"distribution": distribution, **bounds}},
        )


def test_resume_overrides_clean_output_dir(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="resume=true takes precedence"):
        config = SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            resume=True,
            clean_output_dir=True,
        )
    assert config.resume is True
    assert config.clean_output_dir is False


def test_resume_rejects_unseeded_random_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="seed"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 4},
            parameters={"optim.lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
            resume=True,
        )


def test_resume_accepts_seeded_random_strategy(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 4, "seed": 11},
        parameters={"optim.lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
        resume=True,
    )
    assert config.strategy.seed == 11


def test_int_uniform_rejects_non_divisible_step(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="divisible"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 4},
            parameters={"optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 4}},
        )


def test_objective_config_round_trips(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        objective={"metric": "val/loss", "direction": "minimize"},
    )
    assert config.objective.metric == "val/loss"
    assert config.objective.direction == "minimize"
    assert config.objective.source == "final_summary"


@pytest.mark.parametrize("metric", ["", "   "])
def test_objective_config_rejects_blank_metric(tmp_path: Path, metric: str) -> None:
    with pytest.raises(ValidationError, match="objective.metric must be non-empty"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            objective={"metric": metric, "direction": "maximize"},
        )


def test_early_stopping_requires_objective(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="early_stopping requires an objective"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            early_stopping={"type": "patience", "patience": 3},
        )


def test_early_stopping_rejects_slurm_scheduler(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="early_stopping is not supported with the SLURM"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "slurm"},
            parameters={"optim.lr": {"values": [1e-5]}},
            objective={"metric": "val/loss", "direction": "minimize"},
            early_stopping={"type": "patience", "patience": 3},
        )


def test_early_stopping_rejects_static_multi_run_lora(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="static multi_run_lora"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
            objective={"metric": "reward", "direction": "maximize"},
            early_stopping={"type": "patience", "patience": 1},
        )


def test_early_stopping_threshold_parses(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        objective={"metric": "val/loss", "direction": "minimize"},
        early_stopping={"type": "threshold", "threshold": 5.0, "min_trials": 2},
    )
    assert config.early_stopping.type == "threshold"
    assert config.early_stopping.threshold == 5.0
    assert config.early_stopping.min_trials == 2


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_early_stopping_threshold_rejects_non_finite_values(
    tmp_path: Path, threshold: float
) -> None:
    with pytest.raises(ValidationError, match="threshold.*finite"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            objective={"metric": "val/loss", "direction": "minimize"},
            early_stopping={"type": "threshold", "threshold": threshold},
        )


def test_early_stopping_threshold_rejects_boolean_value(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="cannot be boolean"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            objective={"metric": "val/loss", "direction": "minimize"},
            early_stopping={"type": "threshold", "threshold": True},
        )


def test_optuna_strategy_requires_objective(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Optuna strategy requires an objective"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 4},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        )


def test_optuna_strategy_rejects_async_slurm_scheduler(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError, match="Optuna strategy is not supported with the asynchronous SLURM"
    ):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "slurm"},
            strategy={"type": "optuna", "num_trials": 4},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_optuna_strategy_accepts_synchronous_slurm_scheduler(tmp_path: Path) -> None:
    SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True},
        strategy={"type": "optuna", "num_trials": 4},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )


def test_optuna_pruner_accepts_synchronous_slurm_scheduler(tmp_path: Path) -> None:
    """Median/ASHA/Hyperband pruners now work with the synchronous SLURM
    scheduler: the controller submits with ``sbatch --parsable``, polls
    metrics.jsonl + squeue, and ``scancel``s the job on a prune signal."""
    SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True},
        strategy={
            "type": "optuna",
            "num_trials": 4,
            "pruner": {"type": "median"},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )


def test_slurm_max_parallel_accepts_optuna_without_pruner(tmp_path: Path) -> None:
    """max_parallel > 1 with TPE Optuna and no pruner is the supported
    happy path for parallel SLURM-sync sweeps."""
    SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 3},
        strategy={"type": "optuna", "num_trials": 6, "sampler": "tpe"},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )


def test_slurm_max_parallel_requires_synchronous(tmp_path: Path) -> None:
    """max_parallel > 1 without synchronous=true is incoherent — the
    asynchronous SLURM scheduler submits jobs and exits, so it cannot
    manage concurrent in-flight trials."""
    with pytest.raises(ValidationError, match="max_parallel > 1 requires scheduler.synchronous=true"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "slurm", "synchronous": False, "max_parallel": 2},
            strategy={"type": "optuna", "num_trials": 6},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_slurm_max_parallel_accepts_pruner(tmp_path: Path) -> None:
    """Parallel SLURM-sync now supports pruners: each worker thread holds
    its own optuna_trial from study.ask(), and Optuna's storage backend
    serializes concurrent report/should_prune calls across threads — the
    same contract that makes Optuna's own study.optimize(n_jobs>1) work."""
    SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 3},
        strategy={
            "type": "optuna",
            "num_trials": 6,
            "pruner": {"type": "median"},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )


def test_early_stopping_accepts_synchronous_slurm_scheduler(tmp_path: Path) -> None:
    SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True},
        parameters={"optim.lr": {"values": [1e-5]}},
        objective={"metric": "val/loss", "direction": "minimize"},
        early_stopping={"type": "patience", "patience": 3},
    )


def test_optuna_strategy_resume_requires_storage(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="strategy.storage"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 4},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
            resume=True,
        )


@pytest.mark.parametrize("storage", ["", "   "])
def test_optuna_strategy_rejects_blank_storage(tmp_path: Path, storage: str) -> None:
    with pytest.raises(ValidationError, match="Optuna storage"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 4, "storage": storage},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_optuna_strategy_rejects_max_parallel_gt_one(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Optuna strategy on the local scheduler runs sequentially"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 2,
                "gpu_assignment": {"visible_devices": [[0], [1]]},
            },
            strategy={"type": "optuna", "num_trials": 4},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_optuna_strategy_parses_with_storage(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 4,
            "sampler": "random",
            "seed": 42,
            "storage": "sqlite:///optuna.db",
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )
    assert isinstance(config.strategy, OptunaStrategyConfig)
    assert config.strategy.sampler == "random"
    assert config.strategy.storage == "sqlite:///optuna.db"


def test_optuna_strategy_default_pruner_is_none(tmp_path: Path) -> None:
    """Phase 5a configs must keep parsing: pruner defaults to type='none'."""
    from prime_rl.configs.sweep import NoPrunerConfig

    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 2},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )
    assert isinstance(config.strategy.pruner, NoPrunerConfig)
    assert config.strategy.poll_interval_seconds == 5.0


def test_optuna_strategy_rejects_non_finite_poll_interval(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="poll_interval_seconds.*finite"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 2, "poll_interval_seconds": float("inf")},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_optuna_strategy_rejects_boolean_poll_interval(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="cannot be boolean"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 2, "poll_interval_seconds": True},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_optuna_strategy_rejects_out_of_range_seed(tmp_path: Path, seed: int) -> None:
    with pytest.raises(ValidationError, match="Optuna seed"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 2, "seed": seed},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_optuna_choice_parameters_accept_storage_safe_scalars(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 2},
        parameters={
            "optim.name": {"values": ["adam", "sgd"]},
            "optim.seed": {"values": [1, 2]},
            "optim.lr": {"values": [1e-5, 3e-5]},
            "model.enabled": {"values": [False, True]},
        },
        objective={"metric": "reward", "direction": "maximize"},
    )

    assert config.strategy.type == "optuna"


@pytest.mark.parametrize("value", [["adam", "sgd"], {"name": "adam"}, ("adam", "sgd")])
def test_optuna_choice_parameters_reject_structured_values(tmp_path: Path, value) -> None:
    with pytest.raises(ValidationError, match="storage-safe primitive"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 2},
            parameters={"optim": {"values": [value]}},
            objective={"metric": "reward", "direction": "maximize"},
        )


@pytest.mark.parametrize("values", [[True, 1], [False, 0], [1, 1.0], ["adam", "adam"]])
def test_optuna_choice_parameters_reject_equality_collisions(
    tmp_path: Path, values: list[object]
) -> None:
    with pytest.raises(ValidationError, match="equality-colliding"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "optuna", "num_trials": 2},
            parameters={"optim.choice": {"values": values}},
            objective={"metric": "reward", "direction": "maximize"},
        )


def test_grid_choice_parameters_allow_structured_toml_values(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"optim": {"values": [{"name": "adam"}, ["sgd", "momentum"], ("adam", "w")]}},
    )

    assert config.parameters["optim"].values[0] == {"name": "adam"}


def test_optuna_strategy_parses_median_pruner(tmp_path: Path) -> None:
    from prime_rl.configs.sweep import MedianPrunerConfig

    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 4,
            "pruner": {"type": "median", "n_startup_trials": 3, "n_warmup_steps": 10, "interval_steps": 2},
            "poll_interval_seconds": 1.5,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )
    assert isinstance(config.strategy.pruner, MedianPrunerConfig)
    assert config.strategy.pruner.n_startup_trials == 3
    assert config.strategy.pruner.n_warmup_steps == 10
    assert config.strategy.pruner.interval_steps == 2
    assert config.strategy.poll_interval_seconds == 1.5


def test_optuna_strategy_parses_asha_and_hyperband_pruners(tmp_path: Path) -> None:
    from prime_rl.configs.sweep import AshaPrunerConfig, HyperbandPrunerConfig

    asha_config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 4,
            "pruner": {"type": "asha", "min_resource": 8, "reduction_factor": 3},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )
    assert isinstance(asha_config.strategy.pruner, AshaPrunerConfig)
    assert asha_config.strategy.pruner.min_resource == 8
    assert asha_config.strategy.pruner.reduction_factor == 3

    hyperband_config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 4,
            "pruner": {"type": "hyperband", "min_resource": 4, "max_resource": 32, "reduction_factor": 4},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )
    assert isinstance(hyperband_config.strategy.pruner, HyperbandPrunerConfig)
    assert hyperband_config.strategy.pruner.min_resource == 4
    assert hyperband_config.strategy.pruner.max_resource == 32


@pytest.mark.parametrize("min_resource", [0, -1])
def test_optuna_strategy_rejects_invalid_asha_min_resource(
    tmp_path: Path, min_resource: int
) -> None:
    with pytest.raises(ValidationError, match="ASHA pruner min_resource"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={
                "type": "optuna",
                "num_trials": 4,
                "pruner": {"type": "asha", "min_resource": min_resource},
            },
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


@pytest.mark.parametrize(
    ("min_resource", "max_resource"),
    [(1, 0), (8, 4)],
)
def test_optuna_strategy_rejects_invalid_hyperband_max_resource(
    tmp_path: Path, min_resource: int, max_resource: int
) -> None:
    with pytest.raises(ValidationError, match="Hyperband pruner max_resource"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={
                "type": "optuna",
                "num_trials": 4,
                "pruner": {
                    "type": "hyperband",
                    "min_resource": min_resource,
                    "max_resource": max_resource,
                },
            },
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
            objective={"metric": "reward", "direction": "maximize"},
        )


# ---------------------------------------------------------------------------
# Phase 7a — multi_run_lora scheduler
# ---------------------------------------------------------------------------


def test_multi_run_lora_scheduler_parses(tmp_path: Path) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig

    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 4,
            "shared": [tmp_path / "shared.toml"],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
    )
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    assert config.scheduler.max_concurrent_runs == 4
    assert config.scheduler.shared == [tmp_path / "shared.toml"]


def test_multi_run_lora_rejects_non_orchestrator_parameter(tmp_path: Path) -> None:
    """Allowlist: trainer/model/deployment/inference can't vary inside a shared trainer."""
    with pytest.raises(ValidationError, match="not in the allowlist"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={"trainer.optim.lr": {"values": [1e-5, 3e-5]}},
        )


def test_multi_run_lora_accepts_lora_alpha_param(tmp_path: Path) -> None:
    """orchestrator.model.lora.* is in the allowlist."""
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [tmp_path / "shared.toml"],
        },
        parameters={"orchestrator.model.lora.alpha": {"values": [16.0, 32.0]}},
    )
    assert "orchestrator.model.lora.alpha" in config.parameters


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("orchestrator.optim.lr", 3e-5),
        ("orchestrator.model.lora.rank", 8),
        ("orchestrator.batch_size", 128),
        ("orchestrator.token_batch_size", 8192),
        ("orchestrator.oversampling_factor", 2.0),
        ("orchestrator.max_inflight_rollouts", 512),
        ("orchestrator.rollouts_per_example", 8),
        ("orchestrator.max_off_policy_steps", 4),
        ("orchestrator.strict_async_level", True),
        ("orchestrator.seed", 123),
        ("orchestrator.tasks_per_minute", 60),
        ("orchestrator.train.sampling", {"temperature": 0.7, "extra_body": {"top_k": -1}}),
        ("orchestrator.train.sampling.temperature", 0.7),
        ("orchestrator.train.sampling.max_tokens", 128),
        ("orchestrator.train.sampling.extra_body", {"top_k": -1}),
        ("orchestrator.train.sampling.extra_body.top_k", -1),
        ("orchestrator.train.num_workers", 2),
        ("orchestrator.train.max_retries", 1),
        ("orchestrator.eval.sampling", {"temperature": 0.5, "extra_body": {"min_p": 0.0}}),
        ("orchestrator.eval.sampling.temperature", 0.5),
        ("orchestrator.eval.sampling.max_tokens", 128),
        ("orchestrator.eval.sampling.extra_body.min_p", 0.0),
        ("orchestrator.eval.interval", 10),
        ("orchestrator.eval.eval_base_model", False),
        ("orchestrator.eval.skip_eval_on_resume", False),
        ("orchestrator.eval.cancel_inflight_rollouts_on_eval", True),
        ("orchestrator.buffer.easy_fraction", 0.25),
        ("orchestrator.buffer.hash_keys", ["prompt"]),
    ],
)
def test_multi_run_lora_accepts_materializable_orchestrator_paths(
    tmp_path: Path, path: str, value
) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [tmp_path / "shared.toml"],
        },
        parameters={path: {"values": [value]}},
    )
    assert path in config.parameters


@pytest.mark.parametrize(
    "path",
    [
        "orchestrator.batch_size_extra",
        "orchestrator.batch_size.foo",
        "orchestrator.token_batch_size.foo",
        "orchestrator.max_steps",
        "orchestrator.max_async_level",
        "orchestrator.optim.nope",
        "orchestrator.model.lora.nope",
        "orchestrator.train.sampling.nope",
        "orchestrator.train.env.id",
        "orchestrator.eval.sampling.nope",
        "orchestrator.eval.env.id",
        "orchestrator.buffer.hash_keys.foo",
        "orchestrator.buffer.unknown",
    ],
)
def test_multi_run_lora_rejects_unmaterializable_or_unknown_paths(
    tmp_path: Path, path: str
) -> None:
    with pytest.raises(ValidationError, match="not in the allowlist"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={path: {"values": [1]}},
        )


@pytest.mark.parametrize(
    ("path", "value", "offending"),
    [
        ("orchestrator.batch_size", {"extra": 128}, "orchestrator.batch_size.extra"),
        ("orchestrator.buffer.hash_keys", [{"extra": "prompt"}], "orchestrator.buffer.hash_keys.extra"),
        ("orchestrator.buffer.hash_keys", [[{"extra": "prompt"}]], "orchestrator.buffer.hash_keys.extra"),
    ],
)
def test_multi_run_lora_rejects_structured_choice_leaves_under_exact_fields(
    tmp_path: Path, path: str, value, offending: str
) -> None:
    with pytest.raises(ValidationError, match=offending):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={path: {"values": [value]}},
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("orchestrator.batch_size", {}, "scalar field"),
        ("orchestrator.batch_size", [128], "scalar field"),
        ("orchestrator.buffer.hash_keys", [], "non-empty list"),
        ("orchestrator.buffer.hash_keys", "prompt", "list\\[str\\]"),
        ("orchestrator.buffer.hash_keys", [1], "list\\[str\\]"),
        ("orchestrator.train.sampling", 1, "table/dict"),
        ("orchestrator.train.sampling", {"temperature": []}, "orchestrator.train.sampling.temperature"),
        ("orchestrator.train.sampling", {"extra_body": 1}, "orchestrator.train.sampling.extra_body"),
        ("orchestrator.eval.sampling", {"extra_body": 1}, "orchestrator.eval.sampling.extra_body"),
        ("orchestrator.train.sampling.extra_body", 1, "table/dict"),
    ],
)
def test_multi_run_lora_rejects_wrong_exact_field_choice_shapes(
    tmp_path: Path, path: str, value, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={path: {"values": [value]}},
        )


def test_multi_run_lora_rejects_distributions_for_exact_table_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="explicit choice"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 1, "seed": 7},
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={
                "orchestrator.train.sampling.extra_body": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 1.0,
                }
            },
        )


def test_multi_run_lora_rejects_both_batching_modes(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="batch_size.*token_batch_size"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={
                "orchestrator.batch_size": {"values": [128]},
                "orchestrator.token_batch_size": {"values": [8192]},
            },
        )


def test_multi_run_lora_rejects_token_batch_with_oversampling_factor(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="oversampling_factor.*token_batch_size"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={
                "orchestrator.token_batch_size": {"values": [8192]},
                "orchestrator.oversampling_factor": {"values": [2.0]},
            },
        )


@pytest.mark.parametrize("section", ["train", "eval"])
def test_multi_run_lora_rejects_both_sampling_token_aliases(tmp_path: Path, section: str) -> None:
    with pytest.raises(ValidationError, match="max_completion_tokens.*max_tokens"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={
                f"orchestrator.{section}.sampling.max_completion_tokens": {"values": [64]},
                f"orchestrator.{section}.sampling.max_tokens": {"values": [128]},
            },
        )


@pytest.mark.parametrize("section", ["train", "eval"])
def test_multi_run_lora_rejects_sampling_token_aliases_inside_table_choice(
    tmp_path: Path, section: str
) -> None:
    with pytest.raises(ValidationError, match="max_completion_tokens.*max_tokens"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={
                f"orchestrator.{section}.sampling": {
                    "values": [{"max_completion_tokens": 64, "max_tokens": 128}]
                }
            },
        )


def test_multi_run_lora_rejects_extra_body_parent_and_child_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="parent path and one of its sub-paths"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={
                "orchestrator.train.sampling.extra_body": {"values": [{"top_k": -1}]},
                "orchestrator.train.sampling.extra_body.min_p": {"values": [0.0]},
            },
        )


def test_multi_run_lora_rejects_sft_entrypoint(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="RL-only"):
        SweepConfig(
            entrypoint="sft",
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        )


def test_multi_run_lora_rejects_resume(tmp_path: Path) -> None:
    """Phase 7a: resume against a still-running shared trainer is deferred."""
    with pytest.raises(ValidationError, match="Resume is not supported"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "multi_run_lora",
                "max_concurrent_runs": 2,
                "shared": [tmp_path / "shared.toml"],
            },
            parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
            resume=True,
        )


def test_multi_run_lora_accepts_optuna_strategy(tmp_path: Path) -> None:
    """Phase 7b: Optuna + multi_run_lora is supported via the wave driver."""
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [tmp_path / "shared.toml"],
        },
        strategy={"type": "optuna", "num_trials": 4},
        parameters={"orchestrator.optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
    )
    assert config.strategy.type == "optuna"
    assert config.scheduler.type == "multi_run_lora"
