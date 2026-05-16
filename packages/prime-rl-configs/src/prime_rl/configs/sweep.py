import math
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Discriminator, Field, Tag, model_validator

from prime_rl.utils.config import BaseConfig

OPTUNA_SEED_MAX = 2**32 - 1


class ChoiceParameterConfig(BaseConfig):
    """Choice-valued parameter sampled from an explicit list."""

    distribution: Literal["choice"] = "choice"
    values: Annotated[list[Any], Field(description="Explicit values to sweep over.")]

    @model_validator(mode="after")
    def validate_values(self):
        if not self.values:
            raise ValueError("Sweep parameter values must be non-empty")
        for idx, value in enumerate(self.values):
            _validate_choice_value(value, f"values[{idx}]")
        return self


class UniformParameterConfig(BaseConfig):
    """Continuous parameter sampled uniformly on [min, max]."""

    distribution: Literal["uniform"]
    min: float
    max: float

    @model_validator(mode="before")
    @classmethod
    def reject_bool_bounds(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("min", "max"), "Uniform parameter")
        return data

    @model_validator(mode="after")
    def validate_range(self):
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("Uniform parameter min and max must be finite")
        if self.min >= self.max:
            raise ValueError("Uniform parameter requires min < max")
        if not math.isfinite(self.max - self.min):
            raise ValueError("Uniform parameter range must be finite")
        return self


class LogUniformParameterConfig(BaseConfig):
    """Continuous parameter sampled uniformly in log-space on [min, max]."""

    distribution: Literal["log_uniform"]
    min: float
    max: float

    @model_validator(mode="before")
    @classmethod
    def reject_bool_bounds(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("min", "max"), "Log-uniform parameter")
        return data

    @model_validator(mode="after")
    def validate_range(self):
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("Log-uniform parameter min and max must be finite")
        if self.min <= 0 or self.max <= 0:
            raise ValueError("Log-uniform parameter requires positive min and max")
        if self.min >= self.max:
            raise ValueError("Log-uniform parameter requires min < max")
        return self


class IntUniformParameterConfig(BaseConfig):
    """Integer parameter sampled uniformly from {min, min+step, ..., max}."""

    distribution: Literal["int_uniform"]
    min: int
    max: int
    step: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="before")
    @classmethod
    def reject_bool_bounds(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("min", "max", "step"), "Int-uniform parameter")
        return data

    @model_validator(mode="after")
    def validate_range(self):
        if self.min >= self.max:
            raise ValueError("Int-uniform parameter requires min < max")
        if (self.max - self.min) % self.step != 0:
            raise ValueError(
                f"Int-uniform range [{self.min}, {self.max}] is not divisible by step {self.step}; "
                "non-divisible ranges silently truncate the search space (the inclusive max is never sampled). "
                "Pick a step that divides (max - min) evenly."
            )
        return self


def _parameter_discriminator(value: Any) -> str:
    """Default to ``choice`` so the bare ``{"values": [...]}`` form keeps working."""
    if isinstance(value, dict):
        return value.get("distribution", "choice")
    return getattr(value, "distribution", "choice")


SweepParameterConfig: TypeAlias = Annotated[
    Annotated[ChoiceParameterConfig, Tag("choice")]
    | Annotated[UniformParameterConfig, Tag("uniform")]
    | Annotated[LogUniformParameterConfig, Tag("log_uniform")]
    | Annotated[IntUniformParameterConfig, Tag("int_uniform")],
    Discriminator(_parameter_discriminator),
]


def _validate_choice_value(value: Any, path: str) -> None:
    """Validate that a choice value can be written to generated TOML."""
    if value is None:
        raise ValueError(
            f"Sweep choice parameter {path} cannot be None; use the string 'None' for nullable target fields."
        )
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Sweep choice parameter {path} must be finite")
        return
    if isinstance(value, str):
        return
    if isinstance(value, dict):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise ValueError(
                f"Sweep choice parameter {path} has non-string TOML table key(s): {non_string_keys}"
            )
        for key, child in value.items():
            _validate_choice_value(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _validate_choice_value(child, f"{path}[{idx}]")
        return
    raise ValueError(
        f"Sweep choice parameter {path} has value of type {type(value).__name__}, "
        "which cannot be written to generated TOML."
    )


def _is_optuna_storage_safe_choice(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, int, float, str)):
        return not isinstance(value, float) or math.isfinite(value)
    return False


def _optuna_choices_are_equal(left: Any, right: Any) -> bool:
    return left == right


def _reject_bool_fields(data: Any, fields: tuple[str, ...], label: str) -> None:
    if not isinstance(data, dict):
        return
    bool_fields = [field for field in fields if isinstance(data.get(field), bool)]
    if bool_fields:
        raise ValueError(f"{label} numeric field(s) cannot be boolean: {bool_fields}")


def _choice_value_leaf_paths(parent_path: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [parent_path]
        paths: list[str] = []
        for key, child in value.items():
            child_path = f"{parent_path}.{key}"
            paths.extend(_choice_value_leaf_paths(child_path, child))
        return paths
    if isinstance(value, (list, tuple)):
        paths: list[str] = []
        for child in value:
            if isinstance(child, (dict, list, tuple)):
                paths.extend(_choice_value_leaf_paths(parent_path, child))
        return paths or [parent_path]
    return [parent_path]


def _choice_value_leaf_items(parent_path: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            return [(parent_path, value)]
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_path = f"{parent_path}.{key}"
            items.extend(_choice_value_leaf_items(child_path, child))
        return items
    if isinstance(value, (list, tuple)):
        items: list[tuple[str, Any]] = []
        for child in value:
            if isinstance(child, (dict, list, tuple)):
                items.extend(_choice_value_leaf_items(parent_path, child))
        return items or [(parent_path, value)]
    return [(parent_path, value)]


def _effective_parameter_paths(parameters: dict[str, "SweepParameterConfig"]) -> tuple[str, ...]:
    paths: list[str] = []
    for path, parameter in parameters.items():
        paths.append(path)
        if isinstance(parameter, ChoiceParameterConfig):
            for value in parameter.values:
                paths.extend(_choice_value_leaf_paths(path, value))
    return tuple(dict.fromkeys(paths))


class GridStrategyConfig(BaseConfig):
    """Exhaustive grid over choice-valued parameters."""

    type: Literal["grid"] = "grid"


class RandomStrategyConfig(BaseConfig):
    """Independent random samples from the declared parameter distributions."""

    type: Literal["random"] = "random"
    num_trials: Annotated[int, Field(ge=1, description="Number of trials to draw.")]
    seed: Annotated[int | None, Field(description="Optional seed for reproducibility.")] = None

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("num_trials", "seed"), "Random strategy")
        return data


class NoPrunerConfig(BaseConfig):
    """Disable pruning. Trials run to completion regardless of intermediate values."""

    type: Literal["none"] = "none"


class MedianPrunerConfig(BaseConfig):
    """Optuna's MedianPruner: prune trials whose intermediate value falls below
    the running median of completed trials at the same step."""

    type: Literal["median"] = "median"
    n_startup_trials: Annotated[
        int,
        Field(ge=0, description="Trials that must complete before pruning is enabled."),
    ] = 5
    n_warmup_steps: Annotated[
        int,
        Field(ge=0, description="Steps within a trial that are exempt from pruning."),
    ] = 0
    interval_steps: Annotated[
        int,
        Field(ge=1, description="Pruning is only checked every Nth reported step."),
    ] = 1

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(
            data,
            ("n_startup_trials", "n_warmup_steps", "interval_steps"),
            "Median pruner",
        )
        return data


class AshaPrunerConfig(BaseConfig):
    """Optuna's SuccessiveHalvingPruner (ASHA). Promotes trials whose intermediate
    value is in the top ``1/reduction_factor`` at each rung."""

    type: Literal["asha"] = "asha"
    min_resource: Annotated[
        int | Literal["auto"],
        Field(description="Minimum resource (steps) before a trial can be pruned."),
    ] = "auto"
    reduction_factor: Annotated[
        int,
        Field(ge=2, description="At each rung, keep the top 1/reduction_factor of trials."),
    ] = 4
    min_early_stopping_rate: Annotated[
        int,
        Field(ge=0, description="Bracket index offset; 0 enables the most aggressive bracket."),
    ] = 0

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(
            data,
            ("min_resource", "reduction_factor", "min_early_stopping_rate"),
            "ASHA pruner",
        )
        return data

    @model_validator(mode="after")
    def validate_min_resource(self):
        if isinstance(self.min_resource, int) and self.min_resource < 1:
            raise ValueError("ASHA pruner min_resource must be >= 1 or 'auto'")
        return self


class HyperbandPrunerConfig(BaseConfig):
    """Optuna's HyperbandPruner: runs successive-halving across multiple brackets."""

    type: Literal["hyperband"] = "hyperband"
    min_resource: Annotated[
        int,
        Field(ge=1, description="Smallest resource budget evaluated in any bracket."),
    ] = 1
    max_resource: Annotated[
        int | Literal["auto"],
        Field(description="Largest resource budget; ``auto`` infers from reported steps."),
    ] = "auto"
    reduction_factor: Annotated[
        int,
        Field(ge=2, description="At each rung, keep the top 1/reduction_factor of trials."),
    ] = 3

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(
            data,
            ("min_resource", "max_resource", "reduction_factor"),
            "Hyperband pruner",
        )
        return data

    @model_validator(mode="after")
    def validate_resources(self):
        if isinstance(self.max_resource, int) and self.max_resource < self.min_resource:
            raise ValueError("Hyperband pruner max_resource must be >= min_resource or 'auto'")
        return self


PrunerConfig: TypeAlias = Annotated[
    NoPrunerConfig | MedianPrunerConfig | AshaPrunerConfig | HyperbandPrunerConfig,
    Field(discriminator="type"),
]


class OptunaStrategyConfig(BaseConfig):
    """Adaptive sampling backed by Optuna.

    Samplers: ``tpe`` (default) and ``random``. Pruners: ``none`` (default),
    ``median``, ``asha`` (successive-halving), and ``hyperband``. Pruners need
    intermediate metric reporting from the trial; the controller polls a
    sidecar metrics stream while the trial runs and calls
    ``optuna_trial.report``/``should_prune`` between samples.

    Storage defaults to in-memory; pass a SQLAlchemy URL (e.g.
    ``"sqlite:///optuna.db"``) to persist the study across resume.
    """

    type: Literal["optuna"] = "optuna"
    num_trials: Annotated[int, Field(ge=1, description="Number of trials to evaluate.")]
    seed: int | None = None
    sampler: Literal["tpe", "random"] = "tpe"
    pruner: PrunerConfig = NoPrunerConfig()
    storage: Annotated[
        str | None,
        Field(description="SQLAlchemy storage URL for study persistence; in-memory if unset."),
    ] = None
    study_name: Annotated[
        str | None,
        Field(description="Optuna study_name; defaults to the sweep name."),
    ] = None
    poll_interval_seconds: Annotated[
        float,
        Field(
            gt=0,
            description=(
                "How often the controller polls the trial's intermediate metrics "
                "while pruning is enabled. Ignored when pruner.type == 'none'."
            ),
        ),
    ] = 5.0

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("num_trials", "seed", "poll_interval_seconds"), "Optuna strategy")
        return data

    @model_validator(mode="after")
    def validate_optuna_fields(self):
        if not math.isfinite(self.poll_interval_seconds):
            raise ValueError("Optuna poll_interval_seconds must be finite")
        if self.seed is not None and not 0 <= self.seed <= OPTUNA_SEED_MAX:
            raise ValueError(f"Optuna seed must be between 0 and {OPTUNA_SEED_MAX}")
        if self.storage is not None and not self.storage.strip():
            raise ValueError("Optuna storage must be a non-empty SQLAlchemy URL when set")
        return self


SearchStrategyConfig: TypeAlias = Annotated[
    GridStrategyConfig | RandomStrategyConfig | OptunaStrategyConfig,
    Field(discriminator="type"),
]


class LocalGpuAssignmentConfig(BaseConfig):
    """Static round-robin assignment of CUDA_VISIBLE_DEVICES to local workers.

    Each entry in ``visible_devices`` is one device group that pins one trial
    subprocess. Groups are disjoint by construction so two parallel workers
    never share a GPU. ``mode`` is currently fixed to ``"static"``; future
    modes (``"exclusive"`` for live GPU discovery, ``"none"`` to leave
    ``CUDA_VISIBLE_DEVICES`` untouched) will land in later phases.
    """

    mode: Literal["static"] = "static"
    visible_devices: Annotated[
        list[list[int]],
        Field(min_length=1, description="Disjoint device groups assigned to parallel workers."),
    ]

    @model_validator(mode="before")
    @classmethod
    def reject_bool_devices(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        groups = data.get("visible_devices")
        if not isinstance(groups, list):
            return data
        for group_idx, group in enumerate(groups):
            if not isinstance(group, list):
                continue
            for device_idx, device in enumerate(group):
                if isinstance(device, bool):
                    raise ValueError(
                        "Local GPU assignment visible_devices entries cannot be boolean: "
                        f"visible_devices[{group_idx}][{device_idx}]"
                    )
        return data

    @model_validator(mode="after")
    def validate_groups(self):
        if any(not group for group in self.visible_devices):
            raise ValueError("Each visible_devices group must contain at least one device index")
        flat = [device for group in self.visible_devices for device in group]
        if any(device < 0 for device in flat):
            raise ValueError("visible_devices indices must be non-negative")
        if len(flat) != len(set(flat)):
            raise ValueError("Each device may only appear in one visible_devices group")
        return self


class LocalSweepSchedulerConfig(BaseConfig):
    """Run generated trials as local subprocesses."""

    type: Literal["local"] = "local"

    max_parallel: Annotated[int, Field(ge=1, description="Maximum local trials to run concurrently.")] = 1
    gpu_assignment: Annotated[
        LocalGpuAssignmentConfig | None,
        Field(description="Required for max_parallel > 1; pins each worker to a disjoint device group."),
    ] = None

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("max_parallel",), "Local scheduler")
        return data

    @model_validator(mode="after")
    def validate_parallel(self):
        if self.max_parallel > 1:
            if self.gpu_assignment is None:
                raise ValueError(
                    "max_parallel > 1 requires explicit gpu_assignment so parallel workers do not "
                    "silently colocate trainer/inference stacks on the same GPUs."
                )
            available = len(self.gpu_assignment.visible_devices)
            if available < self.max_parallel:
                raise ValueError(
                    f"max_parallel={self.max_parallel} requires at least {self.max_parallel} "
                    f"visible_devices groups, got {available}."
                )
        return self


class SlurmSweepSchedulerConfig(BaseConfig):
    """Submit generated trials through the target entrypoint's SLURM support.

    Throughput is governed by the SLURM cluster. When ``synchronous = true``
    the controller blocks on each trial; with ``max_parallel > 1`` the
    controller drives up to N concurrent trials through ``sbatch
    --parsable`` and shared-FS polling so Optuna's ask/tell loop can fill
    the cluster instead of serializing one trial at a time.

    When ``synchronous = true``, the controller submits each trial via
    ``sbatch --wait`` (or ``sbatch --parsable`` + polling in pruning /
    parallel mode) and observes per-trial completion. This lets Optuna and
    trial-level early stopping work over SLURM (the controller learns each
    trial's objective before proposing the next), at the cost of trials
    being scheduled at the controller's pace rather than the cluster's
    queue cadence.
    """

    type: Literal["slurm"] = "slurm"
    synchronous: Annotated[
        bool,
        Field(
            description=(
                "Block on each sbatch submission via 'sbatch --wait' so the "
                "controller observes per-trial completion. Required to pair "
                "Optuna or early stopping with the SLURM scheduler."
            ),
        ),
    ] = False
    max_parallel: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Maximum concurrent in-flight SLURM jobs the controller will "
                "manage. Only meaningful with synchronous=true; the controller "
                "submits up to this many trials, polls each via shared-FS "
                "metrics.jsonl + squeue, and replaces them with fresh Optuna "
                "asks as they complete. With TPE this enables constant_liar "
                "sampling so concurrent asks don't collide on the same region."
            ),
        ),
    ] = 1


# Parameter paths a multi_run_lora sweep is allowed to vary. Must stay in
# sync with both the OrchestratorConfig schema (paths must actually resolve)
# and what the trainer's MultiRunManager treats as per-run-safe; see
# src/prime_rl/trainer/runs.py for the runtime validation hook. Anything
# under trainer.*, model.*, deployment.*, or inference.* is shared across
# runs and would silently mismatch between trials, so it is rejected at
# config-load time.
#
# Prefixes match arbitrary dict subtrees (paths must continue under them);
# fields match concrete schema leaves exactly. Splitting them avoids letting
# bogus paths like ``orchestrator.batch_size_extra`` slip through a
# startswith check.
#
# Targets that resolve to a list (e.g. ``orchestrator.train.env``,
# ``orchestrator.eval.env``) cannot be allowlisted: the sweep materializer's
# ``set_dotted_path`` only walks dict tables, so a path like
# ``orchestrator.train.env.id`` would produce a dict-shaped override that
# RLConfig rejects with "Input should be a valid list".
#
# Fields coupled to the shared trainer (e.g. ``orchestrator.max_steps`` and
# ``orchestrator.max_async_level``) are intentionally not allowlisted. The
# shared trainer owns the actual loop length and weight-broadcast retention
# window for all runs in the wave.
MULTI_RUN_LORA_PARAMETER_PREFIXES: tuple[str, ...] = (
    "orchestrator.train.sampling.extra_body.",
    "orchestrator.eval.sampling.extra_body.",
)
MULTI_RUN_LORA_PARAMETER_FIELDS: frozenset[str] = frozenset(
    {
        "orchestrator.optim.lr",
        "orchestrator.model.lora.name",
        "orchestrator.model.lora.rank",
        "orchestrator.model.lora.alpha",
        "orchestrator.batch_size",
        "orchestrator.token_batch_size",
        "orchestrator.oversampling_factor",
        "orchestrator.max_inflight_rollouts",
        "orchestrator.rollouts_per_example",
        "orchestrator.max_off_policy_steps",
        "orchestrator.strict_async_level",
        "orchestrator.seed",
        "orchestrator.tasks_per_minute",
        "orchestrator.train.sampling",
        "orchestrator.train.sampling.temperature",
        "orchestrator.train.sampling.repetition_penalty",
        "orchestrator.train.sampling.max_completion_tokens",
        "orchestrator.train.sampling.max_tokens",
        "orchestrator.train.sampling.min_tokens",
        "orchestrator.train.sampling.seed",
        "orchestrator.train.sampling.extra_body",
        "orchestrator.train.num_workers",
        "orchestrator.train.max_retries",
        "orchestrator.eval.sampling",
        "orchestrator.eval.sampling.temperature",
        "orchestrator.eval.sampling.repetition_penalty",
        "orchestrator.eval.sampling.top_p",
        "orchestrator.eval.sampling.top_k",
        "orchestrator.eval.sampling.min_p",
        "orchestrator.eval.sampling.max_completion_tokens",
        "orchestrator.eval.sampling.max_tokens",
        "orchestrator.eval.sampling.min_tokens",
        "orchestrator.eval.sampling.reasoning_effort",
        "orchestrator.eval.sampling.seed",
        "orchestrator.eval.sampling.extra_body",
        "orchestrator.eval.num_examples",
        "orchestrator.eval.rollouts_per_example",
        "orchestrator.eval.num_workers",
        "orchestrator.eval.max_retries",
        "orchestrator.eval.interval",
        "orchestrator.eval.eval_base_model",
        "orchestrator.eval.skip_eval_on_resume",
        "orchestrator.eval.cancel_inflight_rollouts_on_eval",
        # BufferConfig: scalar leaves, plus hash_keys as a whole-list swap.
        # Sub-paths under hash_keys (it's a list[str]) are unreachable via
        # set_dotted_path, so only the exact field is allowlisted.
        "orchestrator.buffer.seed",
        "orchestrator.buffer.easy_threshold",
        "orchestrator.buffer.hard_threshold",
        "orchestrator.buffer.easy_fraction",
        "orchestrator.buffer.hard_fraction",
        "orchestrator.buffer.online_difficulty_filtering",
        "orchestrator.buffer.hash_keys",
    }
)
MULTI_RUN_LORA_LIST_PARAMETER_FIELDS: frozenset[str] = frozenset({"orchestrator.buffer.hash_keys"})
MULTI_RUN_LORA_DICT_PARAMETER_FIELDS: frozenset[str] = frozenset(
    {
        "orchestrator.train.sampling",
        "orchestrator.train.sampling.extra_body",
        "orchestrator.eval.sampling",
        "orchestrator.eval.sampling.extra_body",
    }
)
MULTI_RUN_LORA_SCALAR_PARAMETER_FIELDS: frozenset[str] = (
    MULTI_RUN_LORA_PARAMETER_FIELDS
    - MULTI_RUN_LORA_LIST_PARAMETER_FIELDS
    - MULTI_RUN_LORA_DICT_PARAMETER_FIELDS
)


def _multi_run_lora_exact_field_shape_errors(parameters: dict[str, SweepParameterConfig]) -> list[str]:
    errors: list[str] = []
    for path, parameter in parameters.items():
        if path.startswith(MULTI_RUN_LORA_PARAMETER_PREFIXES):
            continue
        if path not in MULTI_RUN_LORA_PARAMETER_FIELDS:
            continue
        if not isinstance(parameter, ChoiceParameterConfig):
            if path in MULTI_RUN_LORA_LIST_PARAMETER_FIELDS or path in MULTI_RUN_LORA_DICT_PARAMETER_FIELDS:
                errors.append(f"{path}: list/table fields must use explicit choice values")
            continue
        leaf_values: dict[str, list[Any]] = {}
        for value in parameter.values:
            leaf_items = _choice_value_leaf_items(path, value)
            if all(leaf_path != path for leaf_path, _ in leaf_items):
                leaf_values.setdefault(path, []).append(value)
            for leaf_path, leaf_value in leaf_items:
                leaf_values.setdefault(leaf_path, []).append(leaf_value)
        for leaf_path, values in leaf_values.items():
            if leaf_path.startswith(MULTI_RUN_LORA_PARAMETER_PREFIXES):
                continue
            if leaf_path not in MULTI_RUN_LORA_PARAMETER_FIELDS:
                continue
            if leaf_path in MULTI_RUN_LORA_SCALAR_PARAMETER_FIELDS:
                bad_values = [value for value in values if isinstance(value, (dict, list, tuple))]
                if bad_values:
                    errors.append(f"{leaf_path}: scalar field cannot use structured choice value(s) {bad_values!r}")
            elif leaf_path in MULTI_RUN_LORA_LIST_PARAMETER_FIELDS:
                bad_values = [
                    value
                    for value in values
                    if not isinstance(value, (list, tuple))
                    or not value
                    or any(not isinstance(item, str) for item in value)
                ]
                if bad_values:
                    errors.append(f"{leaf_path}: must use non-empty list[str] choice value(s), got {bad_values!r}")
            elif leaf_path in MULTI_RUN_LORA_DICT_PARAMETER_FIELDS:
                bad_values = [value for value in values if not isinstance(value, dict)]
                if bad_values:
                    errors.append(f"{leaf_path}: must use table/dict choice value(s), got {bad_values!r}")
    return errors


class MultiRunLoRASchedulerConfig(BaseConfig):
    """Run all trials concurrently against one shared trainer + inference.

    Static sweeps launch a single ``rl-multi-run`` invocation that brings up
    one trainer (with ``trainer.max_concurrent_runs >= num_trials``), one
    inference server, and ``num_trials`` orchestrators — one per trial.
    Optuna sweeps run in waves so in-flight trials can be pruned between
    intermediate metric reports. Resume against a still-running trainer is
    intentionally deferred to a later phase.
    """

    type: Literal["multi_run_lora"] = "multi_run_lora"
    max_concurrent_runs: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Number of concurrent orchestrator runs against the shared trainer. "
                "Must match (or be <=) trainer.max_concurrent_runs in the shared base config."
            ),
        ),
    ]
    shared: Annotated[
        list[Path],
        Field(
            min_length=1,
            description=(
                "RLConfig base TOML(s) describing the shared trainer + inference. "
                "Trial overrides apply to the orchestrator block only."
            ),
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("max_concurrent_runs",), "multi_run_lora scheduler")
        return data


SweepSchedulerConfig: TypeAlias = Annotated[
    LocalSweepSchedulerConfig | SlurmSweepSchedulerConfig | MultiRunLoRASchedulerConfig,
    Field(discriminator="type"),
]


class SweepWandbConfig(BaseConfig):
    """W&B metadata injected into generated trials."""

    enabled: bool = True
    group: str | None = None
    tags: list[str] = ["sweep"]


class ObjectiveConfig(BaseConfig):
    """Names the metric the sweep optimizes and where to read it from."""

    metric: Annotated[
        str,
        Field(description="Metric key inside final_summary.json (forward-slash-separated)."),
    ]
    direction: Literal["maximize", "minimize"]
    source: Literal["final_summary"] = "final_summary"

    @model_validator(mode="after")
    def validate_metric(self):
        if not self.metric.strip():
            raise ValueError("objective.metric must be non-empty")
        return self


class ThresholdStoppingConfig(BaseConfig):
    """Halt the study after a trial whose objective is on the wrong side of a threshold."""

    type: Literal["threshold"] = "threshold"
    threshold: float
    min_trials: Annotated[
        int,
        Field(ge=1, description="Minimum completed trials before threshold can fire."),
    ] = 1

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("threshold", "min_trials"), "Early-stopping threshold")
        return data

    @model_validator(mode="after")
    def validate_threshold(self):
        if not math.isfinite(self.threshold):
            raise ValueError("Early-stopping threshold must be finite")
        return self


class PatienceStoppingConfig(BaseConfig):
    """Halt the study after N consecutive completed trials with no improvement."""

    type: Literal["patience"] = "patience"
    patience: Annotated[int, Field(ge=1, description="Consecutive non-improving trials required to halt.")]
    min_trials: Annotated[
        int,
        Field(ge=1, description="Minimum completed trials before patience can fire."),
    ] = 1

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("patience", "min_trials"), "Patience early stopping")
        return data


EarlyStoppingConfig: TypeAlias = Annotated[
    ThresholdStoppingConfig | PatienceStoppingConfig,
    Field(discriminator="type"),
]


class SweepConfig(BaseConfig):
    """Configures a hyperparameter sweep study."""

    name: str | None = None
    entrypoint: Literal["rl", "sft"] = "rl"
    base: list[Path]
    output_dir: Path
    strategy: SearchStrategyConfig = GridStrategyConfig()
    scheduler: SweepSchedulerConfig = LocalSweepSchedulerConfig()
    parameters: dict[str, SweepParameterConfig]
    wandb: SweepWandbConfig | None = SweepWandbConfig()
    objective: ObjectiveConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None
    continue_on_failure: Annotated[
        bool,
        Field(description="Schedule remaining trials when one fails. Set false to halt-on-first-fail."),
    ] = True
    retry_budget: Annotated[
        int,
        Field(ge=0, description="Retry a failed trial up to this many times before marking it failed."),
    ] = 1
    resume: Annotated[
        bool,
        Field(description="Reattach to an existing study output dir; preserve completed trial state."),
    ] = False
    dry_run: bool = False
    clean_output_dir: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_bool_numeric_fields(cls, data: Any) -> Any:
        _reject_bool_fields(data, ("retry_budget",), "Sweep")
        return data

    @model_validator(mode="after")
    def validate_sweep(self):
        if not self.base:
            raise ValueError("Sweep base must include at least one target config file")
        if not self.parameters:
            raise ValueError("Sweep parameters must include at least one parameter")
        paths = tuple(self.parameters)
        effective_paths = _effective_parameter_paths(self.parameters)
        effective_path_set = set(effective_paths)
        invalid_paths = [
            path for path in effective_paths if not path or any(not part for part in path.split("."))
        ]
        if invalid_paths:
            raise ValueError(
                "Sweep parameter paths must be non-empty dot-separated config segments. "
                f"Invalid path(s): {invalid_paths}"
            )
        output_dir_paths = [path for path in effective_paths if "output_dir" in path.split(".")]
        if output_dir_paths:
            raise ValueError(
                "Sweep parameters cannot set output_dir fields; "
                "the sweep materializer owns each trial's output directories. "
                f"Invalid path(s): {output_dir_paths}"
            )
        if self.wandb is not None and self.wandb.enabled:
            managed_wandb_paths = ("wandb.group", "wandb.name", "wandb.tags")
            managed_component_wandb_paths = tuple(
                f"{component}.wandb.{field}"
                for component in ("trainer", "orchestrator")
                for field in ("project", "entity", "name", "group", "tags", "offline")
            )
            managed_component_wandb_tables = ("trainer.wandb", "orchestrator.wandb")
            wandb_paths = [
                path
                for path in effective_paths
                if path == "wandb"
                or any(path == managed or path.startswith(f"{managed}.") for managed in managed_wandb_paths)
                or path in managed_component_wandb_tables
                or any(
                    path == managed or path.startswith(f"{managed}.")
                    for managed in managed_component_wandb_paths
                )
            ]
            if wandb_paths:
                raise ValueError(
                    "Sweep parameters cannot set sweep-managed W&B identity/shared fields while sweep wandb "
                    f"injection is enabled. Disable [wandb] injection or remove path(s): {wandb_paths}"
                )
        path_conflicts = [
            (parent, child)
            for parent in paths
            for child in paths
            if parent != child and child.startswith(f"{parent}.")
        ]
        if path_conflicts:
            raise ValueError(
                "Sweep parameters cannot include both a parent path and one of its sub-paths: "
                f"{path_conflicts}. Split these into separate sweeps or choose one override shape."
            )
        if self.resume and self.clean_output_dir:
            warnings.warn(
                "resume=true takes precedence over clean_output_dir=true; "
                "ignoring clean_output_dir so existing trial state is preserved.",
                stacklevel=2,
            )
            self.clean_output_dir = False
        if isinstance(self.strategy, GridStrategyConfig):
            non_choice = [
                path for path, parameter in self.parameters.items() if not isinstance(parameter, ChoiceParameterConfig)
            ]
            if non_choice:
                raise ValueError(
                    "Grid strategy only supports choice (values=...) parameters, "
                    f"but these declare distributions instead: {non_choice}"
                )
        if self.resume and isinstance(self.strategy, RandomStrategyConfig) and self.strategy.seed is None:
            raise ValueError(
                "resume requires a deterministic trial set, but the random strategy has no seed. "
                "Set strategy.seed so trial IDs match the previous study, or drop resume."
            )
        if self.early_stopping is not None and self.objective is None:
            raise ValueError(
                "early_stopping requires an objective so the controller knows which metric to compare."
            )
        if (
            self.early_stopping is not None
            and isinstance(self.scheduler, SlurmSweepSchedulerConfig)
            and not self.scheduler.synchronous
        ):
            raise ValueError(
                "early_stopping is not supported with the SLURM scheduler unless "
                "scheduler.synchronous=true: the asynchronous SLURM scheduler submits "
                "jobs and exits, so it never observes trial completion to decide when to halt."
            )
        if (
            isinstance(self.scheduler, SlurmSweepSchedulerConfig)
            and self.scheduler.max_parallel > 1
            and not self.scheduler.synchronous
        ):
            raise ValueError(
                "scheduler.max_parallel > 1 requires scheduler.synchronous=true: the "
                "controller cannot manage concurrent in-flight jobs without observing each "
                "one's terminal state, which is what the synchronous mode provides."
            )
        if (
            self.early_stopping is not None
            and isinstance(self.scheduler, MultiRunLoRASchedulerConfig)
            and not isinstance(self.strategy, OptunaStrategyConfig)
        ):
            raise ValueError(
                "early_stopping is not supported with static multi_run_lora sweeps: the controller "
                "launches the whole grid/random wave at once, so it cannot stop future trials. "
                "Use the Optuna strategy for wave-by-wave multi_run_lora early stopping."
            )
        if isinstance(self.strategy, OptunaStrategyConfig):
            if self.objective is None:
                raise ValueError("Optuna strategy requires an objective to optimize.")
            if (
                isinstance(self.scheduler, SlurmSweepSchedulerConfig)
                and not self.scheduler.synchronous
            ):
                raise ValueError(
                    "Optuna strategy is not supported with the asynchronous SLURM scheduler: "
                    "the controller must observe each trial's objective before proposing the "
                    "next one. Set scheduler.synchronous=true to submit each trial with "
                    "'sbatch --wait' so the controller blocks per trial."
                )
            if isinstance(self.scheduler, LocalSweepSchedulerConfig) and self.scheduler.max_parallel > 1:
                raise ValueError(
                    "Optuna strategy on the local scheduler runs sequentially (ask/tell needs "
                    "each trial's objective before proposing the next), so scheduler.max_parallel "
                    "must be 1. Use scheduler.type='slurm' with synchronous=true to drive "
                    "max_parallel > 1 over SLURM."
                )
            if self.resume and self.strategy.storage is None:
                raise ValueError(
                    "Resume with the Optuna strategy requires strategy.storage so the study "
                    "can be reloaded; in-memory studies vanish when the controller exits."
                )
            storage_unsafe_choice_paths = [
                path
                for path, parameter in self.parameters.items()
                if isinstance(parameter, ChoiceParameterConfig)
                and not all(_is_optuna_storage_safe_choice(value) for value in parameter.values)
            ]
            if storage_unsafe_choice_paths:
                raise ValueError(
                    "Optuna categorical parameters only support storage-safe primitive choices "
                    f"(bool, int, finite float, or str). Invalid parameter path(s): {storage_unsafe_choice_paths}"
                )
            ambiguous_choice_paths = []
            for path, parameter in self.parameters.items():
                if not isinstance(parameter, ChoiceParameterConfig):
                    continue
                if any(
                    _optuna_choices_are_equal(left, right)
                    for idx, left in enumerate(parameter.values)
                    for right in parameter.values[idx + 1 :]
                ):
                    ambiguous_choice_paths.append(path)
            if ambiguous_choice_paths:
                raise ValueError(
                    "Optuna categorical parameters cannot include duplicate or equality-colliding "
                    f"choices because Optuna storage cannot distinguish them. Invalid parameter path(s): "
                    f"{ambiguous_choice_paths}"
                )
        if isinstance(self.scheduler, MultiRunLoRASchedulerConfig):
            if self.entrypoint != "rl":
                raise ValueError(
                    "multi_run_lora scheduler is RL-only; the shared-trainer architecture "
                    "depends on the trainer's MultiRunManager which only the rl entrypoint runs."
                )
            if self.resume:
                raise ValueError(
                    "Resume is not supported with the multi_run_lora scheduler in Phase 7b; "
                    "re-attaching to a still-running shared trainer needs reconciliation work "
                    "that lands in Phase 7c."
                )
            offending = [
                path
                for path in effective_paths
                if path not in MULTI_RUN_LORA_PARAMETER_FIELDS
                and not any(path.startswith(prefix) for prefix in MULTI_RUN_LORA_PARAMETER_PREFIXES)
            ]
            if offending:
                allowed = ", ".join(
                    (*MULTI_RUN_LORA_PARAMETER_PREFIXES, *sorted(MULTI_RUN_LORA_PARAMETER_FIELDS))
                )
                raise ValueError(
                    "multi_run_lora sweeps may only vary per-run orchestrator fields. "
                    f"These parameter paths are not in the allowlist ({allowed}): {offending}. "
                    "Trainer/model/deployment/inference settings cannot vary inside one shared trainer."
                )
            shape_errors = _multi_run_lora_exact_field_shape_errors(self.parameters)
            if shape_errors:
                raise ValueError(
                    "multi_run_lora exact-field sweep parameters must match the allowlisted field shape. "
                    f"Invalid value(s): {shape_errors}"
                )
            if (
                "orchestrator.batch_size" in self.parameters
                and "orchestrator.token_batch_size" in self.parameters
            ):
                raise ValueError(
                    "multi_run_lora sweeps must set either orchestrator.batch_size or "
                    "orchestrator.token_batch_size, not both."
                )
            if (
                "orchestrator.token_batch_size" in self.parameters
                and "orchestrator.oversampling_factor" in self.parameters
            ):
                raise ValueError(
                    "multi_run_lora sweeps cannot set orchestrator.oversampling_factor with "
                    "orchestrator.token_batch_size; oversampling only applies to rollout batching."
                )
            for prefix in ("orchestrator.train.sampling", "orchestrator.eval.sampling"):
                has_canonical = f"{prefix}.max_completion_tokens" in effective_path_set
                has_alias = f"{prefix}.max_tokens" in effective_path_set
                if has_canonical and has_alias:
                    raise ValueError(
                        f"multi_run_lora sweeps must set either {prefix}.max_completion_tokens "
                        f"or {prefix}.max_tokens, not both."
                    )
        return self
