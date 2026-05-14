import hashlib
import json
import shlex
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomli_w

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.metrics import coerce_finite_float
from prime_rl.sweep.reproducibility import file_checksum
from prime_rl.utils.config import BaseConfig, cli


@dataclass(frozen=True)
class Trial:
    id: str
    label: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TrialArtifacts:
    trial: Trial
    trial_dir: Path
    run_dir: Path
    overrides_path: Path
    resolved_path: Path
    command_path: Path
    status_path: Path
    command: list[str]
    resolved_checksum: str
    base_checksums: dict[str, str]


def set_dotted_path(data: dict[str, Any], path: str, value: Any) -> None:
    if not path:
        raise ValueError("Sweep parameter path cannot be empty")

    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {path}: {part} is already set to a non-table value")
        current = child
    current[parts[-1]] = value


def build_nested_overrides(flat_overrides: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in flat_overrides.items():
        set_dotted_path(nested, path, value)
    return nested


def merge_nested_overrides(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merge_nested_overrides(current, value)
        else:
            target[key] = value


def _load_nested_toml(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        merge_nested_overrides(merged, data)
    return merged


def sanitize_label_part(value: Any) -> str:
    text = str(value)
    for char in ("/", "\\", " ", ":", ",", "[", "]", "{", "}", "'", '"'):
        text = text.replace(char, "_")
    return text


def trial_label(parameters: dict[str, Any], max_len: int = 96) -> str:
    parts = []
    for path, value in parameters.items():
        name = path.split(".")[-1].replace("_", "-")
        parts.append(f"{name}_{sanitize_label_part(value)}")
    label = "-".join(parts)
    return label if len(label) <= max_len else ""


def command_for_trial(
    entrypoint: Literal["rl", "sft"],
    base_paths: list[Path],
    overrides_path: Path,
) -> list[str]:
    """Compose the launcher command from base files plus the generated overrides.

    This matches the form a user would type by hand and keeps per-trial diffs
    small. The frozen ``resolved.toml`` is written separately as a reproducible
    artifact but is not used as the launch input.
    """
    cmd = ["uv", "run", entrypoint]
    for base in base_paths:
        cmd.extend(["@", base.as_posix()])
    cmd.extend(["@", overrides_path.as_posix()])
    return cmd


def write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def toml_checksum(data: dict[str, Any]) -> str:
    """SHA-256 hex digest of the TOML bytes this module writes for ``data``."""
    return hashlib.sha256(tomli_w.dumps(data).encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


class SweepStatusError(RuntimeError):
    """Raised when a sweep status artifact cannot be trusted."""


def read_status_json(status_path: Path) -> dict[str, Any]:
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError as exc:
        raise SweepStatusError(
            f"Sweep status file at {status_path} is not valid JSON. "
            "Restore the trial artifacts or remove the trial directory."
        ) from exc
    if not isinstance(status, dict):
        raise SweepStatusError(
            f"Sweep status file at {status_path} must be a JSON object, "
            f"not {type(status).__name__}. Restore the trial artifacts or remove the trial directory."
        )
    return status


def _materialization_error(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def record_trial_objective(status_path: Path, value: float | None) -> None:
    """Persist an objective value into a trial's status.json, preserving other fields."""
    status = read_status_json(status_path)
    status["objective"] = coerce_finite_float(value)
    write_json(status_path, status)


def record_trial_missing_objective(status_path: Path, metric: str) -> None:
    """Mark a clean process exit as failed when it did not produce the sweep metric."""
    status = read_status_json(status_path)
    status["state"] = "failed"
    status["objective"] = None
    status["failure_stage"] = "objective"
    status["error"] = f"Trial exited successfully but did not record a finite objective for {metric!r}."
    if status.get("returncode") is None:
        status["returncode"] = 0
    write_json(status_path, status)


def record_trial_pruned(
    status_path: Path,
    step: int,
    value: float,
    *,
    returncode: int | None = None,
    finished_at: str | None = None,
) -> None:
    """Mark a trial as pruned by intermediate-metric reporting.

    The Optuna pruning loop terminates the trial subprocess when the sampler
    decides the trajectory is unpromising. We record state="pruned" plus the
    step/value the prune fired on so the manifest can tell pruned trials
    apart from completed and failed runs without re-deriving the cause.
    """
    status = read_status_json(status_path)
    status["state"] = "pruned"
    status["pruned_at_step"] = int(step)
    status["pruned_value"] = float(value)
    status["objective"] = None
    if returncode is not None:
        status["returncode"] = int(returncode)
    if finished_at is not None:
        status["finished_at"] = finished_at
    write_json(status_path, status)


def validate_target_config(entrypoint: Literal["rl", "sft"], args: list[str]) -> BaseConfig:
    config_cls = RLConfig if entrypoint == "rl" else SFTConfig
    return cli(config_cls, args=args)


def _validate_target_with_overrides(
    entrypoint: Literal["rl", "sft"],
    base_paths: list[Path],
    overrides_path: Path,
    overrides: dict[str, Any],
    *,
    avoid_overwriting: bool,
) -> BaseConfig:
    args: list[str] = []
    for base_path in base_paths:
        args.extend(["@", base_path.as_posix()])
    if not avoid_overwriting:
        write_toml(overrides_path, overrides)
        args.extend(["@", overrides_path.as_posix()])
        return validate_target_config(entrypoint, args)

    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=overrides_path.parent,
        prefix=f".{overrides_path.name}.",
        suffix=".tmp.toml",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)
    try:
        write_toml(temp_path, overrides)
        args.extend(["@", temp_path.as_posix()])
        return validate_target_config(entrypoint, args)
    finally:
        temp_path.unlink(missing_ok=True)


def _target_config_to_toml(config: BaseConfig) -> dict[str, Any]:
    return config.model_dump(exclude_none=True, mode="json")


def _get_nested_value(data: dict[str, Any], segments: tuple[str | int, ...]) -> tuple[bool, Any]:
    current: Any = data
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):
                return False, None
            current = current[segment]
        else:
            if not isinstance(current, dict) or segment not in current:
                return False, None
            current = current[segment]
    return True, current


def _canonical_bool_lookup_segments(
    segments: tuple[str | int, ...],
) -> tuple[tuple[str | int, ...], ...]:
    aliases = {
        "max_tokens": "max_completion_tokens",
        "skip_eval_on_restart": "skip_eval_on_resume",
        "timeout_seconds": "timeout",
    }
    leaf = segments[-1] if segments else None
    canonical_leaf = aliases.get(leaf) if isinstance(leaf, str) else None
    if canonical_leaf is None:
        return (segments,)
    return (segments, (*segments[:-1], canonical_leaf))


def _bool_parameter_targets(
    display_path: str,
    lookup_segments: tuple[str | int, ...],
    value: Any,
) -> list[tuple[str, tuple[str | int, ...], bool]]:
    if isinstance(value, bool):
        return [(display_path, lookup_segments, value)]
    if isinstance(value, dict):
        targets: list[tuple[str, tuple[str | int, ...], bool]] = []
        for key, child in value.items():
            child_path = f"{display_path}.{key}"
            targets.extend(_bool_parameter_targets(child_path, (*lookup_segments, key), child))
        return targets
    if isinstance(value, (list, tuple)):
        targets: list[tuple[str, tuple[str | int, ...], bool]] = []
        for idx, child in enumerate(value):
            child_path = f"{display_path}[{idx}]"
            targets.extend(_bool_parameter_targets(child_path, (*lookup_segments, idx), child))
        return targets
    return []


def _reject_bool_target_coercions(
    parameters: dict[str, Any],
    resolved_data: dict[str, Any],
    *,
    path_prefix: str = "",
) -> None:
    """Reject bool choice values when the target field resolved as non-bool."""
    for path, value in parameters.items():
        lookup_path = path
        if path_prefix and lookup_path.startswith(path_prefix):
            lookup_path = lookup_path.removeprefix(path_prefix)
        lookup_segments: tuple[str | int, ...] = tuple(lookup_path.split("."))
        for display_path, segments, bool_value in _bool_parameter_targets(path, lookup_segments, value):
            for candidate in _canonical_bool_lookup_segments(segments):
                found, resolved_value = _get_nested_value(resolved_data, candidate)
                if not found:
                    continue
                if not isinstance(resolved_value, bool):
                    raise ValueError(
                        f"Sweep parameter {display_path!r} uses boolean value {bool_value!r}, "
                        f"but the resolved target field is {type(resolved_value).__name__}. "
                        "Boolean choice values are only valid for boolean target fields."
                    )
                break


def _merge_wandb_overrides(config: SweepConfig, flat_overrides: dict[str, Any], trial: Trial) -> None:
    if config.wandb is None or not config.wandb.enabled:
        return

    group = config.wandb.group or config.name
    if group is not None:
        flat_overrides["wandb.group"] = group

    flat_overrides["wandb.name"] = trial.label or trial.id

    tags = list(dict.fromkeys([*config.wandb.tags, "sweep", f"trial:{trial.id}"]))
    if config.name is not None:
        tags.append(f"study:{config.name}")
    flat_overrides["wandb.tags"] = list(dict.fromkeys(tags))


TERMINAL_RESUME_STATES = frozenset({"completed", "submitted"})


class SweepDriftError(RuntimeError):
    """Raised when --resume would skip a trial whose effective config has changed."""


def _existing_terminal_status(status_path: Path) -> dict[str, Any] | None:
    """Return parsed status.json if its state should be preserved on resume."""
    if not status_path.exists():
        return None
    status = read_status_json(status_path)
    if status.get("state") in TERMINAL_RESUME_STATES:
        return status
    return None


def _check_resume_drift(
    trial: Trial,
    preserved_status: dict[str, Any],
    expected: dict[str, Any] | None,
    new_resolved_checksum: str,
    new_base_checksums: dict[str, str],
) -> None:
    """Refuse to skip a terminal trial whose recorded config differs from the live one.

    Trial IDs hash sweep parameters only, so a base TOML edit between runs
    leaves the ID stable while the resolved config changes underneath us.
    Without this check ``--resume`` would silently honor the old ``status.json``
    and skip work that no longer reflects the current configuration.
    """
    if expected is None:
        raise SweepDriftError(
            f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
            "the previous manifest has no checksum entry for it. Drop --resume to start fresh, "
            "restore the manifest, or remove the trial directory."
        )

    expected_resolved = expected.get("resolved_checksum")
    expected_bases = expected.get("base_checksums") or {}
    if expected_resolved is None or not isinstance(expected_bases, dict):
        raise SweepDriftError(
            f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
            "its previous manifest entry is missing resolved/base checksums. Drop --resume to "
            "start fresh, restore the manifest, or remove the trial directory."
        )

    missing_bases = [base for base in new_base_checksums if base not in expected_bases]
    if missing_bases:
        raise SweepDriftError(
            f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
            f"its previous manifest entry is missing checksum(s) for base file(s): {missing_bases}. "
            "Drop --resume to start fresh, restore the manifest, or remove the trial directory."
        )

    extra_bases = [base for base in expected_bases if base not in new_base_checksums]
    if extra_bases:
        raise SweepDriftError(
            f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
            f"its previous manifest entry has extra base file(s) no longer present: {extra_bases}. "
            "Drop --resume to start fresh, restore the original base config list, or remove the trial directory."
        )

    changed_bases = [
        base for base, checksum in new_base_checksums.items() if expected_bases[base] != checksum
    ]
    resolved_drift = expected_resolved != new_resolved_checksum

    if not (changed_bases or resolved_drift):
        return

    detail = f"changed base files: {changed_bases}" if changed_bases else "the resolved config changed"
    raise SweepDriftError(
        f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
        f"{detail}. Drop --resume to start fresh, revert the change, or remove the trial directory."
    )


def materialize_trial(
    config: SweepConfig,
    trial: Trial,
    resume: bool = False,
    expected_checksums: dict[str, Any] | None = None,
) -> TrialArtifacts:
    trial_dir = config.output_dir / "trials" / trial.id
    run_dir = trial_dir / "run"
    overrides_path = trial_dir / "overrides.toml"
    resolved_path = trial_dir / "resolved.toml"
    command_path = trial_dir / "command.txt"
    status_path = trial_dir / "status.json"

    flat_overrides = dict(trial.parameters)
    flat_overrides["output_dir"] = run_dir.as_posix()
    _merge_wandb_overrides(config, flat_overrides, trial)

    overrides = build_nested_overrides(flat_overrides)

    preserved_status = _existing_terminal_status(status_path) if resume else None

    resolved_config = _validate_target_with_overrides(
        config.entrypoint,
        config.base,
        overrides_path,
        overrides,
        avoid_overwriting=preserved_status is not None,
    )
    resolved_data = _target_config_to_toml(resolved_config)
    _reject_bool_target_coercions(trial.parameters, resolved_data)

    command = command_for_trial(config.entrypoint, config.base, overrides_path)

    resolved_checksum = toml_checksum(resolved_data)
    base_checksums = {base.as_posix(): file_checksum(base) for base in config.base}

    if preserved_status is not None:
        if preserved_status.get("id") != trial.id:
            raise SweepDriftError(
                f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
                f"status.json belongs to {preserved_status.get('id')!r}. Restore the trial artifacts "
                "or remove the mismatched trial directory."
            )
        _check_resume_drift(trial, preserved_status, expected_checksums, resolved_checksum, base_checksums)
    write_pending_status = preserved_status is None

    write_toml(overrides_path, overrides)
    write_toml(resolved_path, resolved_data)
    command_path.write_text(shlex.join(command) + "\n")

    if write_pending_status:
        write_json(
            status_path,
            {
                "id": trial.id,
                "label": trial.label,
                "state": "pending",
                "pid": None,
                "slurm_job_id": None,
                "gpu_group": None,
                "returncode": None,
                "objective": None,
            },
        )

    return TrialArtifacts(
        trial=trial,
        trial_dir=trial_dir,
        run_dir=run_dir,
        overrides_path=overrides_path,
        resolved_path=resolved_path,
        command_path=command_path,
        status_path=status_path,
        command=command,
        resolved_checksum=resolved_checksum,
        base_checksums=base_checksums,
    )


def record_trial_materialization_failure(
    config: SweepConfig,
    trial: Trial,
    exc: BaseException,
    *,
    finished_at: str | None = None,
) -> TrialArtifacts:
    """Write a manifestable artifact for an Optuna trial that failed config validation."""
    trial_dir = config.output_dir / "trials" / trial.id
    run_dir = trial_dir / "run"
    overrides_path = trial_dir / "overrides.toml"
    resolved_path = trial_dir / "resolved.toml"
    command_path = trial_dir / "command.txt"
    status_path = trial_dir / "status.json"

    flat_overrides = dict(trial.parameters)
    flat_overrides["output_dir"] = run_dir.as_posix()
    _merge_wandb_overrides(config, flat_overrides, trial)
    write_toml(overrides_path, build_nested_overrides(flat_overrides))

    command = command_for_trial(config.entrypoint, config.base, overrides_path)
    command_path.write_text(shlex.join(command) + "\n")
    resolved_path.unlink(missing_ok=True)

    status = {
        "id": trial.id,
        "label": trial.label,
        "state": "failed",
        "pid": None,
        "slurm_job_id": None,
        "gpu_group": None,
        "returncode": -1,
        "objective": None,
        "failure_stage": "materialization",
        "error": _materialization_error(exc),
    }
    if finished_at is not None:
        status["finished_at"] = finished_at
    write_json(status_path, status)

    return TrialArtifacts(
        trial=trial,
        trial_dir=trial_dir,
        run_dir=run_dir,
        overrides_path=overrides_path,
        resolved_path=resolved_path,
        command_path=command_path,
        status_path=status_path,
        command=command,
        resolved_checksum="",
        base_checksums={base.as_posix(): file_checksum(base) for base in config.base},
    )


def _merge_multi_run_wandb_overrides(
    config: SweepConfig, flat_overrides: dict[str, Any], trial: Trial
) -> None:
    """Tag the per-run orchestrator's W&B run with sweep + trial metadata."""
    if config.wandb is None or not config.wandb.enabled:
        return

    group = config.wandb.group or config.name
    if group is not None:
        flat_overrides["orchestrator.wandb.group"] = group

    flat_overrides["orchestrator.wandb.name"] = trial.label or trial.id

    tags = list(dict.fromkeys([*config.wandb.tags, "sweep", f"trial:{trial.id}"]))
    if config.name is not None:
        tags.append(f"study:{config.name}")
    flat_overrides["orchestrator.wandb.tags"] = list(dict.fromkeys(tags))


def multi_run_shared_dir(config: SweepConfig) -> Path:
    """Directory that hosts the shared trainer's output and per-run subdirs.

    The trainer's ``MultiRunManager`` scans ``<dir>/run_*`` so every trial
    directory must sit directly under this path with a ``run_`` prefix.
    """
    return config.output_dir / "shared"


def multi_run_trial_dir(config: SweepConfig, trial: Trial) -> Path:
    """Per-trial directory the trainer will discover as a ``run_*`` slot."""
    return multi_run_shared_dir(config) / f"run_{trial.id}"


def write_multi_run_output_override(shared_dir: Path) -> Path:
    """Write the trainer ``output_dir`` pin used by ``rl-multi-run`` launches.

    Pinning ``output_dir = <shared_dir>`` keeps the trainer's
    ``MultiRunManager`` scanning the right ``run_*`` slots regardless of what
    the base TOML carries. Materialization and the launcher both call this so
    the override file is a known, replayable artifact rather than an
    implementation detail of ``build_multi_run_command``.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    path = shared_dir / "_output_override.toml"
    write_toml(path, {"output_dir": shared_dir.as_posix()})
    return path


def _finalize_multi_run_lora_config(
    orchestrator_dict: dict[str, Any],
    resolved_rl_config: RLConfig,
    trial_parameters: dict[str, Any],
) -> None:
    """Apply multi-run LoRA relaxations after shared RLConfig validation."""
    trainer = getattr(resolved_rl_config, "trainer", None)
    trainer_model = getattr(trainer, "model", None)
    trainer_lora = getattr(trainer_model, "lora", None)
    model = orchestrator_dict.get("model")
    if trainer_lora is None or not isinstance(model, dict):
        return
    lora = model.get("lora")
    if not isinstance(lora, dict):
        return

    if lora.get("rank") is None:
        lora["rank"] = trainer_lora.rank
    if lora.get("alpha") is None:
        lora["alpha"] = trainer_lora.alpha
    rank = lora["rank"]
    if isinstance(rank, int) and not isinstance(rank, bool) and rank > trainer_lora.rank:
        raise ValueError(
            f"orchestrator.model.lora.rank ({rank}) exceeds "
            f"trainer.model.lora.rank ({trainer_lora.rank})"
        )

    lora_shape_changed = (
        "orchestrator.model.lora.rank" in trial_parameters
        or "orchestrator.model.lora.alpha" in trial_parameters
    )
    if lora_shape_changed and "orchestrator.model.lora.name" not in trial_parameters:
        lora["name"] = f"r{lora['rank']}-a{lora['alpha']}"


def _validate_multi_run_shared_config(resolved_rl_config: RLConfig, scheduler: Any) -> None:
    if not isinstance(resolved_rl_config, RLConfig):
        return

    trainer_lora = resolved_rl_config.trainer.model.lora
    if trainer_lora is None:
        raise ValueError("multi_run_lora requires trainer.model.lora in the shared RLConfig.")

    trainer_max_runs = resolved_rl_config.trainer.max_concurrent_runs
    scheduler_max_runs = scheduler.max_concurrent_runs
    if trainer_max_runs < scheduler_max_runs:
        raise ValueError(
            "multi_run_lora scheduler.max_concurrent_runs must be <= "
            f"trainer.max_concurrent_runs in the shared RLConfig "
            f"(got scheduler={scheduler_max_runs}, trainer={trainer_max_runs})."
        )


def _finalize_multi_run_batching_config(
    orchestrator_dict: dict[str, Any],
    trial_parameters: dict[str, Any],
    explicit_shared_orchestrator_fields: set[str],
) -> None:
    sets_batch_size = "orchestrator.batch_size" in trial_parameters
    sets_token_batch_size = "orchestrator.token_batch_size" in trial_parameters
    sets_oversampling_factor = "orchestrator.oversampling_factor" in trial_parameters
    if sets_batch_size and sets_token_batch_size:
        raise ValueError("Set either orchestrator.batch_size or orchestrator.token_batch_size, not both.")
    if sets_batch_size or sets_oversampling_factor:
        orchestrator_dict.pop("token_batch_size", None)
    if sets_token_batch_size:
        orchestrator_dict.pop("batch_size", None)
        orchestrator_dict.pop("oversampling_factor", None)
    if (
        (sets_batch_size or sets_token_batch_size or sets_oversampling_factor)
        and "orchestrator.max_inflight_rollouts" not in trial_parameters
        and "max_inflight_rollouts" not in explicit_shared_orchestrator_fields
    ):
        orchestrator_dict.pop("max_inflight_rollouts", None)


def _canonicalize_multi_run_sampling_aliases(
    orchestrator_dict: dict[str, Any],
    trial_parameters: dict[str, Any],
) -> None:
    for section in ("train", "eval"):
        path = f"orchestrator.{section}.sampling.max_tokens"
        if path not in trial_parameters:
            continue
        section_config = orchestrator_dict.get(section)
        if not isinstance(section_config, dict):
            continue
        sampling = section_config.get("sampling")
        if not isinstance(sampling, dict):
            continue
        if "max_tokens" in sampling:
            sampling["max_completion_tokens"] = sampling.pop("max_tokens")


def _shared_section(shared_orchestrator: dict[str, Any], section: str) -> dict[str, Any]:
    section_data: dict[str, Any] = {}
    if section == "train":
        if "env" in shared_orchestrator:
            section_data["env"] = shared_orchestrator["env"]
        if "sampling" in shared_orchestrator:
            section_data["sampling"] = shared_orchestrator["sampling"]
    raw = shared_orchestrator.get(section)
    if isinstance(raw, dict):
        merge_nested_overrides(section_data, raw)
    return section_data


def _raw_envs(shared_section: dict[str, Any]) -> list[Any]:
    envs = shared_section.get("env", [])
    return envs if isinstance(envs, list) else []


def _raw_env(raw_envs: list[Any], idx: int) -> dict[str, Any]:
    if idx >= len(raw_envs):
        return {}
    raw = raw_envs[idx]
    return raw if isinstance(raw, dict) else {}


def _clear_inherited_env_default(
    env: dict[str, Any],
    raw_env: dict[str, Any],
    field: str,
) -> None:
    if field not in raw_env:
        env.pop(field, None)


def _clear_inherited_sampling_default(
    env: dict[str, Any],
    raw_env: dict[str, Any],
    field: str,
) -> None:
    raw_sampling = raw_env.get("sampling")
    raw_sampling = raw_sampling if isinstance(raw_sampling, dict) else {}
    aliases = (field, "max_tokens") if field == "max_completion_tokens" else (field,)
    if any(alias in raw_sampling for alias in aliases):
        return
    sampling = env.get("sampling")
    if isinstance(sampling, dict):
        sampling.pop(field, None)


def _clear_inherited_extra_body_default(
    env: dict[str, Any],
    raw_env: dict[str, Any],
    key: str,
    group_extra_body: dict[str, Any],
) -> None:
    raw_sampling = raw_env.get("sampling")
    raw_sampling = raw_sampling if isinstance(raw_sampling, dict) else {}
    raw_extra_body = raw_sampling.get("extra_body")
    raw_extra_body = raw_extra_body if isinstance(raw_extra_body, dict) else {}
    if key in raw_extra_body:
        return
    sampling = env.get("sampling")
    sampling = sampling if isinstance(sampling, dict) else {}
    extra_body = sampling.get("extra_body")
    if isinstance(extra_body, dict):
        if key in group_extra_body:
            extra_body[key] = group_extra_body[key]
        else:
            extra_body.pop(key, None)


def _swept_sampling_defaults(
    trial_parameters: dict[str, Any],
    section: str,
) -> tuple[set[str], bool, set[str]]:
    prefix = f"orchestrator.{section}.sampling."
    fields: set[str] = set()
    replaces_extra_body = False
    extra_body_keys: set[str] = set()
    for path, value in trial_parameters.items():
        if path == prefix.removesuffix("."):
            if isinstance(value, dict):
                for key in value:
                    if key == "max_tokens":
                        fields.add("max_completion_tokens")
                    elif key == "extra_body":
                        replaces_extra_body = True
                    else:
                        fields.add(key)
            continue
        if not path.startswith(prefix):
            continue
        suffix = path.removeprefix(prefix)
        if suffix == "max_tokens":
            fields.add("max_completion_tokens")
        elif suffix == "extra_body":
            replaces_extra_body = True
        elif suffix.startswith("extra_body."):
            extra_body_keys.add(suffix.removeprefix("extra_body.").split(".", 1)[0])
        else:
            fields.add(suffix.split(".", 1)[0])
    return fields, replaces_extra_body, extra_body_keys


def _clear_inherited_env_defaults_for_section(
    orchestrator_dict: dict[str, Any],
    shared_orchestrator: dict[str, Any],
    trial_parameters: dict[str, Any],
    section: str,
) -> None:
    section_config = orchestrator_dict.get(section)
    if not isinstance(section_config, dict):
        return
    envs = section_config.get("env")
    if not isinstance(envs, list):
        return

    raw_section = _shared_section(shared_orchestrator, section)
    raw_envs = _raw_envs(raw_section)

    env_default_fields: set[str] = set()
    if f"orchestrator.{section}.num_workers" in trial_parameters:
        env_default_fields.add("num_workers")
    if f"orchestrator.{section}.max_retries" in trial_parameters:
        env_default_fields.add("max_retries")
    if section == "train":
        if any(
            path in trial_parameters
            for path in (
                "orchestrator.batch_size",
                "orchestrator.oversampling_factor",
                "orchestrator.max_inflight_rollouts",
            )
        ):
            env_default_fields.add("num_workers")
    else:
        for field in ("num_examples", "rollouts_per_example", "interval"):
            if f"orchestrator.{section}.{field}" in trial_parameters:
                env_default_fields.add(field)
        if any(
            path in trial_parameters
            for path in (
                "orchestrator.eval.num_examples",
                "orchestrator.eval.rollouts_per_example",
            )
        ):
            env_default_fields.add("num_workers")

    sampling_fields, replaces_extra_body, extra_body_keys = _swept_sampling_defaults(
        trial_parameters, section
    )
    group_sampling = section_config.get("sampling")
    group_sampling = group_sampling if isinstance(group_sampling, dict) else {}
    group_extra_body = group_sampling.get("extra_body")
    group_extra_body = group_extra_body if isinstance(group_extra_body, dict) else {}

    for idx, env in enumerate(envs):
        if not isinstance(env, dict):
            continue
        raw_env = _raw_env(raw_envs, idx)
        for field in env_default_fields:
            _clear_inherited_env_default(env, raw_env, field)
        sampling = env.get("sampling")
        if not isinstance(sampling, dict):
            continue
        for field in sampling_fields:
            _clear_inherited_sampling_default(env, raw_env, field)
        if replaces_extra_body:
            _clear_inherited_sampling_default(env, raw_env, "extra_body")
        for key in extra_body_keys:
            _clear_inherited_extra_body_default(env, raw_env, key, group_extra_body)


def _clear_inherited_env_defaults(
    orchestrator_dict: dict[str, Any],
    shared_orchestrator: dict[str, Any],
    trial_parameters: dict[str, Any],
) -> None:
    _clear_inherited_env_defaults_for_section(
        orchestrator_dict, shared_orchestrator, trial_parameters, "train"
    )
    _clear_inherited_env_defaults_for_section(
        orchestrator_dict, shared_orchestrator, trial_parameters, "eval"
    )


def materialize_multi_run_trial(
    config: SweepConfig,
    trial: Trial,
    scheduler: Any,  # MultiRunLoRASchedulerConfig — typed as Any to avoid an import cycle
) -> TrialArtifacts:
    """Write a per-trial ``run_<id>/control/orch.toml`` for a shared-trainer sweep.

    The shared base TOMLs in ``scheduler.shared`` resolve to a full RLConfig.
    Per-trial parameter overrides (already prefixed with ``orchestrator.``)
    are layered on top, the orchestrator block is extracted, and its TOML is
    written where the trainer's ``MultiRunManager`` will find it. Returned
    ``TrialArtifacts.run_dir`` points at the per-trial directory so the
    sweep's existing metrics readers (``read_final_summary`` /
    ``read_intermediate_metric``) keep working unchanged once the
    orchestrator's ``FileMonitor`` writes ``metrics.jsonl`` there.
    """
    run_dir = multi_run_trial_dir(config, trial)
    control_dir = run_dir / "control"
    overrides_path = run_dir / "overrides.toml"
    resolved_path = run_dir / "resolved.toml"
    command_path = run_dir / "command.txt"
    status_path = run_dir / "status.json"
    orch_config_path = control_dir / "orch.toml"

    # Trial overrides already use orchestrator.* paths thanks to the
    # validator's allowlist. Add the per-run output_dir + W&B identity.
    flat_overrides: dict[str, Any] = dict(trial.parameters)
    flat_overrides["orchestrator.output_dir"] = run_dir.as_posix()
    _merge_multi_run_wandb_overrides(config, flat_overrides, trial)

    overrides = build_nested_overrides(flat_overrides)
    write_toml(overrides_path, overrides)

    # Resolve shared RLConfig without per-run orchestrator overrides. The
    # single-run RLConfig cross-checks require orchestrator LoRA rank/alpha to
    # equal trainer rank/alpha, but multi-run permits per-run rank <= trainer
    # rank. Trial overrides are validated against OrchestratorConfig below.
    args: list[str] = []
    for base_path in scheduler.shared:
        args.extend(["@", base_path.as_posix()])

    resolved_rl_config = validate_target_config("rl", args)
    _validate_multi_run_shared_config(resolved_rl_config, scheduler)
    orchestrator_dict = resolved_rl_config.orchestrator.model_dump(exclude_none=True, mode="json")
    shared_overrides = _load_nested_toml(scheduler.shared)
    shared_orchestrator = shared_overrides.get("orchestrator", {})
    explicit_shared_orchestrator_fields = set(shared_orchestrator) if isinstance(shared_orchestrator, dict) else set()
    merge_nested_overrides(orchestrator_dict, overrides.get("orchestrator", {}))
    _canonicalize_multi_run_sampling_aliases(orchestrator_dict, trial.parameters)
    _finalize_multi_run_batching_config(
        orchestrator_dict,
        trial.parameters,
        explicit_shared_orchestrator_fields,
    )
    _clear_inherited_env_defaults(
        orchestrator_dict,
        shared_orchestrator if isinstance(shared_orchestrator, dict) else {},
        trial.parameters,
    )
    _finalize_multi_run_lora_config(orchestrator_dict, resolved_rl_config, trial.parameters)

    # RLConfig.auto_setup_output_dir resets orchestrator.output_dir to
    # ``<top-level output_dir>/run_default`` during validation, which would
    # collapse every trial onto the same orchestrator directory. Restore the
    # per-trial path so each orch.toml targets its own run_<id> slot.
    orchestrator_dict["output_dir"] = run_dir.as_posix()
    orchestrator_config = OrchestratorConfig(**orchestrator_dict)
    orchestrator_config.output_dir = run_dir
    orchestrator_dict = orchestrator_config.model_dump(exclude_none=True, mode="json")
    _reject_bool_target_coercions(
        trial.parameters,
        orchestrator_dict,
        path_prefix="orchestrator.",
    )

    write_toml(resolved_path, orchestrator_dict)
    write_toml(orch_config_path, orchestrator_dict)

    # Mirror what build_multi_run_command issues so this command.txt is
    # actually replayable: rl-multi-run requires --runs-dir with
    # colon-separated paths, plus the trainer output_dir pin.
    output_override_path = write_multi_run_output_override(multi_run_shared_dir(config))
    command = [
        "rl-multi-run",
        *sum((["@", p.as_posix()] for p in scheduler.shared), []),
        "@",
        output_override_path.as_posix(),
        "--runs-dir",
        run_dir.as_posix(),
    ]
    command_path.write_text(shlex.join(command) + "\n")

    resolved_checksum = file_checksum(resolved_path)
    base_checksums = {base.as_posix(): file_checksum(base) for base in scheduler.shared}

    write_json(
        status_path,
        {
            "id": trial.id,
            "label": trial.label,
            "state": "pending",
            "pid": None,
            "slurm_job_id": None,
            "gpu_group": None,
            "returncode": None,
            "objective": None,
        },
    )

    return TrialArtifacts(
        trial=trial,
        trial_dir=run_dir,
        run_dir=run_dir,
        overrides_path=overrides_path,
        resolved_path=resolved_path,
        command_path=command_path,
        status_path=status_path,
        command=command,
        resolved_checksum=resolved_checksum,
        base_checksums=base_checksums,
    )


def record_multi_run_materialization_failure(
    config: SweepConfig,
    trial: Trial,
    scheduler: Any,
    exc: BaseException,
    *,
    finished_at: str | None = None,
) -> TrialArtifacts:
    """Write a manifestable artifact for a failed per-run orchestrator config."""
    run_dir = multi_run_trial_dir(config, trial)
    control_dir = run_dir / "control"
    overrides_path = run_dir / "overrides.toml"
    resolved_path = run_dir / "resolved.toml"
    command_path = run_dir / "command.txt"
    status_path = run_dir / "status.json"
    orch_config_path = control_dir / "orch.toml"

    flat_overrides: dict[str, Any] = dict(trial.parameters)
    flat_overrides["orchestrator.output_dir"] = run_dir.as_posix()
    _merge_multi_run_wandb_overrides(config, flat_overrides, trial)
    write_toml(overrides_path, build_nested_overrides(flat_overrides))

    output_override_path = write_multi_run_output_override(multi_run_shared_dir(config))
    command = [
        "rl-multi-run",
        *sum((["@", p.as_posix()] for p in scheduler.shared), []),
        "@",
        output_override_path.as_posix(),
        "--runs-dir",
        run_dir.as_posix(),
    ]
    command_path.write_text(shlex.join(command) + "\n")
    resolved_path.unlink(missing_ok=True)
    orch_config_path.unlink(missing_ok=True)

    status = {
        "id": trial.id,
        "label": trial.label,
        "state": "failed",
        "pid": None,
        "slurm_job_id": None,
        "gpu_group": None,
        "returncode": -1,
        "objective": None,
        "failure_stage": "materialization",
        "error": _materialization_error(exc),
    }
    if finished_at is not None:
        status["finished_at"] = finished_at
    write_json(status_path, status)

    control_dir.mkdir(parents=True, exist_ok=True)
    return TrialArtifacts(
        trial=trial,
        trial_dir=run_dir,
        run_dir=run_dir,
        overrides_path=overrides_path,
        resolved_path=resolved_path,
        command_path=command_path,
        status_path=status_path,
        command=command,
        resolved_checksum="",
        base_checksums={base.as_posix(): file_checksum(base) for base in scheduler.shared},
    )
