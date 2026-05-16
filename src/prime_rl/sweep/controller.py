import json
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tomli_w

from prime_rl.configs.sweep import (
    GridStrategyConfig,
    LocalSweepSchedulerConfig,
    MultiRunLoRASchedulerConfig,
    OptunaStrategyConfig,
    RandomStrategyConfig,
    SlurmSweepSchedulerConfig,
    SweepConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    SweepDriftError,
    SweepStatusError,
    Trial,
    TrialArtifacts,
    materialize_multi_run_trial,
    materialize_trial,
    multi_run_shared_dir,
    read_status_json,
    record_multi_run_materialization_failure,
    record_trial_materialization_failure,
    record_trial_missing_objective,
    record_trial_objective,
)
from prime_rl.sweep.metrics import coerce_finite_float, read_final_summary
from prime_rl.sweep.multi_run import run_multi_run_optuna_sweep
from prime_rl.sweep.optuna_loop import run_optuna_sweep, run_optuna_sweep_parallel_slurm
from prime_rl.sweep.reproducibility import git_metadata
from prime_rl.sweep.schedulers import (
    run_trials_locally,
    submit_trials_to_multi_run_lora,
    submit_trials_to_slurm,
    utc_now,
)
from prime_rl.sweep.search import expand_grid, sample_random


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def build_variant(artifact: TrialArtifacts) -> dict[str, Any]:
    # Mirror live trial state into the manifest so jq queries against
    # variants[*].state / variants[*].objective work without separately
    # reading each status.json. status.json is written at materialization
    # (state="pending") and updated as trials run, so we always have at
    # least the pending values to record here.
    try:
        status = read_status_json(artifact.status_path) if artifact.status_path.exists() else {}
    except SweepStatusError:
        status = {}
    return {
        "id": artifact.trial.id,
        "label": artifact.trial.label,
        "output_dir": artifact.run_dir.as_posix(),
        "overrides": artifact.trial.parameters,
        "command": artifact.command,
        "status_path": artifact.status_path.as_posix(),
        "resolved_checksum": artifact.resolved_checksum,
        "base_checksums": artifact.base_checksums,
        "state": status.get("state"),
        "objective": status.get("objective"),
    }


def write_manifest_with_variants(config: SweepConfig, variants: list[dict[str, Any]]) -> None:
    manifest = {
        "name": config.name,
        "entrypoint": config.entrypoint,
        "strategy": config.strategy.model_dump(mode="json"),
        "scheduler": config.scheduler.model_dump(mode="json"),
        "parameters": _manifest_parameters(config),
        "parameter_order": _manifest_parameter_order(config),
        "objective": config.objective.model_dump(mode="json") if config.objective else None,
        "early_stopping": config.early_stopping.model_dump(mode="json") if config.early_stopping else None,
        "git": git_metadata(),
        "variants": variants,
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_manifest(config: SweepConfig, artifacts: list[TrialArtifacts]) -> None:
    write_manifest_with_variants(config, [build_variant(a) for a in artifacts])


def _update_manifest_summary(config: SweepConfig, summary: dict[str, Any] | None) -> None:
    manifest_path = config.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _manifest_variant_count(config: SweepConfig, fallback: int) -> int:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.is_file():
        return fallback
    variants = json.loads(manifest_path.read_text()).get("variants")
    return len(variants) if isinstance(variants, list) else fallback


def _expand_trials(config: SweepConfig) -> list[Trial]:
    if isinstance(config.strategy, GridStrategyConfig):
        return expand_grid(config.parameters)
    if isinstance(config.strategy, RandomStrategyConfig):
        return sample_random(
            config.parameters,
            num_trials=config.strategy.num_trials,
            seed=config.strategy.seed,
        )
    raise ValueError(f"Unsupported sweep strategy: {config.strategy!r}")


def _previous_manifest(config: SweepConfig) -> dict[str, Any] | None:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the previous manifest "
            "is not valid JSON. Restore the manifest or start a fresh study/output_dir."
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the previous manifest "
            "is not a JSON object. Restore the manifest or start a fresh study/output_dir."
        )
    return manifest


def _manifest_objective(config: SweepConfig) -> dict[str, Any] | None:
    return config.objective.model_dump(mode="json") if config.objective else None


def _manifest_parameters(config: SweepConfig) -> dict[str, Any]:
    return {path: parameter.model_dump(mode="json") for path, parameter in config.parameters.items()}


def _manifest_parameter_order(config: SweepConfig) -> list[str]:
    return list(config.parameters)


def _validate_resume_manifest_objective(config: SweepConfig, manifest: dict[str, Any] | None) -> None:
    if manifest is None or not manifest.get("variants"):
        return

    previous_objective = manifest.get("objective")
    current_objective = _manifest_objective(config)
    if previous_objective == current_objective:
        return

    raise RuntimeError(
        "Resume cannot reuse existing trial objectives because the sweep objective changed "
        f"(previous={previous_objective}, current={current_objective}). "
        "Use the original objective or start a fresh study/output_dir."
    )


def _validate_resume_manifest_entrypoint(config: SweepConfig, manifest: dict[str, Any] | None) -> None:
    if manifest is None or not manifest.get("variants"):
        return

    previous_entrypoint = manifest.get("entrypoint")
    if previous_entrypoint == config.entrypoint:
        return

    raise RuntimeError(
        "Resume cannot reuse existing trial results because the sweep entrypoint changed "
        f"(previous={previous_entrypoint}, current={config.entrypoint}). "
        "Use the original entrypoint or start a fresh study/output_dir."
    )


def _validate_resume_manifest_parameters(config: SweepConfig, manifest: dict[str, Any] | None) -> None:
    if manifest is None or not manifest.get("variants"):
        return

    previous_parameters = manifest.get("parameters")
    current_parameters = _manifest_parameters(config)
    if previous_parameters != current_parameters:
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the sweep parameters changed "
            f"(previous={previous_parameters}, current={current_parameters}). "
            "Use the original search space or start a fresh study/output_dir."
        )

    previous_order = manifest.get("parameter_order")
    current_order = _manifest_parameter_order(config)
    if previous_order == current_order:
        return

    raise RuntimeError(
        "Resume cannot reuse existing trial results because the sweep parameter order changed "
        f"(previous={previous_order}, current={current_order}). Trial generation is order-sensitive; "
        "use the original parameter order or start a fresh study/output_dir."
    )


def _validate_resume_manifest_strategy(config: SweepConfig, manifest: dict[str, Any] | None) -> None:
    if manifest is None or not manifest.get("variants"):
        return

    previous_strategy = manifest.get("strategy")
    current_strategy = config.strategy.model_dump(mode="json")
    if not isinstance(previous_strategy, dict):
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the previous manifest "
            "does not record the sweep strategy. Restore the manifest or start a fresh study/output_dir."
        )

    if previous_strategy.get("type") != current_strategy.get("type"):
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the sweep strategy changed "
            f"(previous={previous_strategy}, current={current_strategy}). "
            "Use the original strategy or start a fresh study/output_dir."
        )

    if isinstance(config.strategy, RandomStrategyConfig | OptunaStrategyConfig):
        previous_without_count = dict(previous_strategy)
        current_without_count = dict(current_strategy)
        previous_num_trials = previous_without_count.pop("num_trials", None)
        current_num_trials = current_without_count.pop("num_trials", None)

        compatible = (
            previous_without_count == current_without_count
            and type(previous_num_trials) is int
            and type(current_num_trials) is int
            and current_num_trials >= previous_num_trials
        )
        if compatible:
            return
    elif previous_strategy == current_strategy:
        return

    raise RuntimeError(
        "Resume cannot reuse existing trial results because the sweep strategy changed "
        f"(previous={previous_strategy}, current={current_strategy}). "
        "Only increasing strategy.num_trials is supported on resume; other strategy changes "
        "need a fresh study/output_dir."
    )


def _validate_resume_manifest_scheduler(config: SweepConfig, manifest: dict[str, Any] | None) -> None:
    if manifest is None or not manifest.get("variants"):
        return

    previous_scheduler = manifest.get("scheduler")
    current_scheduler = config.scheduler.model_dump(mode="json")
    if not isinstance(previous_scheduler, dict):
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the previous manifest "
            "does not record the sweep scheduler. Restore the manifest or start a fresh study/output_dir."
        )

    if previous_scheduler.get("type") == current_scheduler.get("type"):
        return

    raise RuntimeError(
        "Resume cannot reuse existing trial results because the sweep scheduler type changed "
        f"(previous={previous_scheduler}, current={current_scheduler}). "
        "Use the original scheduler type or start a fresh study/output_dir."
    )


def _checksums_from_manifest(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map trial_id -> {resolved_checksum, base_checksums} from the prior manifest."""
    if manifest is None:
        return {}

    variants = manifest.get("variants", [])
    if not isinstance(variants, list):
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the previous manifest "
            "does not record variants as a list. Restore the manifest or start a fresh study/output_dir."
        )

    checksums: dict[str, dict[str, Any]] = {}
    invalid_ids: list[Any] = []
    duplicate_ids: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            invalid_ids.append(variant)
            continue
        trial_id = variant.get("id")
        if not isinstance(trial_id, str) or not trial_id:
            invalid_ids.append(trial_id)
            continue
        if trial_id in checksums:
            duplicate_ids.append(trial_id)
            continue
        checksums[trial_id] = {
            "resolved_checksum": variant.get("resolved_checksum"),
            "base_checksums": variant.get("base_checksums") or {},
        }

    if invalid_ids or duplicate_ids:
        raise RuntimeError(
            "Resume cannot reuse existing trial results because the previous manifest "
            "has malformed or duplicate variant id(s). "
            f"Invalid id(s): {invalid_ids}; duplicate id(s): {duplicate_ids}. "
            "Repair the manifest or start a fresh study/output_dir."
        )

    return checksums


def _materialize_study(config: SweepConfig) -> list[TrialArtifacts]:
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest = _previous_manifest(config) if config.resume else None
    if config.resume:
        _validate_resume_manifest_entrypoint(config, previous_manifest)
        _validate_resume_manifest_objective(config, previous_manifest)
        _validate_resume_manifest_parameters(config, previous_manifest)
        _validate_resume_manifest_strategy(config, previous_manifest)
        _validate_resume_manifest_scheduler(config, previous_manifest)
    expected = _checksums_from_manifest(previous_manifest)

    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    trials = _expand_trials(config)
    if config.resume:
        trial_ids = {trial.id for trial in trials}
        extra_manifest_ids = sorted(set(expected) - trial_ids)
        if extra_manifest_ids:
            raise RuntimeError(
                "Resume cannot reuse existing trial results because the previous manifest "
                "has variant id(s) that are not in the regenerated trial set: "
                f"{extra_manifest_ids}. Restore the original sweep definition, repair the manifest, "
                "or start a fresh study/output_dir."
            )
    artifacts: list[TrialArtifacts] = []
    for trial in trials:
        try:
            artifact = materialize_trial(
                config,
                trial,
                resume=config.resume,
                expected_checksums=expected.get(trial.id),
            )
        except Exception as exc:
            if _should_propagate_materialization_error(exc):
                raise
            artifact = record_trial_materialization_failure(config, trial, exc, finished_at=utc_now())
            artifacts.append(artifact)
            if not config.continue_on_failure:
                break
            continue
        artifacts.append(artifact)
    _write_manifest(config, artifacts)
    return artifacts


def _materialize_multi_run_study(config: SweepConfig) -> list[TrialArtifacts]:
    """Materialize all trials as ``run_*`` subdirs under a shared trainer dir.

    Multi-run sweeps invoke ``rl-multi-run`` exactly once with all trials
    laid out up front. Resume against a still-running shared trainer is a
    Phase 7b concern, so this path always writes from scratch.
    """
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    multi_run_shared_dir(config).mkdir(parents=True, exist_ok=True)

    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    trials = _expand_trials(config)
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    if len(trials) > config.scheduler.max_concurrent_runs:
        raise SystemExit(
            f"multi_run_lora scheduler.max_concurrent_runs={config.scheduler.max_concurrent_runs} "
            f"but the search expanded to {len(trials)} trials. Increase max_concurrent_runs "
            "or shrink the search space; wave-based execution requires Optuna (Phase 7b)."
        )
    artifacts: list[TrialArtifacts] = []
    for trial in trials:
        try:
            artifact = materialize_multi_run_trial(config, trial, config.scheduler)
        except Exception as exc:
            artifact = record_multi_run_materialization_failure(
                config, trial, config.scheduler, exc, finished_at=utc_now()
            )
            artifacts.append(artifact)
            if not config.continue_on_failure:
                break
            continue
        artifacts.append(artifact)
    _write_manifest(config, artifacts)
    return artifacts


def _build_trial_callback(
    config: SweepConfig,
    tracker: TrialOutcomeTracker | None,
    *,
    halt_on_missing_objective: bool = False,
):
    if config.objective is None or tracker is None:
        return None

    metric = config.objective.metric

    def on_trial_complete(artifact: TrialArtifacts, returncode: int) -> bool:
        objective = read_final_summary(artifact.run_dir, metric) if returncode == 0 else None
        record_trial_objective(artifact.status_path, objective)
        missing_objective = returncode == 0 and objective is None
        if missing_objective:
            record_trial_missing_objective(artifact.status_path, metric)
        outcome = TrialOutcome(trial_id=artifact.trial.id, label=artifact.trial.label, objective=objective)
        return tracker.observe(outcome) or (halt_on_missing_objective and missing_objective)

    return on_trial_complete


def _count_objective_failures(artifacts: list[TrialArtifacts]) -> int:
    missing = 0
    for artifact in artifacts:
        status = read_status_json(artifact.status_path)
        if status.get("failure_stage") == "objective":
            missing += 1
        elif status.get("state") == "completed" and coerce_finite_float(status.get("objective")) is None:
            missing += 1
    return missing


def _is_materialization_failure(artifact: TrialArtifacts) -> bool:
    return read_status_json(artifact.status_path).get("failure_stage") == "materialization"


def _should_propagate_materialization_error(exc: Exception) -> bool:
    return isinstance(exc, (SweepDriftError, SweepStatusError))


def _seed_tracker_from_resume(tracker: TrialOutcomeTracker, artifacts: list[TrialArtifacts]) -> None:
    """Replay each completed trial's recorded objective into the tracker.

    Without this the resumed scheduler skips already-completed trials, so the
    tracker never sees them — the manifest summary would forget earlier work
    and patience/threshold decisions would not account for completed trials.
    """
    for artifact in artifacts:
        status = read_status_json(artifact.status_path)
        if status.get("state") != "completed":
            continue
        outcome = TrialOutcome(
            trial_id=artifact.trial.id,
            label=artifact.trial.label,
            objective=status.get("objective"),
        )
        tracker.observe(outcome)


def _run_optuna(config: SweepConfig) -> None:
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume:
        previous_manifest = _previous_manifest(config)
        _validate_resume_manifest_entrypoint(config, previous_manifest)
        _validate_resume_manifest_objective(config, previous_manifest)
        _validate_resume_manifest_parameters(config, previous_manifest)
        _validate_resume_manifest_strategy(config, previous_manifest)
        _validate_resume_manifest_scheduler(config, previous_manifest)
    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    if config.dry_run:
        print(
            "Dry run for Optuna strategy is a no-op: trials are proposed sequentially based on "
            "prior objectives, so they cannot be materialized up front."
        )
        return

    from prime_rl.configs.sweep import SlurmSweepSchedulerConfig as _SlurmCfg

    if (
        isinstance(config.scheduler, _SlurmCfg)
        and config.scheduler.synchronous
        and config.scheduler.max_parallel > 1
    ):
        failures, tracker, artifacts = run_optuna_sweep_parallel_slurm(
            config,
            write_manifest_with_variants=write_manifest_with_variants,
            build_variant=build_variant,
        )
    else:
        failures, tracker, artifacts = run_optuna_sweep(
            config,
            write_manifest_with_variants=write_manifest_with_variants,
            build_variant=build_variant,
        )

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        total = _manifest_variant_count(config, len(artifacts))
        print(f"Sweep finished with {failures} failed trial(s) out of {total}.")
        raise SystemExit(1)


def _run_multi_run_static(config: SweepConfig) -> None:
    """Drive a static (grid/random) shared-trainer LoRA sweep through ``rl-multi-run``."""
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    artifacts = _materialize_multi_run_study(config)
    materialization_failures = sum(1 for artifact in artifacts if _is_materialization_failure(artifact))
    launchable_artifacts = [
        artifact for artifact in artifacts if not _is_materialization_failure(artifact)
    ]

    if config.dry_run:
        print(
            f"Dry run complete. Materialized {len(artifacts)} run dir(s) under "
            f"{multi_run_shared_dir(config)}."
        )
        for artifact in artifacts:
            print(f"  {artifact.run_dir}")
        if materialization_failures:
            print(f"Dry run found {materialization_failures} failed trial materialization(s).")
            raise SystemExit(1)
        return

    if materialization_failures > 0 and not config.continue_on_failure:
        print("Skipping multi_run_lora launch: materialization failed and continue_on_failure=false.")
        failures = 0
    elif launchable_artifacts:
        failures = submit_trials_to_multi_run_lora(
            launchable_artifacts,
            shared_paths=config.scheduler.shared,
            shared_dir=multi_run_shared_dir(config),
            continue_on_failure=config.continue_on_failure,
            retry_budget=config.retry_budget,
        )
    else:
        failures = 0

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if config.objective else None
    missing_objective_failures = 0
    if config.objective is not None and tracker is not None:
        for artifact in artifacts:
            # reconcile_multi_run_artifact already chose between completed /
            # failed / pruned; only completed trials have an objective worth
            # rereading from metrics.jsonl, but record_trial_objective(None)
            # for the others keeps status.json's shape consistent.
            status = read_status_json(artifact.status_path)
            if status.get("state") == "completed":
                objective = read_final_summary(artifact.run_dir, config.objective.metric)
                if objective is None:
                    record_trial_missing_objective(artifact.status_path, config.objective.metric)
                    missing_objective_failures += 1
            else:
                objective = None
            if objective is not None or status.get("state") != "completed":
                record_trial_objective(artifact.status_path, objective)
            tracker.observe(
                TrialOutcome(trial_id=artifact.trial.id, label=artifact.trial.label, objective=objective)
            )
        summary = asdict(tracker.summary())

    # Same fix as run_sweep: refresh variants from each trial's final
    # status.json so state/objective reflect the post-wave reality, not
    # the pending values from materialization. Runs outside the
    # objective-tracker branch so state is also refreshed when no
    # objective is configured.
    _write_manifest(config, artifacts)
    if tracker is not None:
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")

    failures += materialization_failures
    failures += missing_objective_failures
    if failures > 0:
        total = _manifest_variant_count(config, len(artifacts))
        print(f"Sweep finished with {failures} failed trial(s) out of {total}.")
        raise SystemExit(1)


def _run_multi_run_optuna(config: SweepConfig) -> None:
    """Drive an Optuna study against ``rl-multi-run`` in waves of size ``max_concurrent_runs``."""
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    assert isinstance(config.strategy, OptunaStrategyConfig)
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    multi_run_shared_dir(config).mkdir(parents=True, exist_ok=True)
    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    if config.dry_run:
        print(
            "Dry run for Optuna + multi_run_lora is a no-op: trials are proposed wave by wave "
            "based on prior objectives, so they cannot be materialized up front."
        )
        return

    failures, tracker, artifacts = run_multi_run_optuna_sweep(
        config,
        write_manifest_with_variants=write_manifest_with_variants,
        build_variant=build_variant,
    )

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        total = _manifest_variant_count(config, len(artifacts))
        print(f"Sweep finished with {failures} failed trial(s) out of {total}.")
        raise SystemExit(1)


def _run_multi_run(config: SweepConfig) -> None:
    """Dispatch a shared-trainer LoRA sweep based on the search strategy."""
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    if isinstance(config.strategy, OptunaStrategyConfig):
        _run_multi_run_optuna(config)
        return
    _run_multi_run_static(config)


def run_sweep(config: SweepConfig) -> None:
    # multi_run_lora dispatches first because the Optuna + multi_run_lora
    # combination has its own wave driver — falling through to _run_optuna
    # would launch single-trial mode against the wrong scheduler.
    if isinstance(config.scheduler, MultiRunLoRASchedulerConfig):
        _run_multi_run(config)
        return

    if isinstance(config.strategy, OptunaStrategyConfig):
        _run_optuna(config)
        return

    artifacts = _materialize_study(config)
    materialization_failures = sum(1 for artifact in artifacts if _is_materialization_failure(artifact))
    launchable_artifacts = [
        artifact for artifact in artifacts if not _is_materialization_failure(artifact)
    ]

    if config.dry_run:
        print(f"Dry run complete. Materialized {len(artifacts)} trial(s) under {config.output_dir}.")
        for artifact in artifacts:
            print(shlex.join(artifact.command))
        if materialization_failures:
            print(f"Dry run found {materialization_failures} failed trial materialization(s).")
            raise SystemExit(1)
        return

    slurm_sync = (
        isinstance(config.scheduler, SlurmSweepSchedulerConfig) and config.scheduler.synchronous
    )
    track_objectives = config.objective is not None and (
        isinstance(config.scheduler, LocalSweepSchedulerConfig) or slurm_sync
    )
    if config.objective is not None and not track_objectives:
        print(
            "Note: objective tracking is only computed for the local scheduler or "
            "synchronous SLURM scheduler; async SLURM submission produces its own "
            "status.json without controller-side reconciliation."
        )
    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if track_objectives else None
    on_trial_complete = _build_trial_callback(
        config,
        tracker,
        halt_on_missing_objective=not config.continue_on_failure,
    )

    if tracker is not None and config.resume:
        _seed_tracker_from_resume(tracker, artifacts)

    failures = 0
    halt_after_materialization_failure = materialization_failures > 0 and not config.continue_on_failure
    counted_completed_missing_objectives = False
    resume_missing_objectives = (
        _count_objective_failures(artifacts)
        if track_objectives and config.resume
        else 0
    )
    if halt_after_materialization_failure:
        print("Skipping trial launch: materialization failed and continue_on_failure=false.")
    elif resume_missing_objectives > 0 and not config.continue_on_failure:
        failures += resume_missing_objectives
        counted_completed_missing_objectives = True
        print("Skipping new trials: resume found completed trial(s) without recorded objectives.")
    elif tracker is not None and tracker.halted:
        print("Skipping new trials: early stopping already triggered by completed trials in the study.")
    elif isinstance(config.scheduler, LocalSweepSchedulerConfig):
        gpu_groups = (
            config.scheduler.gpu_assignment.visible_devices if config.scheduler.gpu_assignment is not None else None
        )
        failures = run_trials_locally(
            launchable_artifacts,
            max_parallel=config.scheduler.max_parallel,
            gpu_groups=gpu_groups,
            continue_on_failure=config.continue_on_failure,
            retry_budget=config.retry_budget,
            on_trial_complete=on_trial_complete,
        )
    elif isinstance(config.scheduler, SlurmSweepSchedulerConfig):
        failures = submit_trials_to_slurm(
            launchable_artifacts,
            continue_on_failure=config.continue_on_failure,
            retry_budget=config.retry_budget,
            synchronous=config.scheduler.synchronous,
            on_trial_complete=on_trial_complete if config.scheduler.synchronous else None,
        )
    else:
        raise ValueError(f"Unsupported sweep scheduler: {config.scheduler}")

    if track_objectives and not counted_completed_missing_objectives:
        failures += _count_objective_failures(artifacts)
    failures += materialization_failures

    # Refresh manifest variants from each trial's final status.json so
    # state/objective reflect the post-run reality, not the pending
    # values written during materialization. Optuna and multi_run_lora
    # paths already rewrite the manifest at the end of their drivers.
    _write_manifest(config, artifacts)

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        print(f"Sweep finished with {failures} failed trial(s) out of {len(artifacts)}.")
        raise SystemExit(1)
