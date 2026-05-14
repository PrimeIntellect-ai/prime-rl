"""Optuna ask/tell driver for sweep trials.

Phase 5a supports TPE / Random samplers without pruning: trials run to
completion, the controller reads the final objective, and tells Optuna the
result before asking for the next parameter set.

Phase 5b adds pruning. When ``strategy.pruner`` is non-trivial the controller
spawns the trial as a child process group, polls ``metrics.jsonl`` while the
trial runs, calls ``optuna_trial.report(value, step)`` and
``optuna_trial.should_prune()`` between samples, and on a prune signal sends
SIGTERM (escalating to SIGKILL) to the trial's process group. Pruned trials
are recorded with ``state="pruned"`` in ``status.json`` and reported to Optuna
as ``TrialState.PRUNED`` so adaptive sampling can distinguish them from
completed runs and outright failures.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prime_rl.configs.sweep import (
    AshaPrunerConfig,
    ChoiceParameterConfig,
    HyperbandPrunerConfig,
    IntUniformParameterConfig,
    LocalSweepSchedulerConfig,
    LogUniformParameterConfig,
    MedianPrunerConfig,
    NoPrunerConfig,
    OptunaStrategyConfig,
    PrunerConfig,
    SlurmSweepSchedulerConfig,
    SweepConfig,
    SweepParameterConfig,
    UniformParameterConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    Trial,
    TrialArtifacts,
    materialize_trial,
    record_trial_materialization_failure,
    record_trial_missing_objective,
    record_trial_objective,
    record_trial_pruned,
    write_json,
)
from prime_rl.sweep.metrics import coerce_finite_float, read_final_summary, read_intermediate_metric
from prime_rl.sweep.reproducibility import file_checksum
from prime_rl.sweep.schedulers import (
    _build_env,
    _reset_metrics_jsonl,
    _run_trial_with_pruning_slurm_sync,
    _run_with_retries,
    _run_with_retries_slurm_sync,
    _write_launch_failure_status,
    _write_status,
    utc_now,
)
from prime_rl.sweep.search import parameters_hash, trial_label

if TYPE_CHECKING:  # pragma: no cover
    import optuna


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Optuna strategy requires the [hpo] extra. Install with: uv sync --extra hpo"
        ) from exc
    return optuna


def _suggest_parameters(
    optuna_trial: optuna.Trial,
    parameters: dict[str, SweepParameterConfig],
) -> dict[str, Any]:
    suggested: dict[str, Any] = {}
    for path, config in parameters.items():
        if isinstance(config, ChoiceParameterConfig):
            suggested[path] = optuna_trial.suggest_categorical(path, config.values)
        elif isinstance(config, UniformParameterConfig):
            suggested[path] = optuna_trial.suggest_float(path, config.min, config.max)
        elif isinstance(config, LogUniformParameterConfig):
            suggested[path] = optuna_trial.suggest_float(path, config.min, config.max, log=True)
        elif isinstance(config, IntUniformParameterConfig):
            suggested[path] = optuna_trial.suggest_int(path, config.min, config.max, step=config.step)
        else:
            raise ValueError(f"Unsupported parameter type for Optuna: {type(config)!r}")
    return suggested


def _build_sampler(
    optuna: Any,
    strategy: OptunaStrategyConfig,
    *,
    concurrent_trials: int = 1,
) -> Any:
    """Construct the Optuna sampler, opting into ``constant_liar`` for TPE
    when more than one trial will be in flight concurrently.

    TPE estimates its density from completed trials only. Without
    ``constant_liar``, concurrent asks see the same set of completed
    trials and can collide on the same region of the search space; with
    it, Optuna assigns a placeholder objective to running trials so the
    next ask is forced to diversify. ``RandomSampler`` doesn't need this.
    """
    if strategy.sampler == "tpe":
        kwargs: dict[str, Any] = {"seed": strategy.seed}
        if concurrent_trials > 1:
            kwargs["constant_liar"] = True
        return optuna.samplers.TPESampler(**kwargs)
    if strategy.sampler == "random":
        return optuna.samplers.RandomSampler(seed=strategy.seed)
    raise ValueError(f"Unsupported Optuna sampler: {strategy.sampler}")


def _build_pruner(optuna: Any, pruner: PrunerConfig) -> Any:
    """Map our discriminated PrunerConfig union to an Optuna pruner instance.

    ``NopPruner`` is Optuna's no-op default; using it explicitly keeps the
    study creation symmetric across pruner choices.
    """
    if isinstance(pruner, NoPrunerConfig):
        return optuna.pruners.NopPruner()
    if isinstance(pruner, MedianPrunerConfig):
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner.n_startup_trials,
            n_warmup_steps=pruner.n_warmup_steps,
            interval_steps=pruner.interval_steps,
        )
    if isinstance(pruner, AshaPrunerConfig):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pruner.min_resource,
            reduction_factor=pruner.reduction_factor,
            min_early_stopping_rate=pruner.min_early_stopping_rate,
        )
    if isinstance(pruner, HyperbandPrunerConfig):
        return optuna.pruners.HyperbandPruner(
            min_resource=pruner.min_resource,
            max_resource=pruner.max_resource,
            reduction_factor=pruner.reduction_factor,
        )
    raise ValueError(f"Unsupported Optuna pruner: {pruner!r}")


def _create_study(optuna: Any, config: SweepConfig) -> Any:
    """Create or reload the Optuna study.

    ``load_if_exists`` is gated on ``config.resume``: a fresh sweep must start
    from an empty optimization history, otherwise old trials would bias the
    sampler and the storage would silently accumulate trials across runs that
    the user thought were independent. With persistent storage and no
    ``resume`` flag, optuna raises ``DuplicatedStudyError`` to surface the
    collision instead of attaching silently.
    """
    strategy = config.strategy
    assert isinstance(strategy, OptunaStrategyConfig)
    assert config.objective is not None  # validated upstream
    direction = "maximize" if config.objective.direction == "maximize" else "minimize"
    concurrent_trials = (
        config.scheduler.max_parallel
        if isinstance(config.scheduler, SlurmSweepSchedulerConfig)
        else 1
    )
    return optuna.create_study(
        study_name=strategy.study_name or config.name or "sweep",
        storage=strategy.storage,
        sampler=_build_sampler(optuna, strategy, concurrent_trials=concurrent_trials),
        pruner=_build_pruner(optuna, strategy.pruner),
        direction=direction,
        load_if_exists=config.resume,
    )


def _make_trial(index: int, parameters: dict[str, Any]) -> Trial:
    trial_id = f"{index:04d}-{parameters_hash(parameters)}"
    label = trial_label(parameters) or trial_id
    return Trial(id=trial_id, label=label, parameters=parameters)


@dataclass
class _PollingOutcome:
    """Result of running a trial with intermediate-metric polling.

    ``unsafe_to_continue`` flags failures where the controller cannot
    confirm the underlying job has stopped — e.g. SLURM ``squeue`` is
    persistently unreachable, or a prune-triggered ``scancel`` did not
    confirm the job left the queue. The outer loop must halt the sweep
    in that case regardless of ``continue_on_failure``: launching the
    next Optuna trial would race the still-active allocation, which is
    worse than a noisy stop.
    """

    state: Literal["completed", "pruned", "failed"]
    returncode: int
    objective: float | None
    pruned_at_step: int | None = None
    pruned_value: float | None = None
    reports_sent: int = 0
    launch_error: bool = False
    launch_exception: OSError | None = None
    unsafe_to_continue: bool = False


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    """Stop the trial subprocess and any descendants it spawned.

    The sweep launches children with ``start_new_session=True`` so the trial
    and everything it spawns share a process group. We send SIGTERM to the
    whole group, wait briefly for graceful exit, then escalate to SIGKILL.
    Killing only the parent is not enough: the trainer/orchestrator/inference
    children would be reparented to init and keep running, holding GPUs and
    skewing the next trial's measurements.
    """
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _run_trial_with_pruning(
    artifact: TrialArtifacts,
    gpu_group: list[int] | None,
    optuna_trial: optuna.Trial,
    metric: str,
    poll_interval: float,
    attempt: int = 1,
) -> _PollingOutcome:
    """Spawn a trial and poll its metrics.jsonl for Optuna pruning decisions.

    Each new ``(step, value)`` pair the trial reports is forwarded to
    ``optuna_trial.report``. After every report we ask
    ``optuna_trial.should_prune()``; on True we terminate the trial's process
    group and return a ``pruned`` outcome. On natural exit we read the final
    objective from the same sidecar so the sampler sees the same value the
    rest of the sweep records.

    No retry loop here: the polling driver is meant to be the caller's
    single attempt, with retries handled by the outer loop only when the
    trial actually fails (returncode != 0 and no prune signal).
    """
    env = _build_env(artifact, gpu_group)
    _reset_metrics_jsonl(artifact)
    _write_status(
        artifact,
        state="running",
        started_at=utc_now(),
        attempts=attempt,
        gpu_group=list(gpu_group) if gpu_group is not None else None,
    )
    try:
        process = subprocess.Popen(artifact.command, env=env, start_new_session=True)
    except OSError as exc:
        return _PollingOutcome(
            state="failed",
            returncode=-1,
            objective=None,
            launch_error=True,
            launch_exception=exc,
        )
    last_reported_step: int | None = None
    reports_sent = 0

    try:
        while True:
            try:
                returncode = process.wait(timeout=poll_interval)
            except subprocess.TimeoutExpired:
                returncode = None

            sample = read_intermediate_metric(artifact.run_dir, metric)
            if sample is not None:
                step, value = sample
                if last_reported_step is None or step > last_reported_step:
                    optuna_trial.report(value, step)
                    last_reported_step = step
                    reports_sent += 1
                    # Only consider pruning while the trial is still running.
                    # If the subprocess already exited, the run produced its
                    # final objective and pruning would discard a valid value.
                    current_returncode = process.poll()
                    if current_returncode is not None:
                        returncode = current_returncode
                    elif optuna_trial.should_prune():
                        _terminate_process_group(process)
                        pruned_returncode = process.returncode if process.returncode is not None else -1
                        record_trial_pruned(
                            artifact.status_path,
                            step,
                            value,
                            returncode=pruned_returncode,
                            finished_at=utc_now(),
                        )
                        return _PollingOutcome(
                            state="pruned",
                            returncode=pruned_returncode,
                            objective=None,
                            pruned_at_step=step,
                            pruned_value=value,
                            reports_sent=reports_sent,
                        )

            if returncode is not None:
                break
    finally:
        # Belt-and-suspenders: if we exit through an unexpected path the
        # trial process must not be left running.
        _terminate_process_group(process)

    if returncode == 0:
        objective = read_final_summary(artifact.run_dir, metric)
        _write_status(artifact, state="completed", finished_at=utc_now(), returncode=0)
        return _PollingOutcome(
            state="completed", returncode=0, objective=objective, reports_sent=reports_sent
        )

    _write_status(artifact, state="failed", finished_at=utc_now(), returncode=returncode)
    return _PollingOutcome(
        state="failed", returncode=returncode, objective=None, reports_sent=reports_sent
    )


def _run_trial_with_pruning_slurm_sync_and_retries(
    artifact: TrialArtifacts,
    optuna_trial: optuna.Trial,
    metric: str,
    poll_interval: float,
    retry_budget: int,
) -> _PollingOutcome:
    """SLURM-sync analog of ``_run_trial_with_pruning_and_retries``.

    Same retry contract: ``completed`` and ``pruned`` return immediately, only
    ``failed`` outcomes are retried, and any sent reports disable retry to
    avoid biasing the pruner across attempts.
    """
    attempts = 0
    while True:
        attempts += 1
        outcome = _run_trial_with_pruning_slurm_sync(
            artifact,
            optuna_trial,
            metric,
            poll_interval,
            attempt=attempts,
        )
        if outcome.state in ("completed", "pruned"):
            return outcome
        if outcome.unsafe_to_continue:
            # The underlying SLURM job may still be alive — retrying would
            # try to submit a second job for the same trial. Bail out so
            # the outer loop can halt the sweep.
            return outcome
        if outcome.launch_error:
            if attempts > retry_budget:
                if outcome.launch_exception is not None:
                    _write_launch_failure_status(artifact, outcome.launch_exception)
                return outcome
            continue
        if outcome.reports_sent > 0:
            return outcome
        if attempts > retry_budget:
            return outcome


def _run_trial_with_pruning_and_retries(
    artifact: TrialArtifacts,
    gpu_group: list[int] | None,
    optuna_trial: optuna.Trial,
    metric: str,
    poll_interval: float,
    retry_budget: int,
) -> _PollingOutcome:
    """Wrap the polling driver in the project's retry-on-failure semantics.

    Pruned and completed outcomes return immediately. Only ``failed``
    outcomes (subprocess returncode != 0 with no prune signal) are retried,
    so a deliberately stopped trial is never resurrected.

    A subtle constraint: ``optuna_trial.report`` calls accumulate on the
    Optuna trial object across retries. If the failed attempt already
    reported intermediate values, those values stay on the trial and will
    bias the pruner's decisions during the retry (and Optuna may also
    silently drop duplicate-step reports). We therefore refuse to retry
    once any reports have been sent — the trial fails outright instead.
    """
    attempts = 0
    while True:
        attempts += 1
        outcome = _run_trial_with_pruning(
            artifact,
            gpu_group,
            optuna_trial,
            metric,
            poll_interval,
            attempt=attempts,
        )
        if outcome.state in ("completed", "pruned"):
            return outcome
        if outcome.launch_error:
            if attempts > retry_budget:
                if outcome.launch_exception is not None:
                    _write_launch_failure_status(artifact, outcome.launch_exception)
                return outcome
            continue
        if outcome.reports_sent > 0:
            # Stale intermediate reports would bias the retry; surface the
            # failure and let the caller record TrialState.FAIL.
            return outcome
        if attempts > retry_budget:
            return outcome


def _load_previous_variants(config: SweepConfig) -> list[dict[str, Any]]:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Optuna resume cannot reuse existing trials because the previous manifest "
            "is not valid JSON. Restore the manifest or start a fresh study/output_dir."
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(
            "Optuna resume cannot reuse existing trials because the previous manifest "
            "is not a JSON object. Restore the manifest or start a fresh study/output_dir."
        )
    variants = manifest.get("variants", [])
    if not isinstance(variants, list):
        raise RuntimeError(
            "Optuna resume requires manifest variants to be recorded as a list. "
            "Restore the manifest or start a fresh study/output_dir."
        )
    non_object_entries = [idx for idx, variant in enumerate(variants) if not isinstance(variant, dict)]
    if non_object_entries:
        raise RuntimeError(
            "Optuna resume requires every manifest variant entry to be a JSON object. "
            f"Invalid variant index(es): {non_object_entries}. Repair the manifest or start a fresh study/output_dir."
        )
    return variants


def _variants_for_trial_number(previous_variants: list[dict[str, Any]], trial_number: int) -> list[dict[str, Any]]:
    prefix = f"{trial_number:04d}-"
    return [variant for variant in previous_variants if str(variant.get("id", "")).startswith(prefix)]


def _variant_trial_number(variant: dict[str, Any]) -> int | None:
    raw_id = variant.get("id")
    if not isinstance(raw_id, str):
        return None
    prefix, separator, _ = raw_id.partition("-")
    if separator != "-" or not prefix.isdigit():
        return None
    return int(prefix)


def _read_manifest_status(status_path: Path, trial_number: int) -> dict[str, Any]:
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Optuna resume requires status.json files to be valid JSON objects. "
            f"Invalid status for trial number {trial_number} at {status_path}. "
            "Restore the trial artifacts or start fresh."
        ) from exc
    if not isinstance(status, dict):
        raise RuntimeError(
            "Optuna resume requires status.json files to be valid JSON objects. "
            f"Status for trial number {trial_number} at {status_path} is {type(status).__name__}. "
            "Restore the trial artifacts or start fresh."
        )
    return status


def _validate_resume_manifest_coverage(study: Any, previous_variants: list[dict[str, Any]]) -> None:
    """Fail closed when Optuna storage has trials the manifest cannot describe."""
    storage_numbers = {trial.number for trial in study.trials}
    manifest_numbers = [_variant_trial_number(variant) for variant in previous_variants]
    invalid_manifest_ids = [
        variant.get("id") for variant, number in zip(previous_variants, manifest_numbers) if number is None
    ]
    manifest_counts = Counter(number for number in manifest_numbers if number is not None)
    manifest_without_storage = sorted(number for number in manifest_counts if number not in storage_numbers)
    missing: list[int] = []
    duplicate = sorted(number for number, count in manifest_counts.items() if count > 1)
    missing_status: list[int] = []
    missing_resolved_checksum: list[int] = []
    status_id_mismatches: list[str] = []
    for trial in study.trials:
        variants = _variants_for_trial_number(previous_variants, trial.number)
        if not variants:
            missing.append(trial.number)
            continue
        variant_id = variants[0].get("id")
        raw_status_path = variants[0].get("status_path")
        if not isinstance(raw_status_path, str) or not raw_status_path:
            missing_status.append(trial.number)
            continue
        status_path = Path(raw_status_path)
        if not status_path.is_file():
            missing_status.append(trial.number)
            continue
        status = _read_manifest_status(status_path, trial.number)
        status_id = status.get("id")
        if status_id != variant_id:
            status_id_mismatches.append(
                f"trial {trial.number}: manifest id={variant_id!r}, status id={status_id!r}"
            )
        resolved_checksum = variants[0].get("resolved_checksum")
        if not isinstance(resolved_checksum, str):
            missing_resolved_checksum.append(trial.number)
    if missing:
        raise RuntimeError(
            "Optuna resume requires manifest variant entries for all existing storage trials. "
            f"Missing trial number(s): {missing}. Restore the manifest or start a fresh study/output_dir."
        )
    if duplicate:
        raise RuntimeError(
            "Optuna resume requires exactly one manifest variant entry per existing storage trial. "
            f"Duplicate trial number(s): {duplicate}. Repair the manifest or start a fresh study/output_dir."
        )
    if invalid_manifest_ids or manifest_without_storage:
        raise RuntimeError(
            "Optuna resume requires manifest variant entries and storage trials to agree. "
            f"Invalid manifest id(s): {invalid_manifest_ids}; "
            f"manifest trial number(s) missing from storage: {manifest_without_storage}. "
            "Restore the Optuna storage or start a fresh study/output_dir."
        )
    if missing_status:
        raise RuntimeError(
            "Optuna resume requires status.json files for all existing storage trials. "
            f"Missing status for trial number(s): {missing_status}. Restore the trial artifacts or start fresh."
        )
    if missing_resolved_checksum:
        raise RuntimeError(
            "Optuna resume requires manifest resolved_checksum entries for all existing storage trials. "
            f"Missing resolved_checksum for trial number(s): {missing_resolved_checksum}. "
            "Restore the manifest or start a fresh study/output_dir."
        )
    if status_id_mismatches:
        raise RuntimeError(
            "Optuna resume requires each status.json id to match its manifest variant id. "
            f"Mismatch(es): {status_id_mismatches}. Restore the trial artifacts or start fresh."
        )


def _validate_resume_manifest_trial_parameters(
    study: Any, previous_variants: list[dict[str, Any]]
) -> None:
    """Fail closed when manifest variant ids/overrides drift from Optuna storage."""
    mismatches: list[str] = []
    for trial in study.trials:
        variants = _variants_for_trial_number(previous_variants, trial.number)
        if len(variants) != 1:
            # _validate_resume_manifest_coverage reports missing / duplicate
            # variants; keep this check focused on parameter identity.
            continue
        variant = variants[0]
        variant_id = variant.get("id")
        overrides = variant.get("overrides")
        if not isinstance(overrides, dict):
            mismatches.append(f"trial {trial.number}: manifest id={variant_id!r} has no overrides object")
            continue

        storage_params = dict(trial.params)
        if overrides != storage_params:
            mismatches.append(
                f"trial {trial.number}: storage params={storage_params!r}, "
                f"manifest overrides={overrides!r}"
            )

        expected_id = f"{trial.number:04d}-{parameters_hash(overrides)}"
        if variant_id != expected_id:
            mismatches.append(
                f"trial {trial.number}: manifest id={variant_id!r}, expected id={expected_id!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Optuna resume requires manifest variant ids and overrides to match Optuna storage "
            f"parameters. Mismatch(es): {mismatches}. Restore the manifest/storage or start fresh."
        )


def _validate_resume_base_checksums(
    config: SweepConfig, study: Any, previous_variants: list[dict[str, Any]]
) -> None:
    """Fail closed when resumed Optuna trials came from different base TOML(s)."""
    current_base_checksums = {base.as_posix(): file_checksum(base) for base in config.base}
    missing_checksum_trials: list[int] = []
    missing_bases: dict[int, list[str]] = {}
    extra_bases: dict[int, list[str]] = {}
    changed_bases: dict[int, list[str]] = {}

    for trial in study.trials:
        variants = _variants_for_trial_number(previous_variants, trial.number)
        if len(variants) != 1:
            # _validate_resume_manifest_coverage reports missing / duplicate
            # variants; keep this check focused on checksum drift.
            continue
        expected_bases = variants[0].get("base_checksums")
        if not isinstance(expected_bases, dict) or not expected_bases:
            missing_checksum_trials.append(trial.number)
            continue

        expected_base_checksums = {str(path): checksum for path, checksum in expected_bases.items()}
        current_paths = set(current_base_checksums)
        expected_paths = set(expected_base_checksums)

        missing = sorted(current_paths - expected_paths)
        extra = sorted(expected_paths - current_paths)
        changed = sorted(
            path
            for path, checksum in current_base_checksums.items()
            if path in expected_base_checksums and expected_base_checksums[path] != checksum
        )
        if missing:
            missing_bases[trial.number] = missing
        if extra:
            extra_bases[trial.number] = extra
        if changed:
            changed_bases[trial.number] = changed

    if not (missing_checksum_trials or missing_bases or extra_bases or changed_bases):
        return

    raise RuntimeError(
        "Optuna resume requires manifest base config checksums for every existing storage trial, "
        "and they must match the current base files. "
        f"Missing checksum trial(s): {missing_checksum_trials}; "
        f"missing base(s): {missing_bases}; extra base(s): {extra_bases}; "
        f"changed base(s): {changed_bases}. "
        "Restore the original base config(s) or start a fresh study/output_dir."
    )


def _validate_resume_status_consistency(
    optuna: Any, study: Any, previous_variants: list[dict[str, Any]]
) -> None:
    """Fail closed when terminal Optuna storage and sweep status files disagree."""
    mismatches: list[str] = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            continue

        _status_path, status = _variant_status_for_trial_number(previous_variants, trial.number)
        if status is None:
            # _validate_resume_manifest_coverage reports missing status files.
            continue

        recorded_state = status.get("state")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            status_objective = coerce_finite_float(status.get("objective"))
            storage_objective = coerce_finite_float(trial.value)
            if (
                recorded_state != "completed"
                or status_objective is None
                or storage_objective is None
                or status_objective != storage_objective
            ):
                mismatches.append(
                    f"trial {trial.number}: storage=COMPLETE({storage_objective!r}), "
                    f"status={recorded_state!r}({status.get('objective')!r})"
                )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            status_objective = coerce_finite_float(status.get("objective"))
            if recorded_state != "pruned" or status_objective is not None:
                mismatches.append(
                    f"trial {trial.number}: storage=PRUNED, "
                    f"status={recorded_state!r}({status.get('objective')!r})"
                )
        elif trial.state == optuna.trial.TrialState.FAIL:
            status_objective = coerce_finite_float(status.get("objective"))
            if recorded_state != "failed" or status_objective is not None:
                mismatches.append(
                    f"trial {trial.number}: storage=FAIL, "
                    f"status={recorded_state!r}({status.get('objective')!r})"
                )
        else:
            mismatches.append(
                f"trial {trial.number}: unsupported storage state {trial.state!r} "
                f"with status={recorded_state!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Optuna resume requires terminal status.json files to match Optuna storage state. "
            f"Mismatch(es): {mismatches}. Restore the trial artifacts/storage or start fresh."
        )


def _count_optuna_failures(optuna: Any, study: Any) -> int:
    return sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL)


def _seed_tracker_from_previous(tracker: TrialOutcomeTracker, previous_variants: list[dict[str, Any]]) -> None:
    for variant in previous_variants:
        raw_status_path = variant.get("status_path")
        if not isinstance(raw_status_path, str) or not raw_status_path:
            continue
        status_path = Path(raw_status_path)
        if not status_path.is_file():
            continue
        trial_number = _variant_trial_number(variant)
        status = _read_manifest_status(status_path, -1 if trial_number is None else trial_number)
        if status.get("state") != "completed":
            continue
        tracker.observe(
            TrialOutcome(
                trial_id=variant.get("id", ""),
                label=variant.get("label", "") or variant.get("id", ""),
                objective=status.get("objective"),
            )
        )


def _variant_status_for_trial_number(
    previous_variants: list[dict[str, Any]],
    trial_number: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Match an Optuna trial number to its sweep status via the ``NNNN-...`` id prefix."""
    prefix = f"{trial_number:04d}-"
    for variant in previous_variants:
        if not variant.get("id", "").startswith(prefix):
            continue
        raw_status_path = variant.get("status_path")
        if not isinstance(raw_status_path, str) or not raw_status_path:
            return None, None
        status_path = Path(raw_status_path)
        if not status_path.is_file():
            return status_path, None
        return status_path, _read_manifest_status(status_path, trial_number)
    return None, None


def _reconcile_running_trials(
    optuna: Any, study: Any, previous_variants: list[dict[str, Any]]
) -> tuple[int, int]:
    """Tell Optuna about any RUNNING trials left over from an interrupted run.

    A controller crash between ``study.ask()`` and ``study.tell()`` leaves a
    trial RUNNING in persistent storage forever. On resume we walk those
    trials and:

    - if the matching sweep status.json shows ``completed`` with a finite
      objective, tell Optuna the value so adaptive sampling can use it;
    - if status.json shows ``pruned``, tell ``TrialState.PRUNED`` so the
      sampler treats the slot as a deliberate stop (a crash between
      ``record_trial_pruned`` and ``study.tell(PRUNED)`` would otherwise
      misclassify it as a failure);
    - otherwise tell ``TrialState.FAIL`` and mark any matching stale
      status file failed so Optuna storage and the sweep manifest agree.

    Returns ``(reconciled, failures)``. ``failures`` counts RUNNING trials
    reconciled to ``TrialState.FAIL`` so the sweep process exits non-zero in
    the same way it would have if the original controller had observed the
    failure before crashing.
    """
    reconciled = 0
    failures = 0
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        status_path, status = _variant_status_for_trial_number(previous_variants, trial.number)
        recorded_state = status.get("state") if status is not None else None
        objective: float | None = None
        if recorded_state == "completed":
            value = status.get("objective") if status is not None else None
            objective = coerce_finite_float(value)
        # study.tell() accepts a trial number or a Trial; FrozenTrial is not
        # accepted, so pass trial.number.
        if recorded_state == "pruned":
            study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
            if status_path is not None and status_path.is_file():
                pruned_status = status or {}
                pruned_status["objective"] = None
                write_json(status_path, pruned_status)
        elif objective is not None:
            study.tell(trial.number, objective)
        else:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            failures += 1
            if status_path is not None and status_path.is_file():
                failed_status = status or {}
                raw_returncode = failed_status.get("returncode")
                returncode = raw_returncode if type(raw_returncode) is int else -1
                failed_status.update(
                    {
                        "state": "failed",
                        "finished_at": utc_now(),
                        "returncode": returncode,
                        "objective": None,
                    }
                )
                if returncode == 0:
                    failed_status["failure_stage"] = "objective"
                    failed_status["error"] = (
                        "Trial exited successfully but did not record a finite objective before resume."
                    )
                write_json(status_path, failed_status)
        reconciled += 1
    return reconciled, failures


def run_optuna_sweep(
    config: SweepConfig,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None, list[TrialArtifacts]]:
    """Drive an Optuna study end-to-end.

    Returns ``(failures, tracker, artifacts)`` so the caller can write the
    final manifest summary and exit code in the same shape as the static
    flow. Resume honors persistent storage: previously consumed slots in
    ``study.trials`` are not re-asked, the manifest preserves earlier
    variants, and the tracker is seeded from prior outcomes.
    """
    optuna = _import_optuna()
    strategy = config.strategy
    assert isinstance(strategy, OptunaStrategyConfig)
    # Local and synchronous-SLURM schedulers are both supported. Asynchronous
    # SLURM and multi_run_lora are rejected upstream by the SweepConfig validator.
    assert isinstance(config.scheduler, (LocalSweepSchedulerConfig, SlurmSweepSchedulerConfig))

    study = _create_study(optuna, config)

    if isinstance(config.scheduler, LocalSweepSchedulerConfig):
        gpu_groups = (
            config.scheduler.gpu_assignment.visible_devices
            if config.scheduler.gpu_assignment is not None
            else None
        )
        gpu_group = gpu_groups[0] if gpu_groups else None
    else:
        gpu_group = None

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if config.objective else None

    previous_variants = _load_previous_variants(config) if config.resume else []
    artifacts: list[TrialArtifacts] = []
    failures = 0
    if config.resume:
        _validate_resume_manifest_coverage(study, previous_variants)
        _validate_resume_manifest_trial_parameters(study, previous_variants)
        _validate_resume_base_checksums(config, study, previous_variants)
        _validate_resume_status_consistency(optuna, study, previous_variants)
        reconciled, _ = _reconcile_running_trials(optuna, study, previous_variants)
        if reconciled:
            print(f"Reconciled {reconciled} RUNNING Optuna trial(s) from interrupted resume.")
        failures = _count_optuna_failures(optuna, study)
        if tracker is not None:
            _seed_tracker_from_previous(tracker, previous_variants)
        if failures > 0 and not config.continue_on_failure:
            return failures, tracker, artifacts

    already_consumed = len(study.trials) if config.resume else 0

    for index in range(already_consumed, strategy.num_trials):
        if tracker is not None and tracker.halted:
            break

        optuna_trial = study.ask()
        params = _suggest_parameters(optuna_trial, config.parameters)
        trial = _make_trial(index, params)

        try:
            artifact = materialize_trial(config, trial)
        except Exception as exc:
            # Sampled parameters failed target-config validation. Mark the
            # asked trial failed in Optuna so persistent storage doesn't
            # leak a RUNNING slot, and write a manifest/status artifact so a
            # later resume can account for the terminal storage trial.
            artifact = record_trial_materialization_failure(
                config, trial, exc, finished_at=utc_now()
            )
            artifacts.append(artifact)
            write_manifest_with_variants(
                config, previous_variants + [build_variant(a) for a in artifacts]
            )
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            failures += 1
            print(f"Optuna trial {index:04d} failed materialization: {exc}")
            if not config.continue_on_failure:
                break
            continue

        artifacts.append(artifact)
        write_manifest_with_variants(config, previous_variants + [build_variant(a) for a in artifacts])
        stop_after_trial = False

        slurm_sync = (
            isinstance(config.scheduler, SlurmSweepSchedulerConfig)
            and config.scheduler.synchronous
        )
        if isinstance(strategy.pruner, NoPrunerConfig):
            if slurm_sync:
                returncode = _run_with_retries_slurm_sync(artifact, config.retry_budget)
            else:
                returncode = _run_with_retries(artifact, gpu_group, config.retry_budget)
            objective_value = (
                read_final_summary(artifact.run_dir, config.objective.metric)
                if returncode == 0 and config.objective is not None
                else None
            )
            record_trial_objective(artifact.status_path, objective_value)

            if objective_value is None:
                if returncode == 0:
                    record_trial_missing_objective(artifact.status_path, config.objective.metric)
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(optuna_trial, objective_value)

            # A clean exit without a logged objective (returncode==0 but
            # objective_value is None) is also a sweep-level failure: Optuna
            # learned nothing from it, the sampler recorded TrialState.FAIL,
            # and the user almost certainly wants to be alerted rather than
            # let the sweep finish 'successfully' with no usable results.
            if returncode != 0 or objective_value is None:
                failures += 1
                if not config.continue_on_failure:
                    stop_after_trial = True
        else:
            if slurm_sync:
                outcome = _run_trial_with_pruning_slurm_sync_and_retries(
                    artifact,
                    optuna_trial,
                    config.objective.metric,
                    strategy.poll_interval_seconds,
                    config.retry_budget,
                )
            else:
                outcome = _run_trial_with_pruning_and_retries(
                    artifact,
                    gpu_group,
                    optuna_trial,
                    config.objective.metric,
                    strategy.poll_interval_seconds,
                    config.retry_budget,
                )
            objective_value = outcome.objective
            if outcome.state == "completed":
                record_trial_objective(artifact.status_path, objective_value)
                if objective_value is None:
                    # Completed without a recorded objective (e.g. metric
                    # never logged): treat as a sweep-level failure too,
                    # not just an Optuna FAIL — the sweep produced no
                    # usable result for this trial.
                    record_trial_missing_objective(artifact.status_path, config.objective.metric)
                    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                    failures += 1
                    if not config.continue_on_failure:
                        stop_after_trial = True
                else:
                    study.tell(optuna_trial, objective_value)
            elif outcome.state == "pruned":
                # record_trial_pruned already set status.json fields.
                study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
            else:  # failed
                record_trial_objective(artifact.status_path, None)
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
                if not config.continue_on_failure or outcome.unsafe_to_continue:
                    # ``unsafe_to_continue`` forces a halt even when the user
                    # set ``continue_on_failure=True``: the underlying SLURM
                    # job may still be running (persistent squeue failure or
                    # unconfirmed scancel), and submitting the next trial
                    # would race the still-active allocation.
                    stop_after_trial = True

        if tracker is not None:
            tracker_outcome = TrialOutcome(
                trial_id=trial.id,
                label=trial.label,
                objective=objective_value,
            )
            if tracker.observe(tracker_outcome):
                break
        if stop_after_trial:
            break

    write_manifest_with_variants(config, previous_variants + [build_variant(a) for a in artifacts])

    return failures, tracker, artifacts


@dataclass
class _SlurmSyncWorkerResult:
    """Outcome of one worker thread running a SLURM-sync trial."""

    returncode: int
    objective: float | None


def _run_one_slurm_sync_no_pruner(
    artifact: TrialArtifacts,
    metric: str,
    retry_budget: int,
) -> _SlurmSyncWorkerResult:
    """Worker function for parallel SLURM-sync sweeps without a pruner.

    Synchronous SLURM with no pruner: submit with ``sbatch --wait`` (via
    ``_run_with_retries_slurm_sync``) and read the final objective from
    metrics.jsonl on success.
    """
    returncode = _run_with_retries_slurm_sync(artifact, retry_budget)
    objective = read_final_summary(artifact.run_dir, metric) if returncode == 0 else None
    return _SlurmSyncWorkerResult(returncode=returncode, objective=objective)


def _run_one_slurm_sync_with_pruner(
    artifact: TrialArtifacts,
    optuna_trial: "optuna.Trial",  # noqa: F821 — forward ref for optional dep
    metric: str,
    poll_interval: float,
    retry_budget: int,
) -> "_PollingOutcome":
    """Worker function for parallel SLURM-sync sweeps with a pruner.

    Delegates to ``_run_trial_with_pruning_slurm_sync_and_retries`` (the
    same code path used by the serial pruner+SLURM-sync runner). Optuna's
    storage backend serializes concurrent ``report``/``should_prune``
    calls across worker threads — Optuna's own ``study.optimize(n_jobs>1)``
    documents this contract — so each worker holding its own
    ``optuna_trial`` is safe even when N polling threads call
    ``should_prune`` simultaneously against the same study.
    """
    return _run_trial_with_pruning_slurm_sync_and_retries(
        artifact,
        optuna_trial,
        metric,
        poll_interval,
        retry_budget,
    )


def run_optuna_sweep_parallel_slurm(
    config: SweepConfig,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None, list[TrialArtifacts]]:
    """Drive an Optuna study with up to ``max_parallel`` concurrent SLURM jobs.

    Architecture: the main thread owns the Optuna study and all ask/tell
    interactions; a ``ThreadPoolExecutor`` with ``max_workers=max_parallel``
    runs the per-trial ``sbatch --wait`` calls so up to N trials can be
    in flight at once. As each future completes the main thread tells
    Optuna the outcome and immediately asks for one more trial to refill
    the slot, until ``num_trials`` is reached or a halt condition fires.

    Pruners are supported under parallel SLURM-sync as of this PR. Each
    worker holds its own ``optuna_trial`` (each from a distinct
    ``study.ask()``) and runs the same polling loop the serial pruner
    runner uses. Optuna's storage backend serializes the concurrent
    ``report``/``should_prune`` calls across threads, the same contract
    that makes ``study.optimize(n_jobs>1)`` work in stock Optuna.

    Halt conditions:
    - ``tracker.observe`` returns True → stop submitting new trials, wait
      for in-flight to finish (early stopping).
    - ``unsafe_to_continue`` from a worker (persistent squeue failure or
      unconfirmed scancel during a prune) → halt new submissions, drain.
      Only fires when a pruner is active (the no-pruner runner doesn't
      emit it).
    - ``not config.continue_on_failure`` after any failure → halt new
      submissions, drain.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    optuna = _import_optuna()
    strategy = config.strategy
    assert isinstance(strategy, OptunaStrategyConfig)
    assert isinstance(config.scheduler, SlurmSweepSchedulerConfig)
    assert config.scheduler.synchronous, "parallel SLURM requires synchronous=true"
    max_parallel = config.scheduler.max_parallel
    assert max_parallel > 1
    use_pruner = not isinstance(strategy.pruner, NoPrunerConfig)

    study = _create_study(optuna, config)
    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if config.objective else None

    previous_variants = _load_previous_variants(config) if config.resume else []
    artifacts: list[TrialArtifacts] = []
    failures = 0
    if config.resume:
        _validate_resume_manifest_coverage(study, previous_variants)
        _validate_resume_manifest_trial_parameters(study, previous_variants)
        _validate_resume_base_checksums(config, study, previous_variants)
        _validate_resume_status_consistency(optuna, study, previous_variants)
        reconciled, _ = _reconcile_running_trials(optuna, study, previous_variants)
        if reconciled:
            print(f"Reconciled {reconciled} RUNNING Optuna trial(s) from interrupted resume.")
        failures = _count_optuna_failures(optuna, study)
        if tracker is not None:
            _seed_tracker_from_previous(tracker, previous_variants)
        if failures > 0 and not config.continue_on_failure:
            return failures, tracker, artifacts

    already_consumed = len(study.trials) if config.resume else 0
    next_index = already_consumed
    halted = False
    # Manifest writes happen on the main thread (single producer), so no
    # lock is needed — but Optuna's in-memory study is read by the
    # asker; ask/tell calls are all serialized by the main thread loop.

    def ask_and_materialize() -> tuple[Any, TrialArtifacts | None] | None:
        """Get the next trial from Optuna and materialize it.

        Returns ``(optuna_trial, artifact)`` on success, ``(optuna_trial, None)``
        when materialization fails (caller marks Optuna FAIL and skips), or
        ``None`` when there are no more trials to ask for.
        """
        nonlocal next_index, failures
        if halted or next_index >= strategy.num_trials:
            return None
        if tracker is not None and tracker.halted:
            return None
        index = next_index
        next_index += 1
        optuna_trial = study.ask()
        params = _suggest_parameters(optuna_trial, config.parameters)
        trial = _make_trial(index, params)
        try:
            artifact = materialize_trial(config, trial)
        except Exception as exc:
            failed_artifact = record_trial_materialization_failure(
                config, trial, exc, finished_at=utc_now()
            )
            artifacts.append(failed_artifact)
            write_manifest_with_variants(
                config, previous_variants + [build_variant(a) for a in artifacts]
            )
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            failures += 1
            print(f"Optuna trial {index:04d} failed materialization: {exc}")
            return (optuna_trial, None)
        artifacts.append(artifact)
        write_manifest_with_variants(
            config, previous_variants + [build_variant(a) for a in artifacts]
        )
        return (optuna_trial, artifact)

    with ThreadPoolExecutor(
        max_workers=max_parallel, thread_name_prefix="slurm-sync-trial"
    ) as executor:
        in_flight: dict = {}

        def submit_one() -> bool:
            """Ask Optuna for one trial and submit it to the executor.

            Returns True when a trial is in flight, False when there are
            no more trials to launch (either reached ``num_trials`` or a
            halt condition is active).
            """
            while True:
                asked = ask_and_materialize()
                if asked is None:
                    return False
                optuna_trial, artifact = asked
                if artifact is None:
                    # materialization failure already recorded; try next
                    # index to keep the slot full.
                    if not config.continue_on_failure:
                        return False
                    continue
                if use_pruner:
                    future = executor.submit(
                        _run_one_slurm_sync_with_pruner,
                        artifact,
                        optuna_trial,
                        config.objective.metric,
                        strategy.poll_interval_seconds,
                        config.retry_budget,
                    )
                else:
                    future = executor.submit(
                        _run_one_slurm_sync_no_pruner,
                        artifact,
                        config.objective.metric,
                        config.retry_budget,
                    )
                in_flight[future] = (optuna_trial, artifact)
                return True

        for _ in range(max_parallel):
            if not submit_one():
                break

        while in_flight:
            done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                optuna_trial, artifact = in_flight.pop(future)
                trial_id = artifact.trial.id
                trial_label_ = artifact.trial.label
                try:
                    result = future.result()
                except Exception as exc:
                    # Worker threw — surface as failure; the only paths
                    # that raise here are programming errors since
                    # _run_with_retries_slurm_sync catches OSError.
                    record_trial_objective(artifact.status_path, None)
                    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                    failures += 1
                    print(f"Optuna trial {trial_id} worker raised: {exc}")
                    if not config.continue_on_failure:
                        halted = True
                    continue

                objective_value = result.objective
                if isinstance(result, _PollingOutcome):
                    # Pruner path: result carries a tristate (completed/
                    # pruned/failed) plus the unsafe_to_continue flag for
                    # the SLURM-specific failure modes.
                    if result.state == "completed":
                        record_trial_objective(artifact.status_path, objective_value)
                        if objective_value is None:
                            record_trial_missing_objective(
                                artifact.status_path, config.objective.metric
                            )
                            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                            failures += 1
                            if not config.continue_on_failure:
                                halted = True
                        else:
                            study.tell(optuna_trial, objective_value)
                    elif result.state == "pruned":
                        # record_trial_pruned already set status.json fields
                        # in the worker; just tell Optuna so the sampler
                        # treats this trial as a deliberate stop.
                        study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
                    else:  # failed
                        record_trial_objective(artifact.status_path, None)
                        study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                        failures += 1
                        if not config.continue_on_failure or result.unsafe_to_continue:
                            # unsafe_to_continue forces halt regardless of
                            # continue_on_failure: persistent squeue
                            # failure or unconfirmed scancel means the
                            # SLURM job may still be alive and submitting
                            # the next trial would race it.
                            halted = True
                else:
                    # No-pruner path: result is _SlurmSyncWorkerResult.
                    returncode = result.returncode
                    record_trial_objective(artifact.status_path, objective_value)

                    if objective_value is None:
                        if returncode == 0:
                            record_trial_missing_objective(
                                artifact.status_path, config.objective.metric
                            )
                        study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        study.tell(optuna_trial, objective_value)

                    if returncode != 0 or objective_value is None:
                        failures += 1
                        if not config.continue_on_failure:
                            halted = True

                if tracker is not None:
                    if tracker.observe(
                        TrialOutcome(
                            trial_id=trial_id,
                            label=trial_label_,
                            objective=objective_value,
                        )
                    ):
                        halted = True

                # Refill the freed slot unless we are halted.
                if not halted:
                    submit_one()

    write_manifest_with_variants(config, previous_variants + [build_variant(a) for a in artifacts])
    return failures, tracker, artifacts


# Re-exported for the controller to update the manifest summary.
def tracker_summary(tracker: TrialOutcomeTracker) -> dict[str, Any]:
    return asdict(tracker.summary())
