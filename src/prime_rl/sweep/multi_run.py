"""Sweep-side runtime helpers for shared-trainer LoRA multi-run sweeps.

The trainer (``prime_rl/trainer/runs.py:MultiRunManager``) writes
``<run_dir>/control/evicted.txt`` to evict a run when it's about to lose its
LoRA slot. The orchestrator (``prime_rl/orchestrator/orchestrator.py``) polls
that same file at the top of each training loop iteration and exits.

Phase 7b adds a third writer: the sweep controller itself, when an Optuna
sampler decides one of the in-flight trials should be pruned. This module is
the bridge plus the wave driver that runs Optuna against ``rl-multi-run``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.sweep import (
    MultiRunLoRASchedulerConfig,
    OptunaStrategyConfig,
    SweepConfig,
)
from prime_rl.sweep import materialize as _materialize
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    TrialArtifacts,
    materialize_multi_run_trial,
    multi_run_shared_dir,
    read_status_json,
    record_multi_run_materialization_failure,
    record_trial_missing_objective,
    record_trial_objective,
)
from prime_rl.sweep.metrics import read_final_summary, read_intermediate_metric
from prime_rl.sweep.optuna_loop import (
    _create_study,
    _import_optuna,
    _make_trial,
    _suggest_parameters,
)
from prime_rl.sweep.schedulers import (
    _mark_inactive_multi_run_dirs_evicted,
    _read_orchestrator_exit_code,
    _read_status,
    _reset_multi_run_artifact_runtime,
    _write_launch_failure_status,
    _write_status,
    build_multi_run_command,
    reconcile_multi_run_artifact,
    utc_now,
)

if TYPE_CHECKING:  # pragma: no cover
    import optuna

EVICTED_FILENAME = "evicted.txt"

# Subdirs the trainer writes under its ``output_dir`` between runs. Each
# Optuna wave starts a fresh trainer pinned to ``shared_dir``, so leftover
# state from a previous wave (checkpoints, weights, broadcasts, rollouts)
# would be picked up by the new trainer — silently resuming with stale
# checkpoints, or colliding on ``step_*`` writes. Per-trial ``run_<id>``
# directories are *not* listed here; the controller manages those via
# ``_mark_inactive_multi_run_dirs_evicted``.
_TRAINER_OWNED_SUBDIRS = ("weights", "broadcasts", "rollouts", "run_default")


def _resolve_shared_ckpt_dir(shared_paths: list[Path], shared_dir: Path) -> Path:
    """Find where the shared trainer's checkpoints land for this study.

    Falls back to ``<shared_dir>/checkpoints`` when ``ckpt.output_dir`` is
    not set in the shared RLConfig (the auto_setup_ckpt default).
    """
    args: list[str] = []
    for base in shared_paths:
        args.extend(["@", base.as_posix()])
    resolved = _materialize.validate_target_config("rl", args)
    ckpt = getattr(resolved, "ckpt", None)
    override = getattr(ckpt, "output_dir", None) if ckpt is not None else None
    if override is not None:
        return Path(override) / "checkpoints"
    return shared_dir / "checkpoints"


def _reset_trainer_state_for_wave(shared_dir: Path, ckpt_dir: Path) -> None:
    """Wipe trainer-owned artifacts before launching the next Optuna wave.

    Each wave runs a fresh ``rl-multi-run`` whose trainer pins
    ``output_dir = shared_dir``. Without this reset, the trainer for wave
    N+1 starts on top of wave N's checkpoints, weights, and step files;
    checkpoint-enabled configs would silently resume from stale state, and
    broadcast/rollout step directories from the prior wave would collide
    with the new run's step 0 writes. Per-trial ``run_<id>`` directories
    are intentionally preserved — the sweep controller materializes them
    before this function runs.
    """
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    for subdir in _TRAINER_OWNED_SUBDIRS:
        path = shared_dir / subdir
        if path.exists():
            shutil.rmtree(path)


def prune_run(
    run_dir: Path,
    reason: str,
    *,
    step: int | None = None,
    value: float | None = None,
) -> None:
    """Pre-mark the trial pruned in ``status.json`` then write ``evicted.txt``.

    Order matters: the orchestrator's eviction handler raises ``RuntimeError``
    and the orchestrator exits non-zero. If we wrote ``evicted.txt`` first and
    crashed before updating ``status.json``, the launcher's exit-code
    reconciliation would misclassify the deliberately-pruned trial as
    ``failed`` (it has no way to know the eviction was a sampler decision
    rather than a slot-pressure eviction from the trainer).

    ``step`` and ``value`` are recorded on the status when the caller has them
    — Optuna prunes know both — but they're optional so callers without that
    context (manual prune, future heuristics) can still mark the trial pruned.
    """
    status_path = run_dir / "status.json"
    status = read_status_json(status_path)
    status["state"] = "pruned"
    status["pruned_reason"] = reason
    if step is not None:
        status["pruned_at_step"] = int(step)
    if value is not None:
        status["pruned_value"] = float(value)
    # Surfacing pruned trials with a None objective keeps the manifest
    # summary's best-value computation symmetric with single-trial pruning
    # (see materialize.record_trial_pruned).
    status["objective"] = None
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / EVICTED_FILENAME).write_text(reason + "\n")


def _poll_wave_for_pruning(
    optuna: Any,
    proc: subprocess.Popen[bytes],
    artifacts: list[TrialArtifacts],
    optuna_trials: list[optuna.Trial],
    metric: str,
    poll_interval: float,
) -> None:
    """While ``rl-multi-run`` runs, drive Optuna's report/should_prune for each artifact.

    On each tick:

    1. For every artifact whose status is not already ``pruned`` and whose
       orchestrator has not already recorded ``control/exit_code``, read the
       latest ``(step, value)`` from its ``metrics.jsonl`` sidecar.
    2. If we've never reported this step (or any step at all) for this trial,
       call ``optuna_trial.report(value, step)`` and check ``should_prune``.
    3. On a prune signal, write ``status.json`` + ``evicted.txt`` so the
       orchestrator winds down. The trainer's MultiRunManager picks the same
       file up on its next ``discover_runs()`` cycle and frees the LoRA slot.

    The trainer-side eviction handles process termination — we never SIGTERM
    the orchestrator ourselves. Survivors keep running until the wave's
    ``rl-multi-run`` exits naturally.
    """
    last_step: dict[Path, int | None] = {a.run_dir: None for a in artifacts}

    while proc.poll() is None:
        for artifact, optuna_trial in zip(artifacts, optuna_trials):
            try:
                status = _read_status(artifact)
            except FileNotFoundError:
                continue
            if status.get("state") == "pruned":
                continue
            if _read_orchestrator_exit_code(artifact) is not None:
                continue

            sample = read_intermediate_metric(artifact.run_dir, metric)
            if sample is None:
                continue
            step, value = sample
            prev = last_step[artifact.run_dir]
            if prev is not None and step <= prev:
                continue

            optuna_trial.report(value, step)
            last_step[artifact.run_dir] = step

            if _read_orchestrator_exit_code(artifact) is not None:
                continue
            if proc.poll() is not None:
                break
            if optuna_trial.should_prune():
                # The wave may finish between the loop guard and this
                # decision. Once rl-multi-run has exited, final objective
                # reconciliation wins and pruning must not rewrite a
                # completed run as pruned.
                if proc.poll() is not None:
                    break
                prune_run(
                    artifact.run_dir,
                    reason=f"optuna prune at step {step}",
                    step=step,
                    value=value,
                )
        time.sleep(poll_interval)


def _tell_wave_results(
    optuna: Any,
    study: optuna.Study,
    artifacts: list[TrialArtifacts],
    optuna_trials: list[optuna.Trial],
    metric: str,
    aggregate_returncode: int,
) -> tuple[int, list[float | None]]:
    """Reconcile per-trial state and tell Optuna each trial's result.

    Returns ``(failures, objectives)`` where ``objectives[i]`` is the recorded
    objective for ``artifacts[i]`` (or ``None`` for pruned/failed trials).
    The caller folds those into the ``TrialOutcomeTracker`` for the manifest
    summary.
    """
    finished_at = utc_now()
    failures = 0
    objectives: list[float | None] = []

    for artifact, optuna_trial in zip(artifacts, optuna_trials):
        state = reconcile_multi_run_artifact(
            artifact, aggregate_returncode=aggregate_returncode, finished_at=finished_at
        )

        objective: float | None = None
        if state == "completed":
            objective = read_final_summary(artifact.run_dir, metric)
            record_trial_objective(artifact.status_path, objective)
            if objective is None:
                # Clean exit but the metric never showed up — Optuna learned
                # nothing from this slot. Tell FAIL and count it as a sweep
                # failure (mirrors the single-trial Optuna driver).
                record_trial_missing_objective(artifact.status_path, metric)
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
            else:
                study.tell(optuna_trial, objective)
        elif state == "pruned":
            study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
        else:  # failed
            record_trial_objective(artifact.status_path, None)
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            failures += 1

        objectives.append(objective)

    return failures, objectives


def run_multi_run_optuna_sweep(
    config: SweepConfig,
    *,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None, list[TrialArtifacts]]:
    """Drive an Optuna study against ``rl-multi-run`` in waves.

    Each wave asks Optuna for ``min(max_concurrent_runs, remaining)`` trials,
    materializes them as ``run_*`` dirs under the shared trainer dir, spawns
    one ``rl-multi-run`` invocation, polls each run's ``metrics.jsonl`` for
    Optuna ``report``/``should_prune`` decisions, and finally tells Optuna
    each trial's result.

    Slot replacement is intentionally not supported: a slot freed mid-wave
    by pruning sits idle until the wave finishes. True slot replacement
    needs ``rl-multi-run`` to accept new run dirs over the wire (Phase 7c).
    """
    optuna = _import_optuna()
    strategy = config.strategy
    scheduler = config.scheduler
    assert isinstance(strategy, OptunaStrategyConfig)
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)
    assert config.objective is not None  # validated upstream

    study = _create_study(optuna, config)
    metric = config.objective.metric
    wave_size = scheduler.max_concurrent_runs
    total = strategy.num_trials
    poll_interval = strategy.poll_interval_seconds

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping)
    shared_dir = multi_run_shared_dir(config)
    shared_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _resolve_shared_ckpt_dir(scheduler.shared, shared_dir)

    all_artifacts: list[TrialArtifacts] = []
    failures = 0
    submitted = 0
    wave_index = 0

    while submitted < total:
        if tracker.halted:
            break
        this_wave = min(wave_size, total - submitted)
        stop_after_wave = False

        # Wipe trainer-owned artifacts from the prior wave before
        # materializing or launching this one. Wave 0 starts on a clean
        # ``shared_dir`` (just created above), so this is a no-op then;
        # subsequent waves inherit checkpoints/weights/broadcasts/rollouts
        # from the previous trainer and would otherwise resume or collide.
        if wave_index > 0:
            _reset_trainer_state_for_wave(shared_dir, ckpt_dir)
        wave_index += 1

        # 1. Ask Optuna for `this_wave` trials and materialize each.
        # Failed materializations are excluded from the launch wave but kept
        # in the manifest; survivors stay paired so the poll loop and
        # reconcile step have aligned (optuna_trial, artifact) lists.
        wave_pairs: list[tuple[optuna.Trial, TrialArtifacts]] = []
        wave_artifacts_for_manifest: list[TrialArtifacts] = []
        for offset in range(this_wave):
            optuna_trial = study.ask()
            params = _suggest_parameters(optuna_trial, config.parameters)
            sweep_trial = _make_trial(submitted + offset, params)
            try:
                artifact = materialize_multi_run_trial(config, sweep_trial, scheduler)
            except Exception as exc:
                artifact = record_multi_run_materialization_failure(
                    config, sweep_trial, scheduler, exc, finished_at=utc_now()
                )
                wave_artifacts_for_manifest.append(artifact)
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
                print(f"Optuna trial {sweep_trial.id} failed materialization: {exc}")
                if not config.continue_on_failure:
                    stop_after_wave = True
                    break
                continue
            wave_pairs.append((optuna_trial, artifact))
            wave_artifacts_for_manifest.append(artifact)

        if stop_after_wave:
            if wave_pairs:
                finished_at = utc_now()
                for optuna_trial, artifact in wave_pairs:
                    _write_status(
                        artifact,
                        state="failed",
                        finished_at=finished_at,
                        returncode=-1,
                        objective=None,
                        failure_stage="scheduler",
                        error=(
                            "Trial was not launched because another trial in the same Optuna "
                            "multi_run_lora wave failed materialization and continue_on_failure=false."
                        ),
                    )
                    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                    failures += 1
            if wave_artifacts_for_manifest:
                all_artifacts.extend(wave_artifacts_for_manifest)
                write_manifest_with_variants(config, [build_variant(a) for a in all_artifacts])
            break

        # If every trial in the wave failed materialization there's nothing
        # to launch; advance the counter and try the next wave.
        if not wave_pairs:
            if wave_artifacts_for_manifest:
                all_artifacts.extend(wave_artifacts_for_manifest)
                write_manifest_with_variants(config, [build_variant(a) for a in all_artifacts])
            submitted += this_wave
            continue

        wave_optuna_trials = [pair[0] for pair in wave_pairs]
        wave_artifacts = [pair[1] for pair in wave_pairs]
        all_artifacts.extend(wave_artifacts_for_manifest)
        write_manifest_with_variants(config, [build_variant(a) for a in all_artifacts])

        # 2. Spawn rl-multi-run for this wave.
        command = build_multi_run_command(wave_artifacts, scheduler.shared, shared_dir)
        proc = None
        attempts = 0
        while proc is None:
            attempts += 1
            started = utc_now()
            for artifact in wave_artifacts:
                _reset_multi_run_artifact_runtime(artifact)
                _write_status(artifact, state="running", started_at=started, attempts=attempts, gpu_group=None)
            _mark_inactive_multi_run_dirs_evicted(
                shared_dir,
                [artifact.run_dir for artifact in wave_artifacts],
                reason="Inactive run directory is not part of the current Optuna wave.",
            )
            try:
                proc = subprocess.Popen(command)
            except OSError as exc:
                if attempts <= config.retry_budget:
                    continue
                finished_at = utc_now()
                wave_failures = 0
                for optuna_trial, artifact in zip(wave_optuna_trials, wave_artifacts):
                    _write_launch_failure_status(artifact, exc, finished_at=finished_at)
                    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                    wave_failures += 1
                failures += wave_failures

                for artifact in wave_artifacts:
                    tracker.observe(
                        TrialOutcome(
                            trial_id=artifact.trial.id,
                            label=artifact.trial.label,
                            objective=None,
                        )
                    )

                submitted += this_wave
                if not config.continue_on_failure:
                    stop_after_wave = True
                break

        if proc is None:
            if stop_after_wave:
                break
            continue

        try:
            _poll_wave_for_pruning(
                optuna, proc, wave_artifacts, wave_optuna_trials, metric, poll_interval
            )
        finally:
            proc.wait()

        # 4. Reconcile per-trial state and tell Optuna.
        wave_failures, objectives = _tell_wave_results(
            optuna, study, wave_artifacts, wave_optuna_trials, metric, proc.returncode
        )
        failures += wave_failures

        if wave_failures > 0 and not config.continue_on_failure:
            stop_after_wave = True

        # 5. Fold objectives into the tracker for early stopping + summary.
        for artifact, objective in zip(wave_artifacts, objectives):
            tracker.observe(
                TrialOutcome(
                    trial_id=artifact.trial.id,
                    label=artifact.trial.label,
                    objective=objective,
                )
            )
            if tracker.halted:
                break

        submitted += this_wave
        if stop_after_wave:
            break

    write_manifest_with_variants(config, [build_variant(a) for a in all_artifacts])
    return failures, tracker, all_artifacts


# Re-exported for the controller to update the manifest summary.
def tracker_summary(tracker: TrialOutcomeTracker) -> dict[str, Any]:
    return asdict(tracker.summary())
