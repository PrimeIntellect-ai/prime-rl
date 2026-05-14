import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.materialize import Trial, materialize_trial
from prime_rl.sweep.schedulers import (
    run_trials_locally,
    submit_trials_to_multi_run_lora,
    submit_trials_to_slurm,
)


def _materialize(tmp_path: Path, count: int) -> tuple[SweepConfig, list]:
    base_path = tmp_path / "base.toml"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    with open(base_path, "wb") as f:
        tomli_w.dump({"data": {"type": "fake"}, "max_steps": 1}, f)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    artifacts = []
    for idx in range(count):
        trial = Trial(id=f"{idx:04d}-deadbeef", label=f"t{idx}", parameters={"optim.lr": 1e-5})
        artifacts.append(materialize_trial(config, trial))
    return config, artifacts


def test_sequential_run_pins_cuda_visible_devices(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=2)

    captured_envs: list[str | None] = []

    def fake_run(_command, env=None):
        captured_envs.append(env.get("CUDA_VISIBLE_DEVICES") if env is not None else None)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(artifacts, max_parallel=1, gpu_groups=[[2, 3]])

    assert failures == 0
    assert captured_envs == ["2,3", "2,3"]
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "completed"
        assert status["gpu_group"] == [2, 3]


def test_parallel_run_assigns_disjoint_groups_per_worker(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=6)

    inflight_lock = threading.Lock()
    inflight: dict[str, str] = {}
    seen_overlap = False

    def fake_run(_command, env=None):
        nonlocal seen_overlap
        devices = env.get("CUDA_VISIBLE_DEVICES") if env is not None else None
        with inflight_lock:
            if devices in inflight.values():
                seen_overlap = True
            inflight[id(_command)] = devices
        time.sleep(0.01)
        with inflight_lock:
            inflight.pop(id(_command), None)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(
        artifacts,
        max_parallel=3,
        gpu_groups=[[0], [1], [2]],
    )

    assert failures == 0
    assert not seen_overlap
    devices_seen = {json.loads(a.status_path.read_text())["gpu_group"][0] for a in artifacts}
    assert devices_seen == {0, 1, 2}


def test_parallel_run_requires_enough_groups(tmp_path: Path) -> None:
    _, artifacts = _materialize(tmp_path, count=2)

    try:
        run_trials_locally(artifacts, max_parallel=4, gpu_groups=[[0]])
    except ValueError as exc:
        assert "max_parallel=4" in str(exc)
    else:
        raise AssertionError("Expected ValueError when gpu_groups is too short")


def test_parallel_run_records_failures_and_continues(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=4)

    def fake_run(_command, env=None):
        devices = env.get("CUDA_VISIBLE_DEVICES") if env is not None else None
        returncode = 0 if devices == "0" else 1
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(
        artifacts,
        max_parallel=2,
        gpu_groups=[[0], [1]],
        continue_on_failure=True,
        retry_budget=0,
    )

    assert failures > 0


def test_local_run_marks_launch_oserror_failed(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)

    def fake_run(_command, env=None):
        raise FileNotFoundError("missing launcher")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(artifacts, retry_budget=0)

    assert failures == 1
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -1
    assert status["failure_stage"] == "launch"
    assert "FileNotFoundError" in status["error"]


def test_local_run_retries_launch_oserror(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)
    attempts = 0

    def fake_run(_command, env=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("temporary launcher miss")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(artifacts, retry_budget=1)

    assert failures == 0
    assert attempts == 2
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0
    assert status["attempts"] == 2
    assert "failure_stage" not in status


def test_slurm_submission_marks_launch_oserror_failed(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)

    def fake_run(_command):
        raise FileNotFoundError("missing sbatch wrapper")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_slurm(artifacts, retry_budget=0)

    assert failures == 1
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -1
    assert status["failure_stage"] == "launch"


def test_slurm_submission_retries_launch_oserror(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)
    attempts = 0

    def fake_run(_command):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("temporary sbatch wrapper miss")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_slurm(artifacts, retry_budget=1)

    assert failures == 0
    assert attempts == 2
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "submitted"
    assert status["returncode"] == 0
    assert status["attempts"] == 2
    assert "failure_stage" not in status


def test_slurm_sync_dispatches_dry_run_then_sbatch_wait(tmp_path: Path, monkeypatch) -> None:
    """Synchronous SLURM: first call generates the script via --dry-run, second
    call submits with sbatch --wait and blocks. Trial ends up state=completed."""
    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)
    # _materialize uses entrypoint="sft" — the dry-run writes sft.sbatch, not
    # rl.sbatch. This test locks in that the runner derives the filename from
    # the trial's command rather than hard-coding the rl entrypoint.
    script_path = artifact.run_dir / "sft.sbatch"

    calls: list[list[str]] = []

    def fake_run(command, env=None):
        calls.append(list(command))
        if "--dry-run" in command:
            script_path.write_text("#!/usr/bin/env bash\necho hello\n")
            return SimpleNamespace(returncode=0)
        assert command[0] == "sbatch" and "--wait" in command
        assert str(script_path) in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_slurm(artifacts, retry_budget=0, synchronous=True)

    assert failures == 0
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0
    # Order matters: dry-run first, sbatch --wait second.
    assert "--dry-run" in calls[0]
    assert calls[1][0] == "sbatch" and "--wait" in calls[1]


def test_slurm_sync_fires_on_trial_complete_callback(tmp_path: Path, monkeypatch) -> None:
    """Synchronous SLURM with a callback halts further submissions when the
    callback returns True (early stopping path)."""
    _, artifacts = _materialize(tmp_path, count=3)
    for artifact in artifacts:
        artifact.run_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(command, env=None):
        if "--dry-run" in command:
            entrypoint = command[2]
            for arg in command:
                p = Path(arg)
                if p.name == "overrides.toml":
                    run_dir = p.parent / "run"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / f"{entrypoint}.sbatch").write_text("#!/usr/bin/env bash\n")
                    break
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    seen: list[str] = []

    def on_trial_complete(artifact, returncode):
        seen.append(artifact.trial.id)
        return len(seen) >= 1  # halt after first trial

    failures = submit_trials_to_slurm(
        artifacts, retry_budget=0, synchronous=True, on_trial_complete=on_trial_complete
    )

    assert failures == 0
    assert len(seen) == 1
    # The other two trials must still be pending (never reached).
    later = json.loads(artifacts[1].status_path.read_text())
    assert later["state"] == "pending"


def _setup_slurm_sync_fakes(monkeypatch, *, jobid: str, terminal_state: str = "COMPLETED") -> dict:
    """Patch every external SLURM call used by ``_run_trial_with_pruning_slurm_sync``.

    Returns a state dict the test can poke to advance the fake job lifecycle.
    ``_query_squeue_state`` returns ``state['squeue']`` (start "RUNNING"; tests
    flip to ``None`` to signal job exit), ``_query_sacct_state`` returns the
    terminal state, ``_submit_sbatch_parsable`` returns the provided ``jobid``.
    """
    state = {"squeue": "RUNNING", "scancelled": False}

    def fake_render(artifact, env):
        entrypoint = artifact.command[2]
        (artifact.run_dir / f"{entrypoint}.sbatch").write_text("#!/usr/bin/env bash\n")
        return 0

    def fake_submit(_script, _env):
        return 0, jobid

    def fake_squeue(_jid):
        return state["squeue"]

    def fake_sacct(_jid):
        return terminal_state

    def fake_scancel(_jid, **_kwargs):
        state["scancelled"] = True
        state["squeue"] = None
        return True

    monkeypatch.setattr("prime_rl.sweep.schedulers._render_sbatch_script", fake_render)
    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit)
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_squeue_state", fake_squeue)
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", fake_sacct)
    monkeypatch.setattr("prime_rl.sweep.schedulers._scancel_job", fake_scancel)
    monkeypatch.setattr("prime_rl.sweep.schedulers.time.sleep", lambda _s: None)
    return state


def test_slurm_sync_pruning_writes_jobid_and_completes(tmp_path: Path, monkeypatch) -> None:
    """Happy path: trial completes, controller records objective from
    metrics.jsonl, status reflects ``slurm_job_id`` + state=completed."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="42")
    # Job already exited before the controller polled (squeue returns None).
    fake_state["squeue"] = None

    # The fake submitter writes the metrics file the controller will read —
    # i.e. the real trial wrote metrics.jsonl during its run.
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 10, "val/loss": 0.5}) + "\n")
        return 0, "42"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)

    class _Trial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return True

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "completed"
    assert outcome.returncode == 0
    assert outcome.objective == 0.5
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["slurm_job_id"] == "42"


def test_slurm_sync_pruning_scancels_on_prune(tmp_path: Path, monkeypatch) -> None:
    """Prune path: should_prune() triggers ``scancel``, status=pruned, and
    the pruned step/value are recorded from the most recent metrics row."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="99", terminal_state="CANCELLED")
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 5, "val/loss": 1.7}) + "\n")
        return 0, "99"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)

    class _Trial:
        def __init__(self):
            self.reported = []

        def report(self, value, step):
            self.reported.append((step, value))

        def should_prune(self):
            return True

    trial = _Trial()
    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, trial, metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "pruned"
    assert outcome.pruned_at_step == 5
    assert outcome.pruned_value == 1.7
    assert fake_state["scancelled"] is True
    assert trial.reported == [(5, 1.7)]
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pruned"
    assert status["slurm_job_id"] == "99"


def test_slurm_sync_waits_for_sacct_before_failing(tmp_path: Path, monkeypatch) -> None:
    """sacct can lag after squeue clears; the controller must poll a bit
    before declaring the trial failed, otherwise a COMPLETED job is reported
    to Optuna as FAIL whenever the accounting db is congested."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="77", terminal_state="COMPLETED")
    fake_state["squeue"] = None
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 1, "val/loss": 0.25}) + "\n")
        return 0, "77"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)

    sacct_responses = [None, None, "COMPLETED"]

    def lagging_sacct(_jid):
        return sacct_responses.pop(0) if sacct_responses else "COMPLETED"

    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lagging_sacct)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "completed"
    assert outcome.returncode == 0
    assert outcome.objective == 0.25
    # All three lagged responses were consumed (two Nones + the terminal).
    assert sacct_responses == []


def test_slurm_sync_falls_back_to_scontrol_when_sacct_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    """When the cluster has sacct accounting disabled, scontrol's in-memory
    cache is the next signal. A COMPLETED+ExitCode=0:0 reading must mark
    the trial completed and read the objective from metrics.jsonl."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="40")
    fake_state["squeue"] = None
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 19, "val/loss": 0.74}) + "\n")
        return 0, "40"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lambda _jid: None)
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._wait_for_sacct_terminal_state",
        lambda _jid, **_kwargs: None,
    )
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._query_scontrol_outcome",
        lambda _jid: ("COMPLETED", 0),
    )

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.74


def test_slurm_sync_falls_back_to_metrics_when_slurm_state_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    """When both sacct and scontrol are silent (accounting disabled AND
    scontrol cache aged out), the controller must trust metrics.jsonl: a
    finite objective is strong evidence the trial ran to completion."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="40")
    fake_state["squeue"] = None
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 19, "val/loss": 0.74}) + "\n")
        return 0, "40"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lambda _jid: None)
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._wait_for_sacct_terminal_state",
        lambda _jid, **_kwargs: None,
    )
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_scontrol_outcome", lambda _jid: None)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.74
    status = json.loads(artifact.status_path.read_text())
    assert status["slurm_terminal_state"] == "unknown"


def test_slurm_sync_unknown_state_with_no_objective_is_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """If SLURM state is unknown and metrics.jsonl never recorded a finite
    objective, the trial must be marked failed — no objective means we have
    no evidence the trial actually completed its work."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="40")
    fake_state["squeue"] = None
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lambda _jid: None)
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._wait_for_sacct_terminal_state",
        lambda _jid, **_kwargs: None,
    )
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_scontrol_outcome", lambda _jid: None)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "failed"
    status = json.loads(artifact.status_path.read_text())
    assert status["slurm_terminal_state"] == "unknown"


def test_slurm_sync_scontrol_nonzero_exit_is_failure(tmp_path: Path, monkeypatch) -> None:
    """If scontrol reports JobState=COMPLETED but the process exit code is
    non-zero, treat as failed — SLURM considers the allocation complete but
    the trial itself errored."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="40")
    fake_state["squeue"] = None
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lambda _jid: None)
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._wait_for_sacct_terminal_state",
        lambda _jid, **_kwargs: None,
    )
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._query_scontrol_outcome",
        lambda _jid: ("COMPLETED", 1),
    )

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "failed"


def test_slurm_sync_retry_wrapper_does_not_retry_unsafe_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    """The retry wrapper must short-circuit on unsafe_to_continue outcomes:
    re-running would submit a second SLURM job for the same trial while the
    first may still be alive."""
    from prime_rl.sweep import optuna_loop as opt
    from prime_rl.sweep.optuna_loop import _PollingOutcome, _run_trial_with_pruning_slurm_sync_and_retries

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]

    attempts: list[int] = []

    def fake_runner(_artifact, _trial, _metric, _poll_interval, attempt: int = 1):
        attempts.append(attempt)
        return _PollingOutcome(
            state="failed",
            returncode=-1,
            objective=None,
            unsafe_to_continue=True,
        )

    monkeypatch.setattr(opt, "_run_trial_with_pruning_slurm_sync", fake_runner)

    outcome = _run_trial_with_pruning_slurm_sync_and_retries(
        artifact,
        optuna_trial=object(),
        metric="val/loss",
        poll_interval=0.01,
        retry_budget=5,  # large budget — must still short-circuit
    )

    assert outcome.unsafe_to_continue is True
    assert attempts == [1]


def test_slurm_sync_marks_failed_when_scancel_unconfirmed(tmp_path: Path, monkeypatch) -> None:
    """If _scancel_job returns False (job did not leave the queue), the trial
    must be recorded as failed — never as pruned — so the controller does not
    advance Optuna while the SLURM allocation is still active."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="123")
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 1, "val/loss": 0.5}) + "\n")
        return 0, "123"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)
    # scancel returns False — couldn't confirm the job left the queue.
    monkeypatch.setattr("prime_rl.sweep.schedulers._scancel_job", lambda _jid, **_kwargs: False)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return True

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "failed"
    assert outcome.returncode == -1
    # unsafe_to_continue forces the outer loop to halt regardless of
    # continue_on_failure — the SLURM job may still be alive.
    assert outcome.unsafe_to_continue is True
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "failed"
    assert status["failure_stage"] == "scancel"
    assert "123" in status["error"]
    # Critically: state is NOT "pruned" — the controller cannot assume the
    # job is gone, so advancing the Optuna study is unsafe.
    assert fake_state["squeue"] == "RUNNING"  # job state untouched by fake


def test_slurm_sync_hard_fails_after_repeated_squeue_errors(tmp_path: Path, monkeypatch) -> None:
    """When ``squeue`` returns errors back-to-back, the polling loop must NOT
    treat the silence as "job is done" — it must hard-fail the trial so the
    controller does not advance the Optuna study while the SLURM job may
    still be running."""
    from prime_rl.sweep.schedulers import SqueueQueryError, _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    _setup_slurm_sync_fakes(monkeypatch, jobid="55")

    def always_failing_squeue(_jid):
        raise SqueueQueryError("slurmctld unreachable")

    monkeypatch.setattr("prime_rl.sweep.schedulers._query_squeue_state", always_failing_squeue)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "failed"
    assert outcome.unsafe_to_continue is True
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "failed"
    assert status["failure_stage"] == "squeue"


def test_slurm_sync_tolerates_transient_squeue_failure(tmp_path: Path, monkeypatch) -> None:
    """A single transient squeue failure must not break the polling loop —
    the controller should retry, observe the running job, and continue until
    the job legitimately leaves the queue."""
    from prime_rl.sweep.schedulers import SqueueQueryError, _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    _setup_slurm_sync_fakes(monkeypatch, jobid="66", terminal_state="COMPLETED")
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 2, "val/loss": 0.4}) + "\n")
        return 0, "66"

    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics)

    # First call: transient failure. Second: still running. Third: gone.
    squeue_script = [SqueueQueryError("temp glitch"), "RUNNING", None]

    def flaky_squeue(_jid):
        response = squeue_script.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("prime_rl.sweep.schedulers._query_squeue_state", flaky_squeue)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.4
    assert squeue_script == []


def test_slurm_sync_pruning_marks_failed_on_submit_error(tmp_path: Path, monkeypatch) -> None:
    """If sbatch --parsable returns no job id, the trial is marked failed
    with failure_stage=submission so resume can see what went wrong."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    def fake_render(a, _env):
        entrypoint = a.command[2]
        (a.run_dir / f"{entrypoint}.sbatch").write_text("#!/usr/bin/env bash\n")
        return 0

    def fake_submit(_script, _env):
        return 1, None

    monkeypatch.setattr("prime_rl.sweep.schedulers._render_sbatch_script", fake_render)
    monkeypatch.setattr("prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit)

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "failed"
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "failed"
    assert status["failure_stage"] == "submission"


def test_multi_run_lora_marks_all_artifacts_failed_on_launcher_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    _, artifacts = _materialize(tmp_path, count=2)

    def fake_run(_command):
        raise FileNotFoundError("missing rl-multi-run")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_multi_run_lora(
        artifacts,
        shared_paths=[tmp_path / "shared.toml"],
        shared_dir=tmp_path / "shared",
        retry_budget=0,
    )

    assert failures == 2
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "failed"
        assert status["returncode"] == -1
        assert status["failure_stage"] == "launch"


def test_multi_run_lora_retries_launcher_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    _, artifacts = _materialize(tmp_path, count=2)
    attempts = 0

    def fake_run(command):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("temporary rl-multi-run miss")
        runs_idx = command.index("--runs-dir")
        for raw_run_dir in command[runs_idx + 1].split(":"):
            control_dir = Path(raw_run_dir) / "control"
            control_dir.mkdir(parents=True, exist_ok=True)
            (control_dir / "exit_code").write_text("0\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_multi_run_lora(
        artifacts,
        shared_paths=[tmp_path / "shared.toml"],
        shared_dir=tmp_path / "shared",
        retry_budget=1,
    )

    assert failures == 0
    assert attempts == 2
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "completed"
        assert status["returncode"] == 0
        assert status["attempts"] == 2
        assert "failure_stage" not in status


def test_slurm_sync_cancelled_with_training_complete_sentinel_is_completed(
    tmp_path: Path, monkeypatch
) -> None:
    """The rl entrypoint's sbatch template has trainer rank 0 scancel its
    own job after a clean exit, leaving SLURM state=CANCELLED for what was
    actually a successful trial. Presence of ``.training_complete`` in the
    run_dir proves this was the expected self-teardown — the trial must be
    recorded as completed and the objective from metrics.jsonl preserved."""
    from prime_rl.sweep.schedulers import (
        TRAINING_COMPLETE_SENTINEL,
        _run_trial_with_pruning_slurm_sync,
    )

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="40")
    fake_state["squeue"] = None
    metrics_path = artifact.run_dir / "metrics.jsonl"
    sentinel_path = artifact.run_dir / TRAINING_COMPLETE_SENTINEL

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 19, "val/loss": 0.74}) + "\n")
        sentinel_path.write_text("")
        return 0, "40"

    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics
    )
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lambda _jid: None)
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._wait_for_sacct_terminal_state",
        lambda _jid, **_kwargs: None,
    )
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._query_scontrol_outcome",
        lambda _jid: ("CANCELLED", 0),
    )

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.74


def test_slurm_sync_cancelled_without_sentinel_is_failed(
    tmp_path: Path, monkeypatch
) -> None:
    """A CANCELLED state without the self-teardown sentinel is a real
    failure (operator scancel, pre-empt-as-cancel, etc.) and must NOT be
    silently promoted to completed just because metrics.jsonl happens to
    have a row in it."""
    from prime_rl.sweep.schedulers import _run_trial_with_pruning_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    fake_state = _setup_slurm_sync_fakes(monkeypatch, jobid="41")
    fake_state["squeue"] = None
    metrics_path = artifact.run_dir / "metrics.jsonl"

    def fake_submit_after_metrics(_script, _env):
        metrics_path.write_text(json.dumps({"step": 7, "val/loss": 0.42}) + "\n")
        return 0, "41"

    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._submit_sbatch_parsable", fake_submit_after_metrics
    )
    monkeypatch.setattr("prime_rl.sweep.schedulers._query_sacct_state", lambda _jid: None)
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._wait_for_sacct_terminal_state",
        lambda _jid, **_kwargs: None,
    )
    monkeypatch.setattr(
        "prime_rl.sweep.schedulers._query_scontrol_outcome",
        lambda _jid: ("CANCELLED", 0),
    )

    class _Trial:
        def report(self, *_):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_slurm_sync(
        artifact, _Trial(), metric="val/loss", poll_interval=0.01
    )

    assert outcome.state == "failed"
    assert outcome.objective is None


def test_slurm_sync_sbatch_wait_nonzero_with_sentinel_is_completed(
    tmp_path: Path, monkeypatch
) -> None:
    """No-pruner SLURM-sync uses ``sbatch --wait``, which returns non-zero
    when the trainer scancels its own job. The sentinel lets the runner
    distinguish this expected self-teardown from a real job failure."""
    from prime_rl.sweep.schedulers import TRAINING_COMPLETE_SENTINEL, _run_with_retries_slurm_sync

    _, artifacts = _materialize(tmp_path, count=1)
    artifact = artifacts[0]
    artifact.run_dir.mkdir(parents=True, exist_ok=True)
    script_path = artifact.run_dir / "sft.sbatch"

    def fake_run(command, env=None):
        if "--dry-run" in command:
            script_path.write_text("#!/usr/bin/env bash\n")
            return SimpleNamespace(returncode=0)
        # Simulate trainer-rank-0 self-teardown: write the sentinel, then
        # sbatch --wait surfaces the CANCELLED exit as non-zero.
        (artifact.run_dir / TRAINING_COMPLETE_SENTINEL).write_text("")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    returncode = _run_with_retries_slurm_sync(artifact, retry_budget=0)

    assert returncode == 0
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0
