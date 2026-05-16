"""Wave-mode Optuna driver for the multi_run_lora scheduler (Phase 7b).

The full multi-run + Optuna stack exercises trainer + inference + N
orchestrators in one ``rl-multi-run`` invocation per wave; here we stub
``validate_target_config`` for the resolved orchestrator config and
``subprocess.Popen`` for the launcher invocation, plus replace the Optuna
study with a controllable stub. The contract under test is:

- One ``rl-multi-run`` invocation per wave, sized to ``max_concurrent_runs``.
- Mid-wave pruning writes ``<run_dir>/control/evicted.txt`` and records
  ``state="pruned"`` in ``status.json`` *before* the orchestrator exits.
- After each wave, ``study.tell`` is called with the right state per trial:
  PRUNED for pruned trials, the recovered objective for completed trials,
  FAIL for trials whose orchestrator exited non-zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomli
import tomli_w

pytest.importorskip("optuna")

import optuna  # noqa: E402

from prime_rl.configs.sweep import SweepConfig  # noqa: E402
from prime_rl.sweep.controller import run_sweep  # noqa: E402


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _stub_validate_target_config(monkeypatch) -> None:
    from prime_rl.sweep import materialize as mat_mod

    def fake_validate(entrypoint, args):
        overrides_path = Path(args[-1])
        overrides = tomli.loads(overrides_path.read_text())
        orch = overrides.get("orchestrator", {})

        class FakeOrch:
            def model_dump(self, *, exclude_none=True, mode="json"):
                return orch

        return SimpleNamespace(orchestrator=FakeOrch())

    monkeypatch.setattr(mat_mod, "validate_target_config", fake_validate)


class _TrialStub:
    """Minimal Optuna ``Trial`` stand-in.

    Captures ``suggest_*`` calls (returns the first option for determinism),
    records ``report`` calls, and lets the test toggle ``should_prune``.
    """

    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []
        self.should_prune_returns = False

    def suggest_categorical(self, name: str, choices):
        v = choices[0]
        self.params[name] = v
        return v

    def suggest_float(self, name: str, low, high, log=False):
        v = low
        self.params[name] = v
        return v

    def suggest_int(self, name: str, low, high, step=1):
        v = low
        self.params[name] = v
        return v

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.should_prune_returns


class _StudyStub:
    """In-memory Optuna ``Study`` stand-in.

    The wave driver only uses ``ask``, ``tell``, and the
    ``optuna.trial.TrialState`` enum (which is the real one). Tracking trials
    in an instance attribute lets the test inspect what got asked / told.
    """

    def __init__(self) -> None:
        self.asked: list[_TrialStub] = []
        self.tells: list[tuple[_TrialStub, Any, Any]] = []
        self._on_ask = None  # optional callback to mark trials as prune-bound

    def ask(self) -> _TrialStub:
        trial = _TrialStub(number=len(self.asked))
        self.asked.append(trial)
        if self._on_ask is not None:
            self._on_ask(trial)
        return trial

    def tell(self, trial: _TrialStub, value=None, state=None) -> None:
        self.tells.append((trial, value, state))


class _FakePopen:
    """Fake ``subprocess.Popen`` that simulates a wave's ``rl-multi-run`` run.

    On construction it parses ``--runs-dir`` from the command and seeds each
    run dir's ``metrics.jsonl`` with a single ``(step=1, reward=...)`` row so
    the wave driver's poll loop has something to ``report`` on its first
    tick. ``poll()`` returns ``None`` for the first call (the poll loop runs
    once) and ``0`` thereafter; on the transition it writes
    ``<run_dir>/control/exit_code`` for every survivor, mirroring what the
    real launcher does in production. The threshold leaves enough live
    ``poll()`` calls for the controller's per-artifact pre-prune liveness
    checks during the first polling pass.

    Non-``rl-multi-run`` invocations (e.g. ``git rev-parse`` for the manifest's
    git metadata) are delegated to the real ``subprocess.Popen`` — patching
    ``multi_run.subprocess.Popen`` patches the stdlib module attribute, so
    every Popen call in the process flows through here while the test runs.
    """

    instances: list["_FakePopen"] = []
    failed_trial_indices: set[int] = set()
    _real_popen = None  # populated by _install_fake_optuna_runtime

    def __new__(cls, command, **kwargs):
        if not command or command[0] != "rl-multi-run":
            assert cls._real_popen is not None  # set up by the test fixture
            return cls._real_popen(command, **kwargs)
        instance = super().__new__(cls)
        return instance

    def __init__(self, command, **kwargs) -> None:
        if not command or command[0] != "rl-multi-run":
            return  # __new__ delegated to real Popen; nothing to init
        _FakePopen.instances.append(self)
        self.command = list(command)
        idx = command.index("--runs-dir")
        self.run_dirs = [Path(p) for p in command[idx + 1].split(":") if p]
        self._poll_count = 0
        self.returncode: int | None = None

        # Seed metrics so the wave driver has something to report on the
        # first poll tick. Reward 0.5 is arbitrary; the prune decision in
        # tests comes from the trial stub's should_prune flag, not the value.
        for run_dir in self.run_dirs:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 1, "reward": 0.5}) + "\n"
            )

    def poll(self) -> int | None:
        self._poll_count += 1
        if self._poll_count < len(self.run_dirs) + 3:
            return None
        # After one full live polling pass, simulate the launcher writing
        # exit_codes and exiting cleanly. Pruned trials had their orch exit
        # with non-zero in production; the controller pre-marked them so
        # reconcile won't read the exit_code anyway, but we still write 1
        # for diagnostics.
        for run_dir in self.run_dirs:
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            trial_index = int(run_dir.name.removeprefix("run_").split("-", 1)[0])
            should_fail = status.get("state") == "pruned" or trial_index in self.failed_trial_indices
            code = "1\n" if should_fail else "0\n"
            (run_dir / "control" / "exit_code").write_text(code)
        self.returncode = 0
        return 0

    def wait(self) -> int:
        if self.returncode is None:
            self.poll()
        return self.returncode  # type: ignore[return-value]


def _install_fake_optuna_runtime(monkeypatch, study: _StudyStub) -> None:
    """Replace the create-study helper, Popen, and time.sleep so the wave
    driver runs synchronously against the fake process."""
    import subprocess as real_subprocess

    from prime_rl.sweep import multi_run as multi_run_mod

    _FakePopen._real_popen = real_subprocess.Popen
    monkeypatch.setattr(multi_run_mod, "_create_study", lambda *a, **kw: study)
    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(multi_run_mod.time, "sleep", lambda *_a, **_kw: None)
    _FakePopen.instances.clear()
    _FakePopen.failed_trial_indices = set()


def test_multi_run_optuna_wave_prunes_one_trial_and_completes_others(
    tmp_path: Path, monkeypatch
) -> None:
    """One wave of 3 trials: trial #1 is configured to prune; the other
    two run to completion and report their objective."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()

    # Mark the second asked trial as prune-bound the moment Optuna asks for it,
    # so when the wave driver calls should_prune() after report(), we prune it.
    def _on_ask(trial: _TrialStub) -> None:
        if trial.number == 1:
            trial.should_prune_returns = True

    study._on_ask = _on_ask

    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-prune",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 3,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 3, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    # One wave → one Popen.
    assert len(_FakePopen.instances) == 1
    invocation = _FakePopen.instances[0]
    assert invocation.command[0] == "rl-multi-run"
    assert "--runs-dir" in invocation.command

    # Three trials asked, three told.
    assert len(study.asked) == 3
    assert len(study.tells) == 3

    # Trial #1 told as PRUNED, others told a numeric value.
    state_by_number = {trial.number: (value, state) for trial, value, state in study.tells}
    pruned_value, pruned_state = state_by_number[1]
    assert pruned_state == optuna.trial.TrialState.PRUNED
    assert pruned_value is None

    for ok_number in (0, 2):
        value, state = state_by_number[ok_number]
        assert state is None  # natural completion: tell(value) with no state kwarg
        assert value == 0.5

    # The pruned trial has evicted.txt + status.json state="pruned".
    pruned_run_dir = invocation.run_dirs[1]
    assert (pruned_run_dir / "control" / "evicted.txt").exists()
    pruned_status = json.loads((pruned_run_dir / "status.json").read_text())
    assert pruned_status["state"] == "pruned"
    assert pruned_status["returncode"] == 1
    assert "finished_at" in pruned_status
    assert pruned_status["pruned_at_step"] == 1
    assert pruned_status["pruned_reason"].startswith("optuna prune")

    # Manifest summary's ``completed`` count tracks trials with a recorded
    # objective; pruned trials report None and are not counted.
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 2

    # And the per-trial status.json files record one pruned + two completed.
    states_by_index: dict[int, str] = {}
    for variant in manifest["variants"]:
        trial_idx = int(variant["id"].split("-", 1)[0])
        states_by_index[trial_idx] = json.loads(Path(variant["status_path"]).read_text())["state"]
    assert states_by_index == {0: "completed", 1: "pruned", 2: "completed"}


def test_multi_run_optuna_does_not_prune_after_wave_exits(tmp_path: Path, monkeypatch) -> None:
    """If the wave exits after the poll-loop guard, final reconciliation wins."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    study._on_ask = lambda trial: setattr(trial, "should_prune_returns", True)
    _install_fake_optuna_runtime(monkeypatch, study)

    from prime_rl.sweep import multi_run as multi_run_mod

    class _ExitBeforePrunePopen(_FakePopen):
        def poll(self) -> int | None:
            self._poll_count += 1
            if self._poll_count == 1:
                return None
            for run_dir in self.run_dirs:
                (run_dir / "control").mkdir(parents=True, exist_ok=True)
                (run_dir / "control" / "exit_code").write_text("0\n")
            self.returncode = 0
            return 0

    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", _ExitBeforePrunePopen)

    config = SweepConfig(
        name="lora-optuna-exit-before-prune",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "pruner": {"type": "median", "n_startup_trials": 0, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    assert len(study.tells) == 1
    trial, value, state = study.tells[0]
    assert trial.number == 0
    assert value == 0.5
    assert state is None

    run_dir = _FakePopen.instances[0].run_dirs[0]
    assert not (run_dir / "control" / "evicted.txt").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["objective"] == 0.5


def test_multi_run_optuna_does_not_prune_run_with_recorded_exit_code(
    tmp_path: Path, monkeypatch
) -> None:
    """A completed orchestrator can finish before sibling runs keep the wave alive."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    study._on_ask = lambda trial: setattr(trial, "should_prune_returns", True)
    _install_fake_optuna_runtime(monkeypatch, study)

    from prime_rl.sweep import multi_run as multi_run_mod

    class _CompletedRunOpenWavePopen(_FakePopen):
        def __init__(self, command, **kwargs) -> None:
            super().__init__(command, **kwargs)
            for run_dir in self.run_dirs:
                (run_dir / "control").mkdir(parents=True, exist_ok=True)
                (run_dir / "control" / "exit_code").write_text("0\n")

    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", _CompletedRunOpenWavePopen)

    config = SweepConfig(
        name="lora-optuna-completed-run-open-wave",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "pruner": {"type": "median", "n_startup_trials": 0, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    assert len(study.tells) == 1
    trial, value, state = study.tells[0]
    assert trial.number == 0
    assert trial.reports == []
    assert value == 0.5
    assert state is None

    run_dir = _FakePopen.instances[0].run_dirs[0]
    assert not (run_dir / "control" / "evicted.txt").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["objective"] == 0.5


def test_multi_run_optuna_does_not_prune_when_exit_code_appears_after_report(
    tmp_path: Path, monkeypatch
) -> None:
    """A run can finish after the initial exit-code check but before the
    pruning decision. Re-read exit_code so final reconciliation wins."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    class _ExitCodeAfterReportTrial(_TrialStub):
        def report(self, value: float, step: int) -> None:
            super().report(value, step)
            run_dir = _FakePopen.instances[0].run_dirs[self.number]
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            (run_dir / "control" / "exit_code").write_text("0\n")

    class _RaceStudy(_StudyStub):
        def ask(self) -> _TrialStub:
            trial = _ExitCodeAfterReportTrial(number=len(self.asked))
            trial.should_prune_returns = True
            self.asked.append(trial)
            return trial

    study = _RaceStudy()
    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-exit-code-after-report",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "pruner": {"type": "median", "n_startup_trials": 0, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    assert len(study.tells) == 1
    trial, value, state = study.tells[0]
    assert trial.number == 0
    assert trial.reports == [(0.5, 1)]
    assert value == 0.5
    assert state is None

    run_dir = _FakePopen.instances[0].run_dirs[0]
    assert not (run_dir / "control" / "evicted.txt").exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["objective"] == 0.5


def test_multi_run_optuna_runs_in_waves(tmp_path: Path, monkeypatch) -> None:
    """``num_trials=4, max_concurrent_runs=2`` produces two waves of two
    trials each; each wave is one Popen invocation."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-waves",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 4, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    assert len(_FakePopen.instances) == 2
    for invocation in _FakePopen.instances:
        runs_idx = invocation.command.index("--runs-dir")
        run_dirs = invocation.command[runs_idx + 1].split(":")
        assert len(run_dirs) == 2
    for run_dir in _FakePopen.instances[0].run_dirs:
        evicted = run_dir / "control" / "evicted.txt"
        assert evicted.exists()
        assert "current Optuna wave" in evicted.read_text()
    for run_dir in _FakePopen.instances[1].run_dirs:
        assert not (run_dir / "control" / "evicted.txt").exists()

    assert len(study.asked) == 4
    assert len(study.tells) == 4
    # Every tell carries the wave's objective value (0.5 by fake convention).
    for _trial, value, state in study.tells:
        assert state is None
        assert value == 0.5


def test_multi_run_optuna_writes_summary_before_halting_on_wave_failure(
    tmp_path: Path, monkeypatch
) -> None:
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)
    _FakePopen.failed_trial_indices = {1}

    config = SweepConfig(
        name="lora-optuna-wave-failure",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 4, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        retry_budget=0,
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert len(_FakePopen.instances) == 1
    assert len(study.asked) == 2
    assert len(study.tells) == 2

    tell_by_number = {trial.number: (value, state) for trial, value, state in study.tells}
    assert tell_by_number[0] == (0.5, None)
    assert tell_by_number[1] == (None, optuna.trial.TrialState.FAIL)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 1
    assert manifest["summary"]["best_value"] == 0.5
    states_by_index = {
        int(variant["id"].split("-", 1)[0]): json.loads(Path(variant["status_path"]).read_text())["state"]
        for variant in manifest["variants"]
    }
    assert states_by_index == {0: "completed", 1: "failed"}


def test_multi_run_optuna_marks_wave_failed_on_launcher_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    from prime_rl.sweep import multi_run as multi_run_mod

    def fake_popen(command, **kwargs):
        if command and command[0] == "rl-multi-run":
            raise FileNotFoundError("missing rl-multi-run")
        assert _FakePopen._real_popen is not None
        return _FakePopen._real_popen(command, **kwargs)

    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", fake_popen)

    config = SweepConfig(
        name="lora-optuna-launch-failure",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 2, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert len(_FakePopen.instances) == 0
    assert len(study.asked) == 2
    assert [state for _trial, _value, state in study.tells] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.FAIL,
    ]

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 0
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["returncode"] == -1
        assert status["failure_stage"] == "launch"
        assert "FileNotFoundError" in status["error"]


def test_multi_run_optuna_retries_launcher_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    from prime_rl.sweep import multi_run as multi_run_mod

    launch_attempts = 0

    def fake_popen(command, **kwargs):
        nonlocal launch_attempts
        if command and command[0] == "rl-multi-run":
            launch_attempts += 1
            if launch_attempts == 1:
                raise FileNotFoundError("temporary rl-multi-run miss")
            return _FakePopen(command, **kwargs)
        assert _FakePopen._real_popen is not None
        return _FakePopen._real_popen(command, **kwargs)

    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", fake_popen)

    config = SweepConfig(
        name="lora-optuna-launch-retry",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 2, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        retry_budget=1,
        wandb=None,
    )

    run_sweep(config)

    assert launch_attempts == 2
    assert len(_FakePopen.instances) == 1
    assert len(study.asked) == 2
    assert len(study.tells) == 2
    for _trial, value, state in study.tells:
        assert value == 0.5
        assert state is None

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 2
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "completed"
        assert status["returncode"] == 0
        assert status["attempts"] == 2
        assert "failure_stage" not in status


def test_multi_run_optuna_marks_missing_objectives_failed(
    tmp_path: Path, monkeypatch
) -> None:
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})
    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-missing-objective",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "pruner": {"type": "median", "n_startup_trials": 1, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"orchestrator.optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "missing", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)
    assert exc_info.value.code == 1

    assert len(study.tells) == 2
    assert [state for _trial, _value, state in study.tells] == [optuna.trial.TrialState.FAIL] * 2

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 0
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["returncode"] == 0
        assert status["failure_stage"] == "objective"
        assert status["objective"] is None


def test_multi_run_optuna_does_not_launch_partial_wave_after_materialization_failure(
    tmp_path: Path, monkeypatch
) -> None:
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    from prime_rl.sweep import multi_run as multi_run_mod

    original_materialize = multi_run_mod.materialize_multi_run_trial
    calls = {"n": 0}

    def flaky_materialize(config, trial, scheduler):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("fake materialization failure")
        return original_materialize(config, trial, scheduler)

    monkeypatch.setattr(multi_run_mod, "materialize_multi_run_trial", flaky_materialize)

    config = SweepConfig(
        name="lora-optuna-materialization-failure",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 3,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 3, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert len(_FakePopen.instances) == 0
    assert len(study.asked) == 2

    tell_by_number = {trial.number: state for trial, _value, state in study.tells}
    assert tell_by_number == {
        0: optuna.trial.TrialState.FAIL,
        1: optuna.trial.TrialState.FAIL,
    }

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 0
    assert len(manifest["variants"]) == 2
    statuses_by_index = {
        int(variant["id"].split("-", 1)[0]): json.loads(Path(variant["status_path"]).read_text())
        for variant in manifest["variants"]
    }
    assert statuses_by_index[0]["state"] == "failed"
    assert statuses_by_index[0]["returncode"] == -1
    assert statuses_by_index[0]["failure_stage"] == "scheduler"
    assert "not launched" in statuses_by_index[0]["error"]
    assert statuses_by_index[1]["state"] == "failed"
    assert statuses_by_index[1]["returncode"] == -1
    assert statuses_by_index[1]["failure_stage"] == "materialization"
    assert "fake materialization failure" in statuses_by_index[1]["error"]
