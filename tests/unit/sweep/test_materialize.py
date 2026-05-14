import json
import shlex
from pathlib import Path

import pytest
import tomli
import tomli_w
from pydantic import ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.materialize import Trial, materialize_trial, record_trial_objective


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


def test_materialize_trial_writes_artifacts(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        name="unit-sweep",
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
    )
    trial = Trial(id="0000", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)

    overrides = read_toml(artifact.overrides_path)
    assert overrides["output_dir"] == (tmp_path / "study" / "trials" / "0000" / "run").as_posix()
    assert overrides["optim"]["lr"] == 1e-5
    assert overrides["wandb"]["group"] == "unit-sweep"
    assert overrides["wandb"]["name"] == "lr_1e-5"

    resolved = read_toml(artifact.resolved_path)
    assert resolved["output_dir"] == overrides["output_dir"]
    assert resolved["optim"]["lr"] == 1e-5

    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pending"
    assert artifact.command == [
        "uv",
        "run",
        "sft",
        "@",
        base_path.as_posix(),
        "@",
        artifact.overrides_path.as_posix(),
    ]
    assert artifact.command_path.read_text().strip() == shlex.join(artifact.command)


def test_materialize_trial_writes_shell_safe_command(tmp_path: Path) -> None:
    base_path = tmp_path / "base config.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study with spaces",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)

    assert shlex.split(artifact.command_path.read_text().strip()) == artifact.command


def test_record_trial_objective_coerces_non_finite_to_missing(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"state": "completed", "objective": 0.1}))

    record_trial_objective(status_path, float("nan"))

    status = json.loads(status_path.read_text())
    assert status["objective"] is None


def test_materialize_trial_rejects_bad_target_path(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"does.not.exist": {"values": [1]}},
        wandb=None,
    )
    trial = Trial(id="0000", label="bad", parameters={"does.not.exist": 1})

    try:
        materialize_trial(config, trial)
    except (ConfigFileError, ValidationError, SystemExit):
        pass
    else:
        raise AssertionError("Expected target config validation to fail")


def test_materialize_trial_rejects_bool_choice_for_numeric_target(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [True]}},
        wandb=None,
    )
    trial = Trial(id="0000", label="bool_lr", parameters={"optim.lr": True})

    with pytest.raises(ValueError, match="Boolean choice values"):
        materialize_trial(config, trial)


def test_materialize_trial_rejects_nested_bool_choice_for_numeric_target(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim": {"values": [{"lr": True}]}},
        wandb=None,
    )
    trial = Trial(id="0000", label="bool_lr", parameters={"optim": {"lr": True}})

    with pytest.raises(ValueError, match="Boolean choice values"):
        materialize_trial(config, trial)


def test_materialize_trial_allows_bool_choice_for_boolean_target(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"model.debug.random_init": {"values": [True]}},
        wandb=None,
    )
    trial = Trial(
        id="0000",
        label="random_init",
        parameters={"model.debug.random_init": True},
    )

    artifact = materialize_trial(config, trial)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["model"]["debug"]["random_init"] is True


def test_materialize_trial_preserves_completed_status_on_resume(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0, "objective": 0.42})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    materialize_trial(
        config,
        trial,
        resume=True,
        expected_checksums={
            "resolved_checksum": artifact.resolved_checksum,
            "base_checksums": artifact.base_checksums,
        },
    )
    after = json.loads(artifact.status_path.read_text())
    assert after["state"] == "completed"
    assert after["objective"] == 0.42

    materialize_trial(config, trial, resume=False)
    reset = json.loads(artifact.status_path.read_text())
    assert reset["state"] == "pending"


@pytest.mark.parametrize("status_text", ["{not-json", "[]"])
def test_materialize_trial_rejects_malformed_status_on_resume(
    tmp_path: Path, status_text: str
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    artifact.status_path.write_text(status_text)

    with pytest.raises(RuntimeError, match="Sweep status file"):
        materialize_trial(
            config,
            trial,
            resume=True,
            expected_checksums={
                "resolved_checksum": artifact.resolved_checksum,
                "base_checksums": artifact.base_checksums,
            },
        )


def test_materialize_trial_rejects_resume_skip_with_mismatched_status_id(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import SweepDriftError

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"id": "0001-deadbeef", "state": "completed", "returncode": 0, "objective": 0.42})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SweepDriftError, match="status.json belongs"):
        materialize_trial(
            config,
            trial,
            resume=True,
            expected_checksums={
                "resolved_checksum": artifact.resolved_checksum,
                "base_checksums": artifact.base_checksums,
            },
        )


def test_materialize_trial_detects_base_drift_on_resume(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import SweepDriftError

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    expected = {
        "resolved_checksum": artifact.resolved_checksum,
        "base_checksums": artifact.base_checksums,
    }
    original_overrides = artifact.overrides_path.read_text()
    original_resolved = artifact.resolved_path.read_text()
    original_command = artifact.command_path.read_text()

    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 99})

    try:
        materialize_trial(config, trial, resume=True, expected_checksums=expected)
    except SweepDriftError as exc:
        assert "base" in str(exc).lower()
    else:
        raise AssertionError("Expected SweepDriftError when base file changed under a completed trial")
    assert artifact.overrides_path.read_text() == original_overrides
    assert artifact.resolved_path.read_text() == original_resolved
    assert artifact.command_path.read_text() == original_command
    assert not list(artifact.trial_dir.glob(".overrides.toml.*.tmp.toml"))


def test_materialize_trial_detects_removed_base_on_resume(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import SweepDriftError

    base_a = tmp_path / "base-a.toml"
    base_b = tmp_path / "base-b.toml"
    write_toml(base_a, {"data": {"type": "fake"}, "max_steps": 1})
    write_toml(base_b, {})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_a, base_b],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    expected = {
        "resolved_checksum": artifact.resolved_checksum,
        "base_checksums": artifact.base_checksums,
    }
    resumed = SweepConfig(
        entrypoint="sft",
        base=[base_a],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    with pytest.raises(SweepDriftError, match="extra base"):
        materialize_trial(resumed, trial, resume=True, expected_checksums=expected)


def test_materialize_trial_rejects_resume_skip_without_manifest_checksums(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import SweepDriftError

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0, "objective": 0.42})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SweepDriftError, match="no checksum entry"):
        materialize_trial(config, trial, resume=True)


def test_materialize_trial_rejects_resume_skip_with_incomplete_manifest_checksums(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import SweepDriftError

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0, "objective": 0.42})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SweepDriftError, match="missing resolved/base checksums"):
        materialize_trial(
            config,
            trial,
            resume=True,
            expected_checksums={"base_checksums": artifact.base_checksums},
        )

    with pytest.raises(SweepDriftError, match="missing checksum.*base"):
        materialize_trial(
            config,
            trial,
            resume=True,
            expected_checksums={"resolved_checksum": artifact.resolved_checksum, "base_checksums": {}},
        )


def test_record_materialization_failure_clears_stale_resolved_file(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import record_trial_materialization_failure

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-stale", label="stale", parameters={"optim.lr": 1e-5})
    stale_resolved = config.output_dir / "trials" / trial.id / "resolved.toml"
    stale_resolved.parent.mkdir(parents=True, exist_ok=True)
    stale_resolved.write_text("stale = true\n")

    artifact = record_trial_materialization_failure(config, trial, ValueError("bad config"))

    assert not stale_resolved.exists()
    assert artifact.resolved_checksum == ""


def test_record_multi_run_materialization_failure_clears_stale_resolved_files(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import (
        multi_run_trial_dir,
        record_multi_run_materialization_failure,
    )

    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})
    config = SweepConfig(
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)
    trial = Trial(id="0000-stale", label="stale", parameters={"orchestrator.optim.lr": 1e-5})
    run_dir = multi_run_trial_dir(config, trial)
    stale_resolved = run_dir / "resolved.toml"
    stale_orch = run_dir / "control" / "orch.toml"
    stale_orch.parent.mkdir(parents=True, exist_ok=True)
    stale_resolved.write_text("stale = true\n")
    stale_orch.write_text("stale = true\n")

    artifact = record_multi_run_materialization_failure(config, trial, scheduler, ValueError("bad config"))

    assert not stale_resolved.exists()
    assert not stale_orch.exists()
    assert artifact.resolved_checksum == ""


# ---------------------------------------------------------------------------
# Phase 7a — multi_run_lora trial layout
# ---------------------------------------------------------------------------


def _stub_resolved_rl_config(monkeypatch, captured: dict) -> None:
    """Replace validate_target_config so we don't need a fully valid RLConfig."""
    from types import SimpleNamespace

    from prime_rl.sweep import materialize as mat_mod

    def fake_validate(entrypoint, args):
        captured["entrypoint"] = entrypoint
        captured["args"] = list(args)

        class FakeOrch:
            def __init__(self, source: dict) -> None:
                self._source = source
                self.output_dir = Path("/auto-setup/run_default")

            def model_dump(self, *, exclude_none=True, mode="json"):
                data = dict(self._source)
                output_dir = self.output_dir
                data["output_dir"] = (
                    output_dir.as_posix() if isinstance(output_dir, Path) else output_dir
                )
                return data

        trainer = SimpleNamespace(model=SimpleNamespace(lora=None))
        return SimpleNamespace(orchestrator=FakeOrch({}), trainer=trainer)

    monkeypatch.setattr(mat_mod, "validate_target_config", fake_validate)


def test_materialize_multi_run_trial_writes_run_layout(tmp_path: Path, monkeypatch) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    captured: dict = {}
    _stub_resolved_rl_config(monkeypatch, captured)

    config = SweepConfig(
        name="multi-run-test",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"orchestrator.optim.lr": 1e-5})

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    expected_run_dir = tmp_path / "study" / "shared" / "run_0000-deadbeef"
    assert artifact.run_dir == expected_run_dir
    assert artifact.trial_dir == expected_run_dir
    assert (expected_run_dir / "control" / "orch.toml").exists()
    assert (expected_run_dir / "status.json").exists()

    # The output_dir injected into the resolved orch.toml must match the
    # run dir the trainer will discover, otherwise the FileMonitor sidecar
    # would land somewhere the controller never reads.
    overrides = read_toml(artifact.overrides_path)
    assert overrides["orchestrator"]["output_dir"] == expected_run_dir.as_posix()
    assert overrides["orchestrator"]["optim"]["lr"] == 1e-5
    assert read_toml(artifact.resolved_path)["output_dir"] == expected_run_dir.as_posix()
    assert read_toml(expected_run_dir / "control" / "orch.toml")["output_dir"] == expected_run_dir.as_posix()

    from prime_rl.entrypoints.rl_multi_run_args import parse_runs_dirs

    command = shlex.split(artifact.command_path.read_text().strip())
    run_dirs, remaining_argv = parse_runs_dirs(command[1:])
    output_override_path = tmp_path / "study" / "shared" / "_output_override.toml"
    assert command[0] == "rl-multi-run"
    assert run_dirs == [expected_run_dir.resolve()]
    assert remaining_argv == [
        "@",
        shared_path.as_posix(),
        "@",
        output_override_path.as_posix(),
    ]
    assert read_toml(output_override_path)["output_dir"] == (tmp_path / "study" / "shared").as_posix()

    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pending"
    assert status["id"] == "0000-deadbeef"


def test_write_multi_run_output_override_escapes_toml_string(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import write_multi_run_output_override

    shared_dir = tmp_path / 'shared "quoted"'

    output_override = write_multi_run_output_override(shared_dir)

    assert read_toml(output_override)["output_dir"] == shared_dir.as_posix()


@pytest.mark.parametrize("raw", [":run_a", "run_a:", "run_a::run_b"])
def test_parse_runs_dirs_rejects_empty_entries(raw: str) -> None:
    from prime_rl.entrypoints.rl_multi_run_args import parse_runs_dirs

    with pytest.raises(SystemExit, match="empty run directory"):
        parse_runs_dirs(["--runs-dir", raw])


def test_parse_runs_dirs_rejects_duplicate_entries(tmp_path: Path) -> None:
    from prime_rl.entrypoints.rl_multi_run_args import parse_runs_dirs

    run_dir = tmp_path / "run_a"
    with pytest.raises(SystemExit, match="duplicate"):
        parse_runs_dirs(["--runs-dir", f"{run_dir}:{run_dir}"])


def test_materialize_multi_run_trial_injects_wandb_overrides(tmp_path: Path, monkeypatch) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    captured: dict = {}
    _stub_resolved_rl_config(monkeypatch, captured)

    config = SweepConfig(
        name="multi-run-wandb",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0001-cafebabe", label="lr_1e-5", parameters={"orchestrator.optim.lr": 1e-5})

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    wandb = read_toml(artifact.overrides_path)["orchestrator"]["wandb"]
    # group defaults to the sweep name when wandb.group is unset
    assert wandb["group"] == "multi-run-wandb"
    assert wandb["name"] == "lr_1e-5"
    # tags include the canonical sweep markers
    assert "sweep" in wandb["tags"]
    assert "trial:0001-cafebabe" in wandb["tags"]
    assert "study:multi-run-wandb" in wandb["tags"]


def test_materialize_multi_run_trial_allows_per_run_lora_rank_below_trainer(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-lora-rank",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.model.lora.rank": {"values": [4]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0002-rank", label="rank_4", parameters={"orchestrator.model.lora.rank": 4})

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    lora = read_toml(artifact.resolved_path)["model"]["lora"]
    assert lora["rank"] == 4
    assert lora["alpha"] == 32.0
    assert lora["name"] == "r4-a32.0"


def test_materialize_multi_run_trial_rejects_lora_rank_above_trainer(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-lora-rank",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.model.lora.rank": {"values": [16]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0003-rank", label="rank_16", parameters={"orchestrator.model.lora.rank": 16})

    with pytest.raises(ValueError, match="exceeds trainer.model.lora.rank"):
        materialize_multi_run_trial(config, trial, scheduler)


def test_materialize_multi_run_trial_rejects_shared_trainer_without_lora(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["trainer"]["model"].pop("lora", None)
    shared["trainer"].pop("ckpt", None)
    shared_path = tmp_path / "shared-no-lora.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-requires-lora",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0004-lora", label="lr", parameters={"orchestrator.optim.lr": 1e-5})

    with pytest.raises(ValueError, match="requires trainer.model.lora"):
        materialize_multi_run_trial(config, trial, scheduler)


def test_materialize_multi_run_trial_rejects_scheduler_above_trainer_concurrency(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-concurrency",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0005-concurrency", label="lr", parameters={"orchestrator.optim.lr": 1e-5})

    with pytest.raises(ValueError, match="trainer.max_concurrent_runs"):
        materialize_multi_run_trial(config, trial, scheduler)


def test_materialize_multi_run_trial_token_batch_size_requires_explicit_max_inflight(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-token-batch-requires-max-inflight",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.token_batch_size": {"values": [8192]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0004-token-batch",
        label="token_batch_8192",
        parameters={"orchestrator.token_batch_size": 8192},
    )

    with pytest.raises(ValueError, match="max_inflight_rollouts must be set"):
        materialize_multi_run_trial(config, trial, scheduler)


def test_materialize_multi_run_trial_token_batch_size_clears_inherited_batch_size(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["orchestrator"]["max_inflight_rollouts"] = 128
    shared_path = tmp_path / "shared-explicit-max-inflight.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-token-batch",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.token_batch_size": {"values": [8192]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0004-token-batch",
        label="token_batch_8192",
        parameters={"orchestrator.token_batch_size": 8192},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["token_batch_size"] == 8192
    assert resolved["max_inflight_rollouts"] == 128
    assert "batch_size" not in resolved


def test_materialize_multi_run_trial_token_batch_size_clears_inherited_oversampling(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["orchestrator"]["oversampling_factor"] = 2.0
    shared["orchestrator"]["max_inflight_rollouts"] = 256
    shared_path = tmp_path / "shared-explicit-oversampling.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-token-batch",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.token_batch_size": {"values": [8192]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0004-token-batch",
        label="token_batch_8192",
        parameters={"orchestrator.token_batch_size": 8192},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["token_batch_size"] == 8192
    assert resolved["max_inflight_rollouts"] == 256
    assert "batch_size" not in resolved
    assert "oversampling_factor" not in resolved


def test_materialize_multi_run_trial_batch_size_clears_inherited_token_batch_size(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["orchestrator"].pop("batch_size", None)
    shared["orchestrator"]["token_batch_size"] = 4096
    shared["orchestrator"]["max_inflight_rollouts"] = 64
    shared_path = tmp_path / "shared-token-batch.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-rollout-batch",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.batch_size": {"values": [32]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0005-batch",
        label="batch_32",
        parameters={"orchestrator.batch_size": 32},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["batch_size"] == 32
    assert "token_batch_size" not in resolved


def test_materialize_multi_run_trial_oversampling_clears_inherited_token_batch_size(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["orchestrator"].pop("batch_size", None)
    shared["orchestrator"]["token_batch_size"] = 4096
    shared["orchestrator"]["max_inflight_rollouts"] = 256
    shared_path = tmp_path / "shared-token-batch.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-oversampling",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.oversampling_factor": {"values": [2.0]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0005-oversampling",
        label="oversampling_2",
        parameters={"orchestrator.oversampling_factor": 2.0},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["batch_size"] == 128
    assert resolved["oversampling_factor"] == 2.0
    assert resolved["max_inflight_rollouts"] == 256
    assert "token_batch_size" not in resolved


def test_materialize_multi_run_trial_applies_top_level_orchestrator_scalars(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-rollouts-per-example",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={
            "orchestrator.rollouts_per_example": {"values": [8]},
            "orchestrator.strict_async_level": {"values": [True]},
        },
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0006-top-level-scalars",
        label="top_level_scalars",
        parameters={
            "orchestrator.rollouts_per_example": 8,
            "orchestrator.strict_async_level": True,
        },
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["rollouts_per_example"] == 8
    assert resolved["strict_async_level"] is True


def test_materialize_multi_run_trial_rejects_bool_choice_for_numeric_target(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-bool-max-off-policy",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.max_off_policy_steps": {"values": [True]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0007-bool-max-off-policy",
        label="bool_max_off_policy",
        parameters={"orchestrator.max_off_policy_steps": True},
    )

    with pytest.raises(ValueError, match="Boolean choice values"):
        materialize_multi_run_trial(config, trial, scheduler)


def test_materialize_multi_run_trial_applies_train_group_defaults_to_inherited_env(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-train-defaults",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={
            "orchestrator.train.sampling.temperature": {"values": [0.7]},
            "orchestrator.train.sampling.extra_body.min_p": {"values": [0.2]},
            "orchestrator.train.num_workers": {"values": [2]},
            "orchestrator.train.max_retries": {"values": [1]},
        },
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0007-train-defaults",
        label="train_defaults",
        parameters={
            "orchestrator.train.sampling.temperature": 0.7,
            "orchestrator.train.sampling.extra_body.min_p": 0.2,
            "orchestrator.train.num_workers": 2,
            "orchestrator.train.max_retries": 1,
        },
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    train = read_toml(artifact.resolved_path)["train"]
    assert train["sampling"]["temperature"] == 0.7
    assert train["sampling"]["extra_body"]["min_p"] == 0.2
    assert train["num_workers"] == 2
    assert train["max_retries"] == 1
    env = train["env"][0]
    assert env["sampling"]["temperature"] == 0.7
    assert env["sampling"]["extra_body"]["min_p"] == 0.2
    assert env["num_workers"] == 2
    assert env["max_retries"] == 1


def test_materialize_multi_run_trial_applies_train_sampling_table_to_inherited_env(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-train-sampling-table",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={
            "orchestrator.train.sampling": {
                "values": [{"temperature": 0.7, "extra_body": {"min_p": 0.2}}]
            },
        },
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0007-train-sampling-table",
        label="train_sampling_table",
        parameters={
            "orchestrator.train.sampling": {"temperature": 0.7, "extra_body": {"min_p": 0.2}},
        },
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    train = read_toml(artifact.resolved_path)["train"]
    assert train["sampling"]["temperature"] == 0.7
    assert train["sampling"]["extra_body"]["min_p"] == 0.2
    env = train["env"][0]
    assert env["sampling"]["temperature"] == 0.7
    assert env["sampling"]["extra_body"]["min_p"] == 0.2


def test_materialize_multi_run_trial_preserves_explicit_train_env_overrides(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    train_env = shared["orchestrator"]["train"]["env"][0]
    train_env["num_workers"] = 3
    train_env["max_retries"] = 5
    train_env["sampling"] = {"temperature": 0.2}
    shared_path = tmp_path / "shared-explicit-env.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-explicit-train-env",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={
            "orchestrator.train.sampling.temperature": {"values": [0.7]},
            "orchestrator.train.num_workers": {"values": [2]},
            "orchestrator.train.max_retries": {"values": [1]},
        },
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0008-explicit-train-env",
        label="explicit_train_env",
        parameters={
            "orchestrator.train.sampling.temperature": 0.7,
            "orchestrator.train.num_workers": 2,
            "orchestrator.train.max_retries": 1,
        },
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    env = read_toml(artifact.resolved_path)["train"]["env"][0]
    assert env["sampling"]["temperature"] == 0.2
    assert env["num_workers"] == 3
    assert env["max_retries"] == 5


def test_materialize_multi_run_trial_preserves_explicit_train_env_max_tokens_alias(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["orchestrator"]["train"]["env"][0]["sampling"] = {"max_tokens": 32}
    shared_path = tmp_path / "shared-env-max-tokens.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-env-max-tokens",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.train.sampling.max_tokens": {"values": [64]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0009-env-max-tokens",
        label="env_max_tokens",
        parameters={"orchestrator.train.sampling.max_tokens": 64},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    train = read_toml(artifact.resolved_path)["train"]
    assert train["sampling"]["max_completion_tokens"] == 64
    assert train["env"][0]["sampling"]["max_completion_tokens"] == 32


def test_materialize_multi_run_trial_preserves_deprecated_top_level_env_overrides(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    train = shared["orchestrator"].pop("train")
    train["env"][0]["num_workers"] = 3
    shared["orchestrator"]["train"] = {}
    shared["orchestrator"]["env"] = train["env"]
    shared["orchestrator"]["sampling"] = train["sampling"]
    shared_path = tmp_path / "shared-deprecated-env.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-deprecated-env",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.train.num_workers": {"values": [2]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0010-deprecated-env",
        label="deprecated_env",
        parameters={"orchestrator.train.num_workers": 2},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    train = read_toml(artifact.resolved_path)["train"]
    assert train["num_workers"] == 2
    assert train["env"][0]["num_workers"] == 3


def test_materialize_multi_run_trial_applies_eval_group_defaults_to_inherited_env(
    tmp_path: Path,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared = tomli.loads(Path("configs/ci/integration/rl_lora/start.toml").read_text())
    shared["orchestrator"]["eval"] = {}
    shared_path = tmp_path / "shared-eval.toml"
    write_toml(shared_path, shared)

    config = SweepConfig(
        name="multi-run-eval-defaults",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={
            "orchestrator.eval.sampling.temperature": {"values": [0.5]},
            "orchestrator.eval.sampling.extra_body.custom": {"values": [7]},
            "orchestrator.eval.num_examples": {"values": [512]},
            "orchestrator.eval.rollouts_per_example": {"values": [2]},
            "orchestrator.eval.interval": {"values": [50]},
            "orchestrator.eval.max_retries": {"values": [1]},
        },
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0009-eval-defaults",
        label="eval_defaults",
        parameters={
            "orchestrator.eval.sampling.temperature": 0.5,
            "orchestrator.eval.sampling.extra_body.custom": 7,
            "orchestrator.eval.num_examples": 512,
            "orchestrator.eval.rollouts_per_example": 2,
            "orchestrator.eval.interval": 50,
            "orchestrator.eval.max_retries": 1,
        },
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    eval_config = read_toml(artifact.resolved_path)["eval"]
    assert eval_config["sampling"]["temperature"] == 0.5
    assert eval_config["sampling"]["extra_body"]["custom"] == 7
    assert eval_config["num_examples"] == 512
    assert eval_config["rollouts_per_example"] == 2
    assert eval_config["interval"] == 50
    assert eval_config["max_retries"] == 1
    env = eval_config["env"][0]
    assert env["sampling"]["temperature"] == 0.5
    assert env["sampling"]["extra_body"]["custom"] == 7
    assert env["num_examples"] == 512
    assert env["rollouts_per_example"] == 2
    assert env["interval"] == 50
    assert env["max_retries"] == 1
    assert env["num_workers"] == 4


@pytest.mark.parametrize(
    (
        "path",
        "value",
        "expected_batch_size",
        "expected_oversampling_factor",
        "expected_max_inflight",
        "expected_env_workers",
    ),
    [
        ("orchestrator.batch_size", 1024, 1024, None, 1024, 4),
        ("orchestrator.oversampling_factor", 4.0, 128, 4.0, 512, 2),
    ],
)
def test_materialize_multi_run_trial_recomputes_auto_max_inflight_for_rollout_batching(
    tmp_path: Path,
    path: str,
    value: int | float,
    expected_batch_size: int,
    expected_oversampling_factor: float | None,
    expected_max_inflight: int,
    expected_env_workers: int,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-oversampling-factor",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={path: {"values": [value]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0007-batching",
        label="batching",
        parameters={path: value},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    resolved = read_toml(artifact.resolved_path)
    assert resolved["batch_size"] == expected_batch_size
    if expected_oversampling_factor is None:
        assert "oversampling_factor" not in resolved
    else:
        assert resolved["oversampling_factor"] == expected_oversampling_factor
    assert resolved["max_inflight_rollouts"] == expected_max_inflight
    assert resolved["train"]["env"][0]["num_workers"] == expected_env_workers


@pytest.mark.parametrize("section", ["train", "eval"])
def test_materialize_multi_run_trial_canonicalizes_sampling_max_tokens_alias(
    tmp_path: Path,
    section: str,
) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = Path("configs/ci/integration/rl_lora/start.toml")
    config = SweepConfig(
        name="multi-run-max-tokens",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={f"orchestrator.{section}.sampling.max_tokens": {"values": [64]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(
        id="0006-max-tokens",
        label="max_tokens_64",
        parameters={f"orchestrator.{section}.sampling.max_tokens": 64},
    )

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    sampling = read_toml(artifact.resolved_path)[section]["sampling"]
    assert sampling["max_completion_tokens"] == 64
    assert "max_tokens" not in sampling
