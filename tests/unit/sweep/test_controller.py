import json
import shlex
from pathlib import Path

import pytest
import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.controller import run_sweep
from prime_rl.sweep.materialize import SweepDriftError


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_run_sweep_dry_run_materializes_without_launching(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    launched = False

    def fake_local(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        name="unit",
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        dry_run=True,
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    run_sweep(config)

    assert not launched
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    variant_ids = [variant["id"] for variant in manifest["variants"]]
    assert [vid[:4] for vid in variant_ids] == ["0000", "0001"]
    assert all(len(vid) == 13 and vid[4] == "-" for vid in variant_ids)
    assert (tmp_path / "study" / "trials" / variant_ids[0] / "resolved.toml").exists()
    assert manifest["parameters"] == {
        "optim.lr": {"distribution": "choice", "values": [1e-05, 3e-05]}
    }
    assert "git" in manifest
    assert set(manifest["git"]) == {"sha", "dirty"}
    for variant in manifest["variants"]:
        assert len(variant["resolved_checksum"]) == 64
        assert variant["base_checksums"][base_path.as_posix()]


def test_run_sweep_dry_run_prints_shell_safe_commands(tmp_path: Path, capsys) -> None:
    base_path = tmp_path / "base config.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study with spaces",
        dry_run=True,
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    run_sweep(config)

    command_line = [line for line in capsys.readouterr().out.splitlines() if line.startswith("uv run")][0]
    command = shlex.split(command_line)
    assert command[:5] == ["uv", "run", "sft", "@", base_path.as_posix()]
    assert command[5] == "@"
    assert command[6].endswith("overrides.toml")
    assert "study with spaces" in command[6]


def test_run_sweep_records_materialization_failure_and_launches_valid_trials(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    launched_parameters = []

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        launched_parameters.extend(artifact.trial.parameters for artifact in artifacts)
        for artifact in artifacts:
            status = json.loads(artifact.status_path.read_text())
            status.update({"state": "completed", "returncode": 0})
            artifact.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"max_steps": {"values": [1, "bad"]}},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert launched_parameters == [{"max_steps": 1}]

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert [variant["overrides"] for variant in manifest["variants"]] == [
        {"max_steps": 1},
        {"max_steps": "bad"},
    ]

    valid_status = json.loads(Path(manifest["variants"][0]["status_path"]).read_text())
    failed_status = json.loads(Path(manifest["variants"][1]["status_path"]).read_text())
    assert valid_status["state"] == "completed"
    assert failed_status["state"] == "failed"
    assert failed_status["returncode"] == -1
    assert failed_status["objective"] is None
    assert failed_status["failure_stage"] == "materialization"
    assert "max_steps" in failed_status["error"]
    assert not Path(manifest["variants"][1]["status_path"]).with_name("resolved.toml").exists()


def test_run_sweep_dry_run_exits_nonzero_on_materialization_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    def fake_local(*args, **kwargs):
        raise AssertionError("dry run should not launch trials")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        dry_run=True,
        parameters={"max_steps": {"values": [1, "bad"]}},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert "Dry run found 1 failed trial materialization(s)." in capsys.readouterr().out

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    statuses = [json.loads(Path(variant["status_path"]).read_text()) for variant in manifest["variants"]]
    assert [status["state"] for status in statuses] == ["pending", "failed"]
    assert statuses[1]["failure_stage"] == "materialization"


def test_run_sweep_does_not_launch_after_materialization_failure_when_continue_false(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    def fake_local(*args, **kwargs):
        raise AssertionError("continue_on_failure=false should not launch after materialization failure")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"max_steps": {"values": [1, "bad", 2]}},
        continue_on_failure=False,
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert "Skipping trial launch: materialization failed" in capsys.readouterr().out

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert [variant["overrides"] for variant in manifest["variants"]] == [
        {"max_steps": 1},
        {"max_steps": "bad"},
    ]
    statuses = [json.loads(Path(variant["status_path"]).read_text()) for variant in manifest["variants"]]
    assert [status["state"] for status in statuses] == ["pending", "failed"]
    assert statuses[1]["failure_stage"] == "materialization"


def test_run_sweep_dispatches_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    called = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        called["count"] = len(artifacts)
        called["max_parallel"] = max_parallel
        called["gpu_groups"] = gpu_groups
        called["continue_on_failure"] = continue_on_failure
        called["retry_budget"] = retry_budget
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
    )

    run_sweep(config)

    assert called == {
        "count": 1,
        "max_parallel": 1,
        "gpu_groups": None,
        "continue_on_failure": True,
        "retry_budget": 1,
    }


def test_run_sweep_exits_nonzero_when_trials_fail(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    monkeypatch.setattr(
        "prime_rl.sweep.controller.run_trials_locally",
        lambda *args, **kwargs: 2,
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
    )

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when trials failed")


def test_run_sweep_random_strategy_dispatches_through_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        captured["count"] = len(artifacts)
        captured["parameters"] = [artifact.trial.parameters for artifact in artifacts]
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 5, "seed": 13},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "data.temperature": {"distribution": "uniform", "min": 0.6, "max": 1.2},
        },
    )

    run_sweep(config)

    assert captured["count"] == 5
    for params in captured["parameters"]:
        assert 1e-6 <= params["optim.lr"] <= 1e-4
        assert 0.6 <= params["data.temperature"] <= 1.2


def test_run_sweep_resume_skips_completed_trials(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    runs: list[list[str]] = []

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        runs.append([artifact.trial.id for artifact in artifacts])
        first_status = json.loads(artifacts[0].status_path.read_text())
        first_status.update({"state": "completed", "returncode": 0})
        artifacts[0].status_path.write_text(json.dumps(first_status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    run_sweep(SweepConfig(**base_config_kwargs))

    completed_id = runs[0][0]
    pending_id = runs[0][1]

    completed_status_path = tmp_path / "study" / "trials" / completed_id / "status.json"
    pending_status_path = tmp_path / "study" / "trials" / pending_id / "status.json"
    assert json.loads(completed_status_path.read_text())["state"] == "completed"
    pending_status = json.loads(pending_status_path.read_text())
    pending_status.update({"state": "failed", "returncode": 1})
    pending_status_path.write_text(json.dumps(pending_status, indent=2, sort_keys=True) + "\n")

    resume_runs: list[list[str]] = []

    def fake_local_resume(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        resume_runs.append(
            [(artifact.trial.id, json.loads(artifact.status_path.read_text())["state"]) for artifact in artifacts]
        )
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert resume_runs[0] == [(completed_id, "completed"), (pending_id, "pending")]
    assert json.loads(completed_status_path.read_text())["state"] == "completed"


def test_run_sweep_resume_seeds_tracker_from_completed_trials(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    captured_completion = []

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        # First run: mark both trials completed with recorded objectives.
        for value, artifact in zip([0.9, 0.7], artifacts):
            status = json.loads(artifact.status_path.read_text())
            status.update({"state": "completed", "returncode": 0, "objective": value})
            artifact.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    run_sweep(SweepConfig(**base_config_kwargs))

    def fake_local_resume(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        captured_completion.append([artifact.trial.id for artifact in artifacts])
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 2
    assert summary["best_value"] == 0.9


def test_run_sweep_resume_rejects_objective_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0, "objective": 0.9})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials when objective changed")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    drifted = {
        **base_config_kwargs,
        "objective": {"metric": "loss", "direction": "minimize"},
    }
    with pytest.raises(RuntimeError, match="objective changed"):
        run_sweep(SweepConfig(**drifted, resume=True))


def test_run_sweep_resume_propagates_trial_drift_instead_of_recording_failure(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    status_path = Path(manifest["variants"][0]["status_path"])
    original_status = json.loads(status_path.read_text())
    original_overrides = status_path.with_name("overrides.toml").read_text()
    original_resolved = status_path.with_name("resolved.toml").read_text()

    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 2})

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume drift should fail before launching trials")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(SweepDriftError, match="changed base"):
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert json.loads(status_path.read_text()) == original_status
    assert status_path.with_name("overrides.toml").read_text() == original_overrides
    assert status_path.with_name("resolved.toml").read_text() == original_resolved


def test_run_sweep_resume_propagates_malformed_status_instead_of_recording_failure(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    run_sweep(SweepConfig(**base_config_kwargs, dry_run=True))

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    status_path = Path(manifest["variants"][0]["status_path"])
    status_path.write_text("[]\n")

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("malformed status should fail before launching trials")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert status_path.read_text() == "[]\n"


def test_run_sweep_resume_rejects_parameter_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0, "objective": 0.9})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials when parameters changed")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    drifted = {
        **base_config_kwargs,
        "parameters": {"optim.lr": {"values": [1e-5, 1e-4]}},
    }
    with pytest.raises(RuntimeError, match="parameters changed"):
        run_sweep(SweepConfig(**drifted, resume=True))


def test_run_sweep_resume_rejects_parameter_order_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 2, "seed": 13},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
        },
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0, "objective": 0.9})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["parameter_order"] == ["optim.lr", "optim.warmup"]

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials when parameter order changed")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    reordered = {
        **base_config_kwargs,
        "parameters": {
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
        },
    }

    with pytest.raises(RuntimeError, match="parameter order changed"):
        run_sweep(SweepConfig(**reordered, resume=True))


@pytest.mark.parametrize("mutation", ["missing_id", "duplicate_id"])
def test_run_sweep_resume_rejects_malformed_manifest_variant_ids(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_config_kwargs, dry_run=True))

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "missing_id":
        manifest["variants"][0].pop("id")
    else:
        manifest["variants"].append(dict(manifest["variants"][0]))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials with malformed manifest IDs")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="malformed or duplicate variant id"):
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))


def test_run_sweep_resume_rejects_extra_manifest_variant_ids(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_config_kwargs, dry_run=True))

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    extra = dict(manifest["variants"][0])
    extra["id"] = "9999-extra"
    manifest["variants"].append(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials with extra manifest IDs")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="not in the regenerated trial set"):
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))


def test_run_sweep_resume_rejects_non_object_manifest(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_config_kwargs, dry_run=True))
    (tmp_path / "study" / "manifest.json").write_text("[]\n")

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials with malformed manifest")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="previous manifest.*JSON object"):
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))


def test_run_sweep_resume_rejects_invalid_json_manifest(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_config_kwargs, dry_run=True))
    (tmp_path / "study" / "manifest.json").write_text("{not valid json\n")

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials with invalid manifest JSON")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="previous manifest.*valid JSON"):
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))


def test_run_sweep_resume_rejects_entrypoint_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0, "objective": 0.9})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials when entrypoint changed")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="entrypoint changed"):
        run_sweep(SweepConfig(**{**base_config_kwargs, "entrypoint": "rl"}, resume=True))


def test_run_sweep_resume_rejects_random_strategy_seed_drift(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 2, "seed": 13},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0, "objective": 0.9})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials when random seed changed")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    drifted = {
        **base_config_kwargs,
        "strategy": {"type": "random", "num_trials": 2, "seed": 99},
    }
    with pytest.raises(RuntimeError, match="strategy changed"):
        run_sweep(SweepConfig(**drifted, resume=True))


def test_run_sweep_resume_allows_random_num_trials_extension(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 2, "seed": 13},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        status = json.loads(artifacts[0].status_path.read_text())
        status.update({"state": "completed", "returncode": 0, "objective": 0.9})
        artifacts[0].status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    launched: list[int] = []

    def fake_local_resume(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        launched.append(len(artifacts))
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    extended = {
        **base_config_kwargs,
        "strategy": {"type": "random", "num_trials": 3, "seed": 13},
    }
    run_sweep(SweepConfig(**extended, resume=True))

    assert launched == [3]


def test_run_sweep_resume_rejects_scheduler_type_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    monkeypatch.setattr("prime_rl.sweep.controller.submit_trials_to_slurm", lambda *args, **kwargs: 0)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm"},
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    run_sweep(config)

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should fail before launching trials when scheduler type changed")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(RuntimeError, match="scheduler type changed"):
        run_sweep(
            SweepConfig(
                entrypoint="sft",
                base=[base_path],
                output_dir=tmp_path / "study",
                scheduler={"type": "local"},
                parameters={"optim.lr": {"values": [1e-5]}},
                resume=True,
                wandb=None,
            )
        )


def test_run_sweep_resume_short_circuits_when_seeding_triggers_halt(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        for value, artifact in zip([0.9, 0.4], artifacts[:2]):
            status = json.loads(artifact.status_path.read_text())
            status.update({"state": "completed", "returncode": 0, "objective": value})
            artifact.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    invoked = []

    def fake_local_resume(*args, **kwargs):
        invoked.append(True)
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert invoked == []
    summary = json.loads((tmp_path / "study" / "manifest.json").read_text())["summary"]
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"


def test_run_sweep_resume_halts_on_preserved_missing_objective_when_continue_false(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        wandb=None,
    )

    run_sweep(SweepConfig(**base_config_kwargs, dry_run=True))

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    first_status_path = Path(manifest["variants"][0]["status_path"])
    status = json.loads(first_status_path.read_text())
    status.update({"state": "completed", "returncode": 0, "objective": None})
    first_status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    def fake_local_resume(*args, **kwargs):
        raise AssertionError("resume should not launch new trials after a preserved missing objective")

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert exc_info.value.code == 1


def test_run_sweep_skips_tracker_for_slurm_scheduler(tmp_path: Path, monkeypatch, capsys) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    monkeypatch.setattr("prime_rl.sweep.controller.submit_trials_to_slurm", lambda *args, **kwargs: 0)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm"},
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest.get("summary") is None
    assert "objective tracking is only computed for the local scheduler" in capsys.readouterr().out


def test_run_sweep_records_objective_and_halts_on_threshold(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    import subprocess as real_subprocess
    from types import SimpleNamespace

    seq = iter([0.9, 0.8, 0.2])
    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        run_dir = Path(overrides[0]).parent / "run"
        summary_dir = run_dir / "run-fake"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "final_summary.json").write_text(json.dumps({"reward": next(seq)}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4, 1e-3]}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 3
    assert summary["best_value"] == 0.9
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"


def test_run_sweep_counts_clean_exit_without_objective_as_failure(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    import subprocess as real_subprocess
    from types import SimpleNamespace

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        run_dir = Path(overrides[0]).parent / "run"
        summary_dir = run_dir / "run-fake"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.joinpath("final_summary.json").write_text(json.dumps({"other": 0.5}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 0
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["failure_stage"] == "objective"
        assert status["objective"] is None


def test_run_sweep_halts_on_missing_objective_when_continue_false(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    import subprocess as real_subprocess
    from types import SimpleNamespace

    real_run = real_subprocess.run
    spawned = {"n": 0}

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        spawned["n"] += 1
        run_dir = Path(overrides[0]).parent / "run"
        summary_dir = run_dir / "run-fake"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.joinpath("final_summary.json").write_text(json.dumps({"other": 0.5}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        retry_budget=0,
        wandb=None,
    )

    with pytest.raises(SystemExit):
        run_sweep(config)

    assert spawned["n"] == 1


def test_run_sweep_writes_summary_before_halting_on_trial_failure(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    import subprocess as real_subprocess
    from types import SimpleNamespace

    real_run = real_subprocess.run
    spawned = {"n": 0}

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        spawned["n"] += 1
        if spawned["n"] == 1:
            run_dir = Path(overrides[0]).parent / "run"
            summary_dir = run_dir / "run-fake"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_dir.joinpath("final_summary.json").write_text(json.dumps({"reward": 0.8}))
            return SimpleNamespace(returncode=0)
        if spawned["n"] == 2:
            return SimpleNamespace(returncode=9)
        raise AssertionError("continue_on_failure=false should stop before launching later trials")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        retry_budget=0,
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert spawned["n"] == 2
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["summary"]["completed"] == 1
    assert manifest["summary"]["best_value"] == 0.8

    states = [json.loads(Path(variant["status_path"]).read_text())["state"] for variant in manifest["variants"]]
    assert states == ["completed", "failed", "pending"]


def test_run_sweep_passes_gpu_groups_to_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        captured["max_parallel"] = max_parallel
        captured["gpu_groups"] = gpu_groups
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "local",
            "max_parallel": 2,
            "gpu_assignment": {"visible_devices": [[0, 1], [2, 3]]},
        },
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    run_sweep(config)

    assert captured == {"max_parallel": 2, "gpu_groups": [[0, 1], [2, 3]]}
