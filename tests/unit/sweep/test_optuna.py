import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomli_w

pytest.importorskip("optuna")

from prime_rl.configs.sweep import SweepConfig  # noqa: E402
from prime_rl.sweep.controller import run_sweep  # noqa: E402
from prime_rl.sweep.reproducibility import file_checksum  # noqa: E402
from prime_rl.sweep.search import parameters_hash  # noqa: E402


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def with_base_checksums(variant: dict, base_path: Path) -> dict:
    return {
        **variant,
        "resolved_checksum": variant.get("resolved_checksum", "0" * 64),
        "base_checksums": {base_path.as_posix(): file_checksum(base_path)},
    }


def optuna_trial_id(number: int, overrides: dict) -> str:
    return f"{number:04d}-{parameters_hash(overrides)}"


def _install_fake_run(monkeypatch, sequence):
    import subprocess as real_subprocess

    seq = iter(sequence)
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


def test_optuna_random_sweep_records_best_trial(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    rewards = [0.4, 0.7, 0.3, 0.9, 0.2]
    _install_fake_run(monkeypatch, rewards)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 5, "sampler": "random", "seed": 7},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
        },
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 5
    assert summary["best_value"] == max(rewards)
    assert len(manifest["variants"]) == 5
    for variant in manifest["variants"]:
        assert "optim.lr" in variant["overrides"]
        assert "optim.warmup" in variant["overrides"]


def test_optuna_fresh_run_rejects_existing_study_in_storage(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.5])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
            "study_name": "shared",
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))

    import optuna

    with pytest.raises(optuna.exceptions.DuplicatedStudyError):
        run_sweep(SweepConfig(**base_kwargs))


def test_optuna_resume_runs_only_remaining_budget(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    rewards = [0.4, 0.5, 0.6, 0.7, 0.8]
    _install_fake_run(monkeypatch, rewards)

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 5,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
            "study_name": "resume-study",
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    config = SweepConfig(**base_kwargs)
    config.strategy.num_trials = 3
    run_sweep(config)

    first_manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert len(first_manifest["variants"]) == 3
    first_ids = [v["id"] for v in first_manifest["variants"]]

    resume_config = SweepConfig(**base_kwargs, resume=True)
    run_sweep(resume_config)

    final_manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert len(final_manifest["variants"]) == 5
    assert [v["id"] for v in final_manifest["variants"][:3]] == first_ids
    assert final_manifest["summary"]["completed"] == 5
    assert final_manifest["summary"]["best_value"] == max(rewards)


def test_optuna_resume_rejects_base_config_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 2})

    with pytest.raises(RuntimeError, match="base config checksums"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_objective_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    drifted = {
        **base_kwargs,
        "objective": {"metric": "loss", "direction": "minimize"},
    }

    with pytest.raises(RuntimeError, match="objective changed"):
        run_sweep(SweepConfig(**drifted, resume=True))


def test_optuna_resume_rejects_parameter_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    drifted = {
        **base_kwargs,
        "parameters": {"optim.lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-3}},
    }

    with pytest.raises(RuntimeError, match="parameters changed"):
        run_sweep(SweepConfig(**drifted, resume=True))


def test_optuna_resume_rejects_parameter_order_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
        },
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    reordered = {
        **base_kwargs,
        "parameters": {
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
        },
    }

    with pytest.raises(RuntimeError, match="parameter order changed"):
        run_sweep(SweepConfig(**reordered, resume=True))


def test_optuna_resume_rejects_strategy_drift(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
            "study_name": "original-study",
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    drifted = {
        **base_kwargs,
        "strategy": {
            **base_kwargs["strategy"],
            "study_name": "different-study",
        },
    }

    with pytest.raises(RuntimeError, match="strategy changed"):
        run_sweep(SweepConfig(**drifted, resume=True))


def test_optuna_resume_rejects_existing_storage_without_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    (tmp_path / "study" / "manifest.json").unlink()

    with pytest.raises(RuntimeError, match="manifest variant entries"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_without_storage_trials(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_path = tmp_path / "optuna.db"
    storage_url = f"sqlite:///{storage_path}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    storage_path.unlink()

    with pytest.raises(RuntimeError, match="missing from storage"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_variant_without_status(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    Path(manifest["variants"][0]["status_path"]).unlink()

    with pytest.raises(RuntimeError, match="status.json"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_variant_with_missing_status_path_field(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"][0].pop("status_path")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="status.json"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


@pytest.mark.parametrize("status_text", ["{not-json", "[]"])
def test_optuna_resume_rejects_malformed_status_json(
    tmp_path: Path, monkeypatch, status_text: str
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    Path(manifest["variants"][0]["status_path"]).write_text(status_text)

    with pytest.raises(RuntimeError, match="valid JSON objects"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_variant_without_resolved_checksum(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"][0].pop("resolved_checksum")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="resolved_checksum"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_variant_with_mismatched_status_id(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    status_path = Path(manifest["variants"][0]["status_path"])
    status = json.loads(status_path.read_text())
    status["id"] = "0001-wrong"
    status_path.write_text(json.dumps(status))

    with pytest.raises(RuntimeError, match="status.json id"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_variant_with_mismatched_id_hash(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    variant = manifest["variants"][0]
    variant["id"] = "0000-deadbeef"
    status_path = Path(variant["status_path"])
    status = json.loads(status_path.read_text())
    status["id"] = variant["id"]
    status_path.write_text(json.dumps(status))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="ids and overrides"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_manifest_overrides_mismatching_storage_params(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    variant = manifest["variants"][0]
    variant["overrides"] = {"optim.lr": variant["overrides"]["optim.lr"] * 10}
    variant["id"] = optuna_trial_id(0, variant["overrides"])
    status_path = Path(variant["status_path"])
    status = json.loads(status_path.read_text())
    status["id"] = variant["id"]
    status_path.write_text(json.dumps(status))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="storage parameters"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_complete_storage_with_failed_status(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    status_path = Path(manifest["variants"][0]["status_path"])
    status = json.loads(status_path.read_text())
    status.update({"state": "failed", "returncode": -1, "objective": None})
    status_path.write_text(json.dumps(status))

    with pytest.raises(RuntimeError, match="terminal status.json files to match"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


@pytest.mark.parametrize(
    "status_update",
    [
        {"state": "completed", "returncode": 0, "objective": 0.4},
        {"state": "failed", "returncode": 1, "objective": 0.4},
    ],
)
def test_optuna_resume_rejects_failed_storage_status_mismatch(
    tmp_path: Path, monkeypatch, status_update: dict
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"

    import subprocess as real_subprocess

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as initial_exit:
        run_sweep(SweepConfig(**base_kwargs))
    assert initial_exit.value.code == 1

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    status_path = Path(manifest["variants"][0]["status_path"])
    status = json.loads(status_path.read_text())
    status.update(status_update)
    status_path.write_text(json.dumps(status))

    with pytest.raises(RuntimeError, match="terminal status.json files to match"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_rejects_duplicate_manifest_variants(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"].append(dict(manifest["variants"][0]))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="Duplicate"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        (None, "variants to be recorded as a list"),
        ("", "variants to be recorded as a list"),
        ({"id": "0000-not-a-list"}, "variants to be recorded as a list"),
        (["0000-not-an-object"], "variant entry.*JSON object"),
    ],
)
def test_optuna_resume_rejects_malformed_manifest_variants_shape(
    tmp_path: Path, monkeypatch, variants, message: str
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))
    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"] = variants
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match=message):
        run_sweep(SweepConfig(**base_kwargs, resume=True))


def test_optuna_resume_does_not_reconcile_running_trial_without_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    initial = SweepConfig(**base_kwargs)
    initial.strategy.num_trials = 1
    run_sweep(initial)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    pending = study.ask()
    (tmp_path / "study" / "manifest.json").unlink()

    with pytest.raises(RuntimeError, match="manifest variant entries"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    assert study.trials[pending.number].state == optuna.trial.TrialState.RUNNING


def test_optuna_resume_does_not_reconcile_running_trial_with_duplicate_manifest_entries(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    initial = SweepConfig(**base_kwargs)
    initial.strategy.num_trials = 1
    run_sweep(initial)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    pending = study.ask()
    pending_overrides = dict(pending.params)
    pending_id = optuna_trial_id(pending.number, pending_overrides)
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(
        json.dumps({"id": pending_id, "state": "completed", "returncode": 0, "objective": 0.95})
    )

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    variant = {
        "id": pending_id,
        "label": pending_id,
        "overrides": pending_overrides,
        "status_path": status_path.as_posix(),
        "output_dir": (trial_dir / "run").as_posix(),
    }
    variant = with_base_checksums(variant, base_path)
    manifest["variants"].extend([variant, dict(variant)])
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="Duplicate"):
        run_sweep(SweepConfig(**base_kwargs, resume=True))

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    assert study.trials[pending.number].state == optuna.trial.TrialState.RUNNING


def test_optuna_resume_counts_previous_failed_trials(tmp_path: Path, monkeypatch, capsys) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"

    import subprocess as real_subprocess

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
        (summary_dir / "final_summary.json").write_text(json.dumps({"other": 0.4}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as initial_exit:
        run_sweep(SweepConfig(**base_kwargs))
    assert initial_exit.value.code == 1
    capsys.readouterr()

    with pytest.raises(SystemExit) as resume_exit:
        run_sweep(SweepConfig(**base_kwargs, resume=True))
    assert resume_exit.value.code == 1
    assert "Sweep finished with 1 failed trial(s) out of 1." in capsys.readouterr().out


def test_optuna_marks_failed_materialization_in_storage(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.5, 0.6])

    import prime_rl.sweep.optuna_loop as loop

    original_materialize = loop.materialize_trial
    call_count = {"n": 0}

    def flaky_materialize(config, trial, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ValueError("fake config validation failure")
        return original_materialize(config, trial, **kwargs)

    monkeypatch.setattr(loop, "materialize_trial", flaky_materialize)

    base_kwargs = {
        "entrypoint": "sft",
        "base": [base_path],
        "output_dir": tmp_path / "study",
        "strategy": {
            "type": "optuna",
            "num_trials": 3,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        "parameters": {"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        "objective": {"metric": "reward", "direction": "maximize"},
        "wandb": None,
    }

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(SweepConfig(**base_kwargs))
    assert exc_info.value.code == 1

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.FAIL in states
    assert not any(state == optuna.trial.TrialState.RUNNING for state in states)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert len(manifest["variants"]) == 3
    failed_variant = next(variant for variant in manifest["variants"] if variant["id"].startswith("0001-"))
    failed_status = json.loads(Path(failed_variant["status_path"]).read_text())
    assert failed_status["state"] == "failed"
    assert failed_status["returncode"] == -1
    assert failed_status["failure_stage"] == "materialization"
    assert "fake config validation failure" in failed_status["error"]

    with pytest.raises(SystemExit) as resume_exit:
        run_sweep(SweepConfig(**base_kwargs, resume=True))
    assert resume_exit.value.code == 1


def test_optuna_resume_reconciles_running_trial_with_recorded_objective(
    tmp_path: Path, monkeypatch
) -> None:
    """Crash after subprocess finished but before study.tell(): replay the value."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.7, 0.6])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    config = SweepConfig(**base_kwargs)
    config.strategy.num_trials = 1
    run_sweep(config)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    completed_trial = study.trials[0]
    completed_value = completed_trial.value
    completed_params = dict(completed_trial.params)

    # Simulate the post-completion-but-pre-tell crash: ask another trial,
    # leave it RUNNING, and have its sweep status.json record an objective.
    pending = study.ask()
    pending_index = pending.number
    pending_overrides = dict(pending.params)
    pending_id = optuna_trial_id(pending_index, pending_overrides)
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(
        json.dumps({"id": pending_id, "state": "completed", "returncode": 0, "objective": 0.95})
    )

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"].append(
        with_base_checksums(
            {
                "id": pending_id,
                "label": pending_id,
                "overrides": pending_overrides,
                "status_path": status_path.as_posix(),
                "output_dir": (trial_dir / "run").as_posix(),
            },
            base_path,
        )
    )
    manifest_path.write_text(json.dumps(manifest))

    resume_config = SweepConfig(**base_kwargs, resume=True)
    run_sweep(resume_config)

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert all(state == optuna.trial.TrialState.COMPLETE for state in states)
    values = sorted(t.value for t in study.trials)
    assert values == sorted([completed_value, 0.95])
    assert dict(study.trials[0].params) == completed_params


def test_optuna_resume_reconciles_running_trial_with_no_recorded_objective(
    tmp_path: Path, monkeypatch
) -> None:
    """Crash before subprocess finished: mark the orphaned trial FAIL."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4, 0.5])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    initial = SweepConfig(**base_kwargs)
    initial.strategy.num_trials = 1
    run_sweep(initial)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    pending = study.ask()  # leak a RUNNING trial
    pending_overrides = dict(pending.params)
    pending_id = optuna_trial_id(pending.number, pending_overrides)
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(
        json.dumps({"id": pending_id, "state": "running", "returncode": None, "objective": None})
    )

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"].append(
        with_base_checksums(
            {
                "id": pending_id,
                "label": pending_id,
                "overrides": pending_overrides,
                "status_path": status_path.as_posix(),
                "output_dir": (trial_dir / "run").as_posix(),
            },
            base_path,
        )
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(SweepConfig(**base_kwargs, resume=True))
    assert exc_info.value.code == 1

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    # Reconciliation marks the orphan FAIL; the failed slot still counts
    # toward the num_trials budget, matching Optuna's own optimize() semantics.
    assert optuna.trial.TrialState.RUNNING not in states
    assert optuna.trial.TrialState.FAIL in states
    assert sum(1 for s in states if s == optuna.trial.TrialState.COMPLETE) == 1

    status = json.loads(status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -1
    assert status["objective"] is None


def test_optuna_resume_reconciles_running_trial_with_non_finite_objective_as_fail(
    tmp_path: Path, monkeypatch
) -> None:
    """Old status files may contain NaN from before finite objective guards.

    Resume should not pass that value to Optuna as a completed result; it is
    equivalent to a missing objective and must be attributed as a failed
    orphan.
    """
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4, 0.5])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    initial = SweepConfig(**base_kwargs)
    initial.strategy.num_trials = 1
    run_sweep(initial)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    pending = study.ask()
    pending_overrides = dict(pending.params)
    pending_id = optuna_trial_id(pending.number, pending_overrides)
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(
        json.dumps({"id": pending_id, "state": "completed", "returncode": 0, "objective": float("nan")})
    )

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"].append(
        with_base_checksums(
            {
                "id": pending_id,
                "label": pending_id,
                "overrides": pending_overrides,
                "status_path": status_path.as_posix(),
                "output_dir": (trial_dir / "run").as_posix(),
            },
            base_path,
        )
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(SweepConfig(**base_kwargs, resume=True))
    assert exc_info.value.code == 1

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.RUNNING not in states
    assert optuna.trial.TrialState.FAIL in states

    status = json.loads(status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == 0
    assert status["failure_stage"] == "objective"
    assert status["objective"] is None


def test_optuna_resume_reconciles_pruned_trial_as_pruned_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: a controller crash between record_trial_pruned() and
    study.tell(PRUNED) leaves the Optuna trial RUNNING in storage even
    though its sweep status.json reads ``state="pruned"``. Resume must
    reconcile that as TrialState.PRUNED, not the FAIL fallback."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.6, 0.5])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    initial = SweepConfig(**base_kwargs)
    initial.strategy.num_trials = 1
    run_sweep(initial)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)

    # Simulate the crash window: ask leaks a RUNNING trial, and the sweep's
    # status.json for that trial records the prune decision the controller
    # never got to tell Optuna.
    pending = study.ask()
    pending_index = pending.number
    pending_overrides = dict(pending.params)
    pending_id = optuna_trial_id(pending_index, pending_overrides)
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "id": pending_id,
                "state": "pruned",
                # Older/corrupt artifacts may carry a stale objective even
                # though pruned trials must not; resume should normalize it
                # while telling Optuna TrialState.PRUNED.
                "objective": 0.01,
                "pruned_at_step": 5,
                "pruned_value": 0.01,
            }
        )
    )

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"].append(
        with_base_checksums(
            {
                "id": pending_id,
                "label": pending_id,
                "overrides": pending_overrides,
                "status_path": status_path.as_posix(),
                "output_dir": (trial_dir / "run").as_posix(),
            },
            base_path,
        )
    )
    manifest_path.write_text(json.dumps(manifest))

    run_sweep(SweepConfig(**base_kwargs, resume=True))

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.RUNNING not in states
    # Crucial: the orphan with status="pruned" must come back as PRUNED, not
    # FAIL, so the sampler's history correctly reflects deliberate stops.
    assert optuna.trial.TrialState.PRUNED in states
    assert optuna.trial.TrialState.FAIL not in states
    status = json.loads(status_path.read_text())
    assert status["state"] == "pruned"
    assert status["objective"] is None


def test_optuna_sweep_halts_on_threshold(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    rewards = [0.9, 0.8, 0.2, 0.7, 0.6]
    _install_fake_run(monkeypatch, rewards)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 5, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    run_sweep(config)

    summary = json.loads((tmp_path / "study" / "manifest.json").read_text())["summary"]
    assert summary["completed"] == 3
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"
    assert summary["best_value"] == 0.9


# ---------------------------------------------------------------------------
# Phase 5b — pruning
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` used by the pruning driver tests.

    Drops a metrics.jsonl into the run directory on construction so the
    polling reader has data on the very first iteration, then reports a
    configurable returncode after the first ``wait()`` call.
    """

    def __init__(self, command, env=None, returncode: int = 0, rows=None, **kwargs):
        self._returncode_value = returncode
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.pid = 12345
        self._waits = 0
        if env is not None and rows is not None:
            metrics_path = Path(env["PRIME_RL_SWEEP_METRICS_JSONL"])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout=None) -> int:
        self._waits += 1
        # First call to wait blocks the polling loop long enough for one
        # metrics.jsonl read; second call returns the configured returncode.
        if self._waits >= 2:
            self._returncode = self._returncode_value
            return self._returncode_value
        import subprocess

        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)


def _patch_popen_for_trials(monkeypatch, factory) -> None:
    """Intercept Popen calls from the optuna pruning driver only.

    git_metadata() and other infrastructure also use subprocess (and thus
    Popen under the hood). Patching Popen globally breaks them; patching the
    ``subprocess.Popen`` attribute the optuna_loop module imports lets us
    target only the pruning driver's spawn site.
    """
    import subprocess as real_subprocess

    real_popen = real_subprocess.Popen

    def dispatch(command, *args, **kwargs):
        # Real Popen for everything that is not a trial subprocess (uv run rl/sft
        # compositions). The trial command shape is `["uv", "run", "rl"|"sft", "@", base, "@", overrides]`.
        if isinstance(command, (list, tuple)) and command and command[0] == "uv":
            return factory(command, *args, **kwargs)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr("prime_rl.sweep.optuna_loop.subprocess.Popen", dispatch)


def _patch_terminate(monkeypatch, terminated: list[bool]) -> None:
    def fake_terminate(process, grace_seconds=10.0):
        # Mirror the real function's idempotency: a finished process is a
        # no-op so the finally-clause cleanup does not double-count.
        if process.poll() is not None:
            return
        terminated.append(True)
        process._returncode = -15

    monkeypatch.setattr("prime_rl.sweep.optuna_loop._terminate_process_group", fake_terminate)


def test_run_trial_with_pruning_returns_pruned_when_should_prune_fires(tmp_path: Path, monkeypatch) -> None:
    """The polling driver must terminate the process and return PRUNED when
    optuna_trial.should_prune() returns True after a report."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-pruneme", label="pruneme", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]

    def popen_factory(*args, **kwargs):
        return _FakePopen(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    terminated: list[bool] = []
    _patch_terminate(monkeypatch, terminated)

    reports: list[tuple[int, float]] = []

    class FakeOptunaTrial:
        def report(self, value, step):
            reports.append((step, value))

        def should_prune(self):
            return True

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "pruned"
    assert outcome.pruned_at_step == 1
    assert outcome.pruned_value == 0.05
    assert reports == [(1, 0.05)]
    assert terminated == [True]
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pruned"
    assert status["returncode"] == -15
    assert "finished_at" in status
    assert status["pruned_at_step"] == 1
    assert status["pruned_value"] == 0.05
    assert status["objective"] is None


def test_run_trial_with_pruning_records_objective_on_clean_completion(tmp_path: Path, monkeypatch) -> None:
    """A trial that runs to completion without should_prune firing returns the
    final value from metrics.jsonl and is recorded as completed."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-good", label="good", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.4}]

    def popen_factory(*args, **kwargs):
        return _FakePopen(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    _patch_terminate(monkeypatch, [])

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.4
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0


def test_run_trial_with_pruning_does_not_prune_after_subprocess_exit(tmp_path: Path, monkeypatch) -> None:
    """Regression: a completed trial must not be re-classified as pruned even
    if its final intermediate point would have triggered should_prune().

    The polling loop reads the metric *after* process.wait() returns the real
    return code; without the gate on returncode, a should_prune-eligible last
    value would discard the valid final objective.
    """
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-late", label="late", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]

    class _FakePopenAlreadyExited(_FakePopen):
        """Returncode-on-first-wait variant: process is already done by the
        time the polling loop gets to read its metric."""

        def wait(self, timeout=None) -> int:
            self._returncode = self._returncode_value
            return self._returncode_value

    def popen_factory(*args, **kwargs):
        return _FakePopenAlreadyExited(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    terminated: list[bool] = []
    _patch_terminate(monkeypatch, terminated)

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return True  # pruner WOULD prune, but we are past the natural exit

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.05
    assert terminated == []  # never had to terminate
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"


def test_run_trial_with_pruning_does_not_prune_after_timeout_race(
    tmp_path: Path, monkeypatch
) -> None:
    """The process can exit after wait(timeout) times out but before the
    pruning decision. Re-check poll() so a valid final objective wins."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-timeout-race", label="timeout-race", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]

    class _FakePopenExitsOnPoll(_FakePopen):
        def poll(self) -> int | None:
            if self._waits >= 1 and self._returncode is None:
                self._returncode = self._returncode_value
            return self._returncode

    def popen_factory(*args, **kwargs):
        return _FakePopenExitsOnPoll(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    terminated: list[bool] = []
    _patch_terminate(monkeypatch, terminated)

    should_prune_calls = 0

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            nonlocal should_prune_calls
            should_prune_calls += 1
            return True

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.05
    assert should_prune_calls == 0
    assert terminated == []
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0


def test_optuna_no_pruner_counts_clean_exit_without_objective_as_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: when the no-pruner branch sees returncode==0 but the
    metric was never logged, Optuna gets TrialState.FAIL — but the sweep
    must also bump its own failure counter and exit non-zero. Otherwise the
    sweep finishes 'successfully' even though the sampler recorded failed
    trials and the run produced no usable objective."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    # _install_fake_run writes {"reward": ...} to final_summary.json. The
    # sweep asks for a metric named "missing", so read_final_summary returns
    # None on every clean exit.
    _install_fake_run(monkeypatch, [0.5, 0.5, 0.5])

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 3, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "missing", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)
    assert exc_info.value.code == 1

    summary = json.loads((tmp_path / "study" / "manifest.json").read_text())["summary"]
    # No completed trials with a usable objective.
    assert summary["completed"] == 0
    assert summary["best_value"] is None
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["failure_stage"] == "objective"


def test_optuna_no_pruner_halts_when_objective_missing_and_no_continue(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: continue_on_failure=False must halt the sweep on the
    first clean-exit-without-objective trial, not run the rest only to exit
    later."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    spawned = {"n": 0}
    import subprocess as real_subprocess

    real_run = real_subprocess.run

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
        # Writes the metric the sweep is NOT asking for.
        (summary_dir / "final_summary.json").write_text(json.dumps({"reward": 0.5}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 5, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "missing", "direction": "maximize"},
        continue_on_failure=False,
        wandb=None,
    )

    with pytest.raises(SystemExit):
        run_sweep(config)

    # Only the first trial ran; continue_on_failure=False halts immediately.
    assert spawned["n"] == 1


def test_optuna_no_pruner_writes_summary_before_halting_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    spawned = {"n": 0}
    import subprocess as real_subprocess

    real_run = real_subprocess.run

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
            (summary_dir / "final_summary.json").write_text(json.dumps({"reward": 0.9}))
            return SimpleNamespace(returncode=0)
        if spawned["n"] == 2:
            return SimpleNamespace(returncode=2)
        raise AssertionError("continue_on_failure=false should stop before launching later Optuna trials")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 5, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
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
    assert manifest["summary"]["best_value"] == 0.9
    states = [json.loads(Path(variant["status_path"]).read_text())["state"] for variant in manifest["variants"]]
    assert states == ["completed", "failed"]


def test_optuna_pruner_completed_without_objective_counts_as_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression for the pruner-enabled branch: a trial that exits cleanly
    but never logged the metric must count toward the sweep failure tally."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    # Run a sweep that exercises the pruner branch end-to-end. The fake
    # subprocess writes a sidecar for a different metric, so the sweep's
    # configured objective is missing on every trial.
    def popen_factory(*args, **kwargs):
        return _FakePopen(*args, rows=[{"step": 1, "different_metric": 0.5}], returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 3,
            "sampler": "random",
            "seed": 7,
            "pruner": {"type": "median", "n_startup_trials": 1, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)
    assert exc_info.value.code == 1

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["failure_stage"] == "objective"


def test_run_trial_with_pruning_skips_retry_after_intermediate_reports(tmp_path: Path, monkeypatch) -> None:
    """Regression: a failed attempt that already called optuna_trial.report
    must not be retried within the same Optuna trial. Retries on the same
    trial inherit the failed attempt's intermediate values, biasing pruning
    decisions and silently dropping duplicate-step reports."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning_and_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-noretry", label="noretry", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]
    spawned = {"n": 0}

    def popen_factory(*args, **kwargs):
        spawned["n"] += 1
        # Returncode 1 => failed attempt.
        return _FakePopen(*args, rows=rows, returncode=1, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    _patch_terminate(monkeypatch, [])

    reports: list[tuple[int, float]] = []

    class FakeOptunaTrial:
        def report(self, value, step):
            reports.append((step, value))

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_and_retries(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
        retry_budget=3,  # high budget on purpose; the early-exit must override
    )

    assert outcome.state == "failed"
    assert outcome.reports_sent == 1
    # Crucial assertion: only one Popen, despite retry_budget=3, because the
    # first attempt already sent intermediate reports.
    assert spawned["n"] == 1
    assert reports == [(1, 0.05)]


def test_run_trial_with_pruning_records_retry_attempt_count(tmp_path: Path, monkeypatch) -> None:
    """A pruner-enabled trial that fails before reporting can retry; the
    final status should keep the cumulative attempt count."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning_and_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-retry-prune", label="retry", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    spawned = {"n": 0}

    def popen_factory(*args, **kwargs):
        spawned["n"] += 1
        if spawned["n"] == 1:
            return _FakePopen(*args, rows=[], returncode=1, **kwargs)
        return _FakePopen(*args, rows=[{"step": 1, "reward": 0.9}], returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    _patch_terminate(monkeypatch, [])

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_and_retries(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
        retry_budget=1,
    )

    assert outcome.state == "completed"
    assert spawned["n"] == 2
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["attempts"] == 2


def test_run_trial_with_pruning_retries_launch_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning_and_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-launch-error", label="launch-error", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    spawned = {"n": 0}

    def popen_factory(*args, **kwargs):
        spawned["n"] += 1
        if spawned["n"] == 1:
            raise FileNotFoundError("temporary trial launcher miss")
        return _FakePopen(*args, rows=[{"step": 1, "reward": 0.9}], returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_and_retries(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
        retry_budget=1,
    )

    assert outcome.state == "completed"
    assert outcome.returncode == 0
    assert outcome.launch_error is False
    assert spawned["n"] == 2
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0
    assert status["attempts"] == 2
    assert "failure_stage" not in status


def test_run_trial_with_pruning_marks_launch_oserror_failed_after_retry_budget(
    tmp_path: Path, monkeypatch
) -> None:
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning_and_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-launch-error", label="launch-error", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    spawned = {"n": 0}

    def popen_factory(*args, **kwargs):
        spawned["n"] += 1
        raise FileNotFoundError("missing trial launcher")

    _patch_popen_for_trials(monkeypatch, popen_factory)

    class FakeOptunaTrial:
        def report(self, value, step):
            raise AssertionError("launch failures should not report metrics")

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_and_retries(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
        retry_budget=1,
    )

    assert outcome.state == "failed"
    assert outcome.returncode == -1
    assert outcome.launch_error is True
    assert spawned["n"] == 2
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -1
    assert status["failure_stage"] == "launch"


def test_run_with_retries_truncates_metrics_jsonl_between_attempts(tmp_path: Path, monkeypatch) -> None:
    """Regression: if attempt 1 fails after writing higher steps than the
    successful retry, the sidecar must be truncated between attempts so
    read_final_summary returns the retry's value, not the failed run's."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.metrics import read_final_summary
    from prime_rl.sweep.schedulers import _run_with_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-retry", label="retry", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    metrics_path = artifact.run_dir / "metrics.jsonl"
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    attempts = {"n": 0}

    def fake_run(command, env=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Failed attempt wrote higher-step bad value before crashing.
            metrics_path.write_text(json.dumps({"step": 50, "reward": 0.01}) + "\n")
            return SimpleNamespace(returncode=1)
        # Successful retry writes a lower-step good value. Without truncation,
        # read_final_summary would see step=50 reward=0.01 from the failed run
        # and report that instead of step=10 reward=0.9.
        existing = metrics_path.read_text() if metrics_path.exists() else ""
        metrics_path.write_text(existing + json.dumps({"step": 10, "reward": 0.9}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    returncode = _run_with_retries(artifact, gpu_group=None, retry_budget=1)

    assert returncode == 0
    assert attempts["n"] == 2
    # Truncation between attempts must wipe the failed run's row before the
    # successful retry writes its own line.
    assert read_final_summary(artifact.run_dir, "reward") == 0.9


def test_run_with_retries_removes_stale_legacy_summary(tmp_path: Path, monkeypatch) -> None:
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.metrics import read_final_summary
    from prime_rl.sweep.schedulers import _run_with_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-stale-summary", label="stale", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    stale_summary = artifact.run_dir / "run-old" / "final_summary.json"
    stale_summary.parent.mkdir(parents=True, exist_ok=True)
    stale_summary.write_text(json.dumps({"reward": 0.99}))

    def fake_run(command, env=None, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    returncode = _run_with_retries(artifact, gpu_group=None, retry_budget=0)

    assert returncode == 0
    assert not stale_summary.exists()
    assert read_final_summary(artifact.run_dir, "reward") is None


def test_optuna_sweep_with_median_pruner_runs_to_completion(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: sweep with median pruner enabled runs through the pruning
    code path. Trials with monotonically-improving metrics should not be
    pruned, so all trials complete successfully."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    final_rewards = iter([0.4, 0.7, 0.9])

    def popen_factory(*args, **kwargs):
        reward = next(final_rewards)
        rows = [{"step": 1, "reward": reward}]
        return _FakePopen(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 3,
            "sampler": "random",
            "seed": 7,
            "pruner": {"type": "median", "n_startup_trials": 1, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["strategy"]["pruner"]["type"] == "median"
    assert len(manifest["variants"]) == 3
    summary = manifest["summary"]
    # All three trials produced a valid objective; pruner only fires when the
    # trajectory is below the median, which is impossible with one prior
    # completion + an improving series.
    assert summary["completed"] == 3
    assert summary["best_value"] == 0.9


# ---------------------------------------------------------------------------
# Parallel SLURM-sync Optuna driver
# ---------------------------------------------------------------------------


def test_parallel_slurm_sync_runs_all_trials_concurrently(tmp_path: Path, monkeypatch) -> None:
    """With max_parallel=3, the driver should keep up to 3 workers busy and
    still run all num_trials trials. Tracks the peak concurrency observed
    inside the worker."""
    import threading

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    inflight_lock = threading.Lock()
    inflight = 0
    peak_inflight = 0
    rewards_per_call: list[float] = []

    def fake_worker(artifact, metric, retry_budget):
        nonlocal inflight, peak_inflight
        with inflight_lock:
            inflight += 1
            peak_inflight = max(peak_inflight, inflight)
        try:
            import time as _time
            _time.sleep(0.05)
            # Ascending rewards so Optuna sees variation but values
            # are deterministic per call index.
            value = 0.1 + 0.1 * len(rewards_per_call)
            rewards_per_call.append(value)
            # Worker simulates: trial wrote final_summary.json before exit.
            artifact.run_dir.mkdir(parents=True, exist_ok=True)
            (artifact.run_dir / "metrics.jsonl").write_text(
                json.dumps({metric: value, "step": 1}) + "\n"
            )
            from prime_rl.sweep.optuna_loop import _SlurmSyncWorkerResult
            return _SlurmSyncWorkerResult(returncode=0, objective=value)
        finally:
            with inflight_lock:
                inflight -= 1

    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_no_pruner", fake_worker
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 3},
        strategy={"type": "optuna", "num_trials": 6, "sampler": "tpe", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    # All 6 trials should have run; concurrency must have exceeded 1 to
    # prove the threaded driver actually parallelized work.
    assert len(rewards_per_call) == 6
    assert peak_inflight >= 2, f"expected peak_inflight >= 2, got {peak_inflight}"
    assert peak_inflight <= 3

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    # The fake worker bypasses _run_with_retries_slurm_sync, so it doesn't
    # write state="completed" to status.json — but the driver does record
    # the objective via record_trial_objective. Verify the per-trial
    # objective made it into status.json for all six trials.
    objectives = [
        json.loads(Path(variant["status_path"]).read_text())["objective"]
        for variant in manifest["variants"]
    ]
    assert sum(1 for obj in objectives if obj is not None) == 6
    # Six trials in flight at peak proves the driver actually parallelized.
    assert len({obj for obj in objectives if obj is not None}) >= 1


def test_parallel_slurm_sync_tpe_uses_constant_liar(tmp_path: Path) -> None:
    """When max_parallel > 1, the TPE sampler must be built with
    constant_liar=True so concurrent asks don't collide on the same region."""
    from prime_rl.configs.sweep import OptunaStrategyConfig
    from prime_rl.sweep.optuna_loop import _build_sampler, _import_optuna

    optuna = _import_optuna()
    strategy = OptunaStrategyConfig(num_trials=4, sampler="tpe", seed=1)

    serial = _build_sampler(optuna, strategy, concurrent_trials=1)
    parallel = _build_sampler(optuna, strategy, concurrent_trials=3)

    # Optuna stores the flag as a private attribute on TPESampler; verify
    # via the constructor argument we know we set.
    assert getattr(parallel, "_constant_liar", False) is True
    assert getattr(serial, "_constant_liar", False) is False


def test_parallel_slurm_sync_halts_on_early_stop(tmp_path: Path, monkeypatch) -> None:
    """When the early-stopping tracker fires, the driver must stop
    submitting new trials. In-flight trials may still complete, but the
    total trial count should be well under num_trials."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    calls = {"n": 0}
    from prime_rl.sweep.optuna_loop import _SlurmSyncWorkerResult

    def fake_worker(artifact, metric, retry_budget):
        calls["n"] += 1
        # Threshold halts when value is *worse* than threshold. For
        # direction=maximize, worse means value < threshold. Return 0.1
        # so the first completed trial trips the 0.9 threshold halt.
        value = 0.1
        artifact.run_dir.mkdir(parents=True, exist_ok=True)
        (artifact.run_dir / "metrics.jsonl").write_text(
            json.dumps({metric: value, "step": 1}) + "\n"
        )
        return _SlurmSyncWorkerResult(returncode=0, objective=value)

    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_no_pruner", fake_worker
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 2},
        strategy={"type": "optuna", "num_trials": 20, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.9},
        wandb=None,
    )

    run_sweep(config)

    # Initial fill submits max_parallel=2 trials. The first completion
    # trips the halt; the second in-flight trial may already be running
    # and will finish before the halt is observed. No further submissions,
    # so total is bounded by max_parallel — well under num_trials=20.
    assert calls["n"] <= 2, f"expected <= 2 trials, got {calls['n']}"


def test_parallel_slurm_sync_failed_trial_continues_with_continue_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """continue_on_failure=True (default) means a single failed trial does
    not halt the sweep — the failed slot is refilled and remaining trials
    proceed to completion."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    calls = {"n": 0}
    from prime_rl.sweep.optuna_loop import _SlurmSyncWorkerResult

    def fake_worker(artifact, metric, retry_budget):
        calls["n"] += 1
        if calls["n"] == 1:
            # First trial fails (non-zero returncode, no objective).
            return _SlurmSyncWorkerResult(returncode=1, objective=None)
        value = 0.5
        artifact.run_dir.mkdir(parents=True, exist_ok=True)
        (artifact.run_dir / "metrics.jsonl").write_text(
            json.dumps({metric: value, "step": 1}) + "\n"
        )
        return _SlurmSyncWorkerResult(returncode=0, objective=value)

    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_no_pruner", fake_worker
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 2},
        strategy={"type": "optuna", "num_trials": 4, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    # SystemExit==1 because the sweep recorded at least one failure, but
    # all four trials still ran (calls.n == 4) because continue_on_failure
    # defaults to True.
    assert exc_info.value.code == 1
    assert calls["n"] == 4


def test_parallel_slurm_sync_pruner_routes_through_pruning_worker(
    tmp_path: Path, monkeypatch
) -> None:
    """When the strategy has a non-trivial pruner, the parallel driver
    must hand each worker its optuna_trial and dispatch to the pruning
    runner rather than the no-pruner runner."""
    import threading

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    inflight_lock = threading.Lock()
    inflight = 0
    peak_inflight = 0
    pruning_worker_calls: list[str] = []
    no_pruner_worker_calls: list[str] = []

    def fake_pruning_worker(artifact, optuna_trial, metric, poll_interval, retry_budget):
        nonlocal inflight, peak_inflight
        with inflight_lock:
            inflight += 1
            peak_inflight = max(peak_inflight, inflight)
        pruning_worker_calls.append(artifact.trial.id)
        try:
            import time as _time
            _time.sleep(0.15)
            value = 0.5
            artifact.run_dir.mkdir(parents=True, exist_ok=True)
            (artifact.run_dir / "metrics.jsonl").write_text(
                json.dumps({metric: value, "step": 1}) + "\n"
            )
            from prime_rl.sweep.optuna_loop import _PollingOutcome
            return _PollingOutcome(
                state="completed", returncode=0, objective=value
            )
        finally:
            with inflight_lock:
                inflight -= 1

    def fake_no_pruner_worker(artifact, metric, retry_budget):
        no_pruner_worker_calls.append(artifact.trial.id)
        from prime_rl.sweep.optuna_loop import _SlurmSyncWorkerResult
        return _SlurmSyncWorkerResult(returncode=0, objective=0.5)

    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_with_pruner", fake_pruning_worker
    )
    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_no_pruner", fake_no_pruner_worker
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 3},
        strategy={
            "type": "optuna",
            "num_trials": 6,
            "sampler": "tpe",
            "seed": 7,
            "pruner": {"type": "median"},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    # All 6 trials should route to the pruning worker, none to the
    # no-pruner worker.
    assert len(pruning_worker_calls) == 6
    assert no_pruner_worker_calls == []
    # Peak concurrency > 1 confirms the threaded driver parallelizes the
    # pruning workers (Optuna ask/tell stays on the main thread, but
    # workers run their polling loops concurrently).
    assert peak_inflight >= 2, f"expected peak_inflight >= 2, got {peak_inflight}"
    assert peak_inflight <= 3


def test_parallel_slurm_sync_pruned_outcome_tells_optuna_pruned(
    tmp_path: Path, monkeypatch
) -> None:
    """A worker returning state='pruned' must result in study.tell with
    TrialState.PRUNED (not FAIL, not COMPLETE) and no objective recorded."""
    import optuna as _optuna

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    told_states: list = []

    def fake_pruning_worker(artifact, optuna_trial, metric, poll_interval, retry_budget):
        from prime_rl.sweep.optuna_loop import _PollingOutcome
        return _PollingOutcome(
            state="pruned",
            returncode=-1,
            objective=None,
            pruned_at_step=3,
            pruned_value=0.2,
            reports_sent=1,
        )

    real_tell = _optuna.Study.tell

    def spy_tell(self, trial, values=None, state=None, **kwargs):
        told_states.append(state)
        return real_tell(self, trial, values=values, state=state, **kwargs)

    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_with_pruner", fake_pruning_worker
    )
    monkeypatch.setattr(_optuna.Study, "tell", spy_tell)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 2},
        strategy={
            "type": "optuna",
            "num_trials": 3,
            "sampler": "tpe",
            "seed": 7,
            "pruner": {"type": "median"},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    # All 3 trials should have been told as PRUNED — never None (success)
    # and never FAIL.
    assert told_states == [
        _optuna.trial.TrialState.PRUNED,
        _optuna.trial.TrialState.PRUNED,
        _optuna.trial.TrialState.PRUNED,
    ]


def test_parallel_slurm_sync_pruner_unsafe_to_continue_halts(
    tmp_path: Path, monkeypatch
) -> None:
    """If a pruning worker returns unsafe_to_continue=True (persistent
    squeue failure or unconfirmed scancel), the driver must halt new
    submissions even with continue_on_failure=True (the default).
    The underlying SLURM job may still be alive."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    calls = {"n": 0}

    def fake_pruning_worker(artifact, optuna_trial, metric, poll_interval, retry_budget):
        from prime_rl.sweep.optuna_loop import _PollingOutcome
        calls["n"] += 1
        return _PollingOutcome(
            state="failed",
            returncode=-1,
            objective=None,
            unsafe_to_continue=True,
        )

    monkeypatch.setattr(
        "prime_rl.sweep.optuna_loop._run_one_slurm_sync_with_pruner", fake_pruning_worker
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "synchronous": True, "max_parallel": 2},
        strategy={
            "type": "optuna",
            "num_trials": 10,
            "sampler": "tpe",
            "seed": 7,
            "pruner": {"type": "median"},
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
        # default continue_on_failure=True — unsafe_to_continue should
        # still halt the sweep.
    )

    with pytest.raises(SystemExit):
        run_sweep(config)

    # Initial fill of max_parallel=2 trials runs; both come back
    # unsafe_to_continue. No more submissions after that. Total <= 2.
    assert calls["n"] <= 2, f"expected <= 2 trials, got {calls['n']}"
