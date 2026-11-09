import sys
from pathlib import Path

import pytest

from prime_rl.configs.shared import ResumeConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.utils.pathing import (
    check_resume_dir_mode,
    clean_future_steps,
    create_attempt_dirs,
    get_batch_dir,
    get_broadcast_dir,
    get_checkpoint_mode,
    get_step_path,
    validate_run_dir,
)


def test_nonexistent_dir_passes(tmp_path):
    run_dir = tmp_path / "does_not_exist"
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_launcher_artifacts_passes(tmp_path):
    run_dir = tmp_path / "submitted"
    run_dir.mkdir()
    config_dir, log_dir = create_attempt_dirs(run_dir)
    (config_dir / "trainer.json").touch()
    next_config_dir, next_log_dir = create_attempt_dirs(run_dir)
    assert next_config_dir.parent.name == "attempt_2"
    assert next_log_dir.name == "attempt_2"
    assert (run_dir / "configs" / "latest").resolve() == next_config_dir.parent
    assert (run_dir / "logs" / "latest").resolve() == next_log_dir
    (run_dir / "launcher").mkdir()
    (run_dir / "launcher" / "rl.sbatch").touch()
    (run_dir / "launcher" / ".trainer.done").touch()
    (run_dir / "launcher" / ".orchestrator.done").touch()
    (run_dir / "launcher" / "logs").mkdir()
    (run_dir / "launcher" / "logs" / "job_1234.log").touch()
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_logs_raises(tmp_path):
    run_dir = tmp_path / "has_logs"
    run_dir.mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "trainer.log").touch()
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains artifacts"):
        validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    run_dir = tmp_path / "has_ckpt"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_0").mkdir()
    (run_dir / "logs").mkdir()

    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=True)

    assert not run_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    run_dir = tmp_path / "does_not_exist"
    validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=True)
    assert not run_dir.exists()


def test_ckpt_output_dir_with_checkpoints_raises(tmp_path):
    run_dir = tmp_path / "fresh"
    ckpt_output_dir = tmp_path / "ckpts"
    ckpt_output_dir.mkdir()
    (ckpt_output_dir / "checkpoints").mkdir()
    (ckpt_output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_run_dir(run_dir, output_dir=tmp_path, resuming=False, clean=False, ckpt_output_dir=ckpt_output_dir)


def test_clean_outside_output_dir_raises(tmp_path):
    escaped = tmp_path / "elsewhere"
    escaped.mkdir()
    with pytest.raises(ValueError, match="remain under output_dir"):
        validate_run_dir(escaped, output_dir=tmp_path / "outputs", resuming=False, clean=True)
    assert escaped.exists()


def test_clean_future_steps_rebuilds_resume_broadcast(tmp_path):
    batch_dir = get_batch_dir(tmp_path)
    broadcast_dir = get_broadcast_dir(tmp_path)
    for parent in (batch_dir, broadcast_dir):
        for step in (1, 2, 3):
            get_step_path(parent, step).mkdir(parents=True)

    clean_future_steps(tmp_path, resume_step=2)

    assert get_step_path(batch_dir, 1).exists()
    assert get_step_path(batch_dir, 2).exists()
    assert not get_step_path(batch_dir, 3).exists()
    assert get_step_path(broadcast_dir, 1).exists()
    assert not get_step_path(broadcast_dir, 2).exists()
    assert not get_step_path(broadcast_dir, 3).exists()


ALL_SKIPS = CheckpointConfig(skip_scheduler=True, skip_optimizer=True, skip_dataloader=True)
"""The ``ckpt`` flags that load only the weights — the workaround from the incident."""


def _write_run_configs(run_dir: Path, *names: str) -> Path:
    """A run directory as a launcher leaves it: ``configs/attempt_1/resolved/<name>.json``."""
    config_dir, _ = create_attempt_dirs(run_dir)
    for name in names:
        (config_dir / f"{name}.json").touch()
    return config_dir


def _step_dir(run_dir: Path, step: int) -> Path:
    step_dir = run_dir / "checkpoints" / f"step_{step}"
    step_dir.mkdir(parents=True)
    return step_dir


def test_resume_dir_from_another_mode_raises(tmp_path):
    """A pretrain checkpoint cannot resume an sft run: the error names both modes and the way out."""
    source_dir = tmp_path / "pretrain-run"
    _write_run_configs(source_dir, "pretrain", "eval")
    step_dir = _step_dir(source_dir, 300)

    with pytest.raises(ValueError) as error:
        check_resume_dir_mode(ResumeConfig(dir=step_dir), "sft")

    message = str(error.value)
    assert f"the pretrain checkpoint {step_dir}" in message
    assert "this run is sft" in message
    assert f'`model.init_from = "{step_dir}"`' in message
    assert "ckpt.skip_scheduler" in message


def test_resume_dir_same_mode_passes(tmp_path):
    """Forking a run of the same kind keeps working, with only online evals' config alongside."""
    source_dir = tmp_path / "sft-run"
    _write_run_configs(source_dir, "sft", "eval")

    check_resume_dir_mode(ResumeConfig(dir=_step_dir(source_dir, 50)), "sft")


def test_resume_dir_unknown_mode_passes(tmp_path):
    """A checkpoint whose run dir keeps no resolved config keeps today's behavior."""
    source_dir = tmp_path / "old-run"
    config_dir, _ = create_attempt_dirs(source_dir)
    (config_dir / "trainer.json").touch()

    check_resume_dir_mode(ResumeConfig(dir=_step_dir(source_dir, 10)), "pretrain")


def test_resume_dir_mode_ignores_configs_no_kind_owns(tmp_path):
    """``trainer.json`` is every trainer's config, and a hosted sft attempt held only ``eval.json``."""
    source_dir = tmp_path / "hosted-run"
    _write_run_configs(source_dir, "trainer", "eval")

    assert get_checkpoint_mode(_step_dir(source_dir, 10)) is None


def test_resume_dir_mode_reads_attempts_without_latest_symlink(tmp_path):
    """A hosted volume restores ``configs/latest`` from R2 as a plain ``latest.rclonelink`` file."""
    source_dir = tmp_path / "hosted-run"
    _write_run_configs(source_dir, "pretrain")
    (source_dir / "configs" / "latest").unlink()
    (source_dir / "configs" / "latest.rclonelink").touch()

    assert get_checkpoint_mode(_step_dir(source_dir, 5)) == "pretrain"


def test_resume_dir_mode_reads_the_newest_attempt_first(tmp_path):
    """``attempt_10`` wins over ``attempt_2``: the attempts are ordered by number, not by name."""
    source_dir = tmp_path / "hosted-run"
    for number, mode in ((2, "pretrain"), (10, "sft")):
        resolved = source_dir / "configs" / f"attempt_{number}" / "resolved"
        resolved.mkdir(parents=True)
        (resolved / f"{mode}.json").touch()
    (source_dir / "configs" / "latest.rclonelink").touch()

    assert get_checkpoint_mode(_step_dir(source_dir, 5)) == "sft"


def test_resume_dir_all_skips_force_the_load(tmp_path):
    source_dir = tmp_path / "rl-run"
    # A local rl launch writes only its subconfigs: the orchestrator's config is rl's alone.
    _write_run_configs(source_dir, "trainer", "orchestrator")
    step_dir = _step_dir(source_dir, 20)

    with pytest.raises(ValueError):
        check_resume_dir_mode(ResumeConfig(dir=step_dir), "pretrain")
    check_resume_dir_mode(ResumeConfig(dir=step_dir), "pretrain", ckpt=ALL_SKIPS)


def test_resume_dir_from_another_mode_raises_for_rl(tmp_path):
    """No ckpt.skip_* flag stands in for the orchestrator's progress: rl cannot force the load."""
    source_dir = tmp_path / "sft-run"
    _write_run_configs(source_dir, "sft")
    step_dir = _step_dir(source_dir, 100)

    with pytest.raises(ValueError) as error:
        check_resume_dir_mode(ResumeConfig(dir=step_dir), "rl", ckpt=ALL_SKIPS)

    assert f'`trainer.model.init_from = "{step_dir}"`' in str(error.value)


def test_sft_trainer_entrypoint_rejects_cross_mode_resume(tmp_path, monkeypatch):
    """A hosted ``-m prime_rl.trainer.sft.train`` stops before the trainer's heavy imports."""
    source_dir = tmp_path / "pretrain-run"
    _write_run_configs(source_dir, "pretrain")
    step_dir = _step_dir(source_dir, 5)

    from prime_rl.trainer.sft.train import main

    monkeypatch.setattr(sys, "argv", ["sft", "--resume.dir", str(step_dir)])
    with pytest.raises(ValueError, match="this run is sft"):
        main()


def test_orchestrator_entrypoint_rejects_cross_mode_resume(tmp_path, monkeypatch):
    """A hosted ``orchestrator`` stops on an sft checkpoint before it loads its own progress."""
    source_dir = tmp_path / "sft-run"
    _write_run_configs(source_dir, "sft")
    step_dir = _step_dir(source_dir, 10)

    import prime_rl.entrypoints.orchestrator as entrypoint

    monkeypatch.setattr(sys, "argv", ["orchestrator", "--resume.dir", str(step_dir)])
    with pytest.raises(ValueError, match="this run is rl"):
        entrypoint.main()
