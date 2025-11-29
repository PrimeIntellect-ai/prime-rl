from pathlib import Path

import tomli_w

from prime_rl.trainer.runs import Runs


def test_initial_state(tmp_path: Path) -> None:
    """Test that Runs initializes correctly."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    assert runs.output_dir == tmp_path
    assert runs.max_runs == 5
    assert len(runs.idx_2_id) == 0
    assert len(runs.id_2_idx) == 0
    assert len(runs.unused_idxs) == 5
    assert runs.run_dirs() == []


def test_detect_new_runs(tmp_path: Path) -> None:
    """Test that new runs are detected correctly."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create some run directories
    (tmp_path / "run_abc123").mkdir()
    (tmp_path / "run_def456").mkdir()

    # Check for changes
    runs.check_for_changes()

    # Verify runs were detected
    assert len(runs.id_2_idx) == 2
    assert len(runs.idx_2_id) == 2
    assert "run_abc123" in runs.id_2_idx
    assert "run_def456" in runs.id_2_idx

    # Verify indices are assigned from available pool
    assert len(runs.unused_idxs) == 3  # 5 - 2 = 3
    assert runs.id_2_idx["run_abc123"] in range(5)
    assert runs.id_2_idx["run_def456"] in range(5)

    # Verify bidirectional mapping
    idx1 = runs.id_2_idx["run_abc123"]
    idx2 = runs.id_2_idx["run_def456"]
    assert runs.idx_2_id[idx1] == "run_abc123"
    assert runs.idx_2_id[idx2] == "run_def456"


def test_detect_deleted_runs(tmp_path: Path) -> None:
    """Test that deleted runs are detected correctly."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create run directories
    run1 = tmp_path / "run_abc123"
    run2 = tmp_path / "run_def456"
    run1.mkdir()
    run2.mkdir()

    # Detect initial runs
    runs.check_for_changes()
    initial_idx1 = runs.id_2_idx["run_abc123"]
    initial_idx2 = runs.id_2_idx["run_def456"]

    assert len(runs.id_2_idx) == 2
    assert len(runs.unused_idxs) == 3

    # Delete one run
    run1.rmdir()
    runs.check_for_changes()

    # Verify run was removed
    assert len(runs.id_2_idx) == 1
    assert len(runs.idx_2_id) == 1
    assert "run_abc123" not in runs.id_2_idx
    assert "run_def456" in runs.id_2_idx
    assert initial_idx1 not in runs.idx_2_id

    # Verify index was returned to unused pool
    assert len(runs.unused_idxs) == 4
    assert initial_idx1 in runs.unused_idxs
    assert initial_idx2 not in runs.unused_idxs


def test_max_runs_limit(tmp_path: Path) -> None:
    """Test that only max_runs are tracked."""
    runs = Runs(output_dir=tmp_path, max_runs=2)

    # Create more runs than max_runs
    (tmp_path / "run_001").mkdir()
    (tmp_path / "run_002").mkdir()
    (tmp_path / "run_003").mkdir()

    runs.check_for_changes()

    # Only max_runs should be tracked
    assert len(runs.id_2_idx) == 2
    assert len(runs.idx_2_id) == 2
    assert len(runs.unused_idxs) == 0

    to_delete_run = runs.get_run_dir(0)
    to_delete_run.rmdir()

    runs.check_for_changes()

    assert len(runs.id_2_idx) == 2
    assert len(runs.idx_2_id) == 2
    assert len(runs.unused_idxs) == 0
    assert to_delete_run not in runs.run_dirs()


def test_run_dirs(tmp_path: Path) -> None:
    """Test that run_dirs returns correct paths."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create run directories
    (tmp_path / "run_abc").mkdir()
    (tmp_path / "run_def").mkdir()

    runs.check_for_changes()

    run_dirs = runs.run_dirs()
    assert len(run_dirs) == 2
    assert tmp_path / "run_abc" in run_dirs
    assert tmp_path / "run_def" in run_dirs


def test_non_run_directories_ignored(tmp_path: Path) -> None:
    """Test that non-run directories are ignored."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create mix of run and non-run directories
    (tmp_path / "run_abc").mkdir()
    (tmp_path / "other_dir").mkdir()
    (tmp_path / "random").mkdir()

    runs.check_for_changes()

    # Only run_* directories should be tracked
    assert len(runs.id_2_idx) == 1
    assert "run_abc" in runs.id_2_idx
    assert "other_dir" not in runs.id_2_idx
    assert "random" not in runs.id_2_idx


def test_config_loading(tmp_path: Path) -> None:
    """Test that orchestrator configs are loaded correctly."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create a run directory with config
    run_dir = tmp_path / "run_test123"
    run_dir.mkdir()
    config_dir = run_dir / "configs"
    config_dir.mkdir()

    # Create a sample orchestrator config
    test_config = {
        "model": {"name": "test-model"},
        "batch_size": 32,
        "max_steps": 1000,
    }
    with open(config_dir / "orch.toml", "wb") as f:
        tomli_w.dump(test_config, f)

    # Detect the run
    runs.check_for_changes()

    # Verify config was loaded
    assert len(runs.config) == 1
    run_idx = runs.id_2_idx["run_test123"]
    assert run_idx in runs.config
    assert runs.config[run_idx]["model"]["name"] == "test-model"
    assert runs.config[run_idx]["batch_size"] == 32
    assert runs.config[run_idx]["max_steps"] == 1000


def test_config_missing(tmp_path: Path) -> None:
    """Test that runs without configs get empty dict."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create a run directory without config
    run_dir = tmp_path / "run_noconfig"
    run_dir.mkdir()

    # Detect the run
    runs.check_for_changes()

    # Verify empty config was set
    assert len(runs.config) == 1
    run_idx = runs.id_2_idx["run_noconfig"]
    assert run_idx in runs.config
    assert runs.config[run_idx] == {}


def test_config_cleanup_on_deletion(tmp_path: Path) -> None:
    """Test that configs are cleaned up when runs are deleted."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create a run directory with config
    run_dir = tmp_path / "run_delete_me"
    run_dir.mkdir()
    config_dir = run_dir / "configs"
    config_dir.mkdir()

    test_config = {"test": "value"}
    with open(config_dir / "orch.toml", "wb") as f:
        tomli_w.dump(test_config, f)

    # Detect the run
    runs.check_for_changes()
    run_idx = runs.id_2_idx["run_delete_me"]
    assert run_idx in runs.config

    # Delete the run directory
    import shutil

    shutil.rmtree(run_dir)
    runs.check_for_changes()

    # Verify config was cleaned up
    assert run_idx not in runs.config
    assert "run_delete_me" not in runs.id_2_idx
