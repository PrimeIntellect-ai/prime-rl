from pathlib import Path

import tomli_w
import torch.distributed as dist

from prime_rl.trainer.runs import Runs

dist.init_process_group(backend="gloo")


def create_run_with_config(
    output_dir: Path,
    run_name: str,
    config: dict[str, object] | None = None,
) -> Path:
    """Helper function to create a run directory with a valid config.

    Args:
        output_dir: Parent directory where the run will be created
        run_name: Name of the run directory (e.g., 'run_abc123')
        config: Optional config dict. If None, uses a default valid config.

    Returns:
        Path to the created run directory
    """
    run_dir = output_dir / run_name
    run_dir.mkdir()
    config_dir = run_dir / "configs"
    config_dir.mkdir()

    if config is None:
        config = {
            "model": {"name": "test-model"},
            "batch_size": 32,
            "rollouts_per_example": 4,
            "env": [{"id": "test-env"}],
        }

    with open(config_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)

    return run_dir


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

    # Create some run directories with valid configs
    for run_name in ["run_abc123", "run_def456"]:
        create_run_with_config(tmp_path, run_name)

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

    # Create run directories with valid configs
    for run_name in ["run_abc123", "run_def456"]:
        create_run_with_config(tmp_path, run_name)

    # Detect initial runs
    runs.check_for_changes()
    initial_idx1 = runs.id_2_idx["run_abc123"]
    initial_idx2 = runs.id_2_idx["run_def456"]

    assert len(runs.id_2_idx) == 2
    assert len(runs.unused_idxs) == 3

    # Delete one run
    run1 = tmp_path / "run_abc123"
    import shutil

    shutil.rmtree(run1)
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

    # Create more runs than max_runs with valid configs
    for run_name in ["run_001", "run_002", "run_003"]:
        create_run_with_config(tmp_path, run_name)

    runs.check_for_changes()

    # Only max_runs should be tracked
    assert len(runs.id_2_idx) == 2
    assert len(runs.idx_2_id) == 2
    assert len(runs.unused_idxs) == 0

    to_delete_run = runs.get_run_dir(0)
    import shutil

    shutil.rmtree(to_delete_run)

    runs.check_for_changes()

    assert len(runs.id_2_idx) == 2
    assert len(runs.idx_2_id) == 2
    assert len(runs.unused_idxs) == 0
    assert to_delete_run not in runs.run_dirs()


def test_run_dirs(tmp_path: Path) -> None:
    """Test that run_dirs returns correct paths."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create run directories with valid configs
    for run_name in ["run_abc", "run_def"]:
        create_run_with_config(tmp_path, run_name)

    runs.check_for_changes()

    run_dirs = runs.run_dirs()
    assert len(run_dirs) == 2
    assert tmp_path / "run_abc" in run_dirs
    assert tmp_path / "run_def" in run_dirs


def test_non_run_directories_ignored(tmp_path: Path) -> None:
    """Test that non-run directories are ignored."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create mix of run and non-run directories
    create_run_with_config(tmp_path, "run_abc")

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
    test_config = {
        "model": {"name": "test-model"},
        "batch_size": 32,
        "max_steps": 1000,
        "rollouts_per_example": 4,
        "env": [{"id": "test-env"}],
    }
    create_run_with_config(tmp_path, "run_test123", config=test_config)

    # Detect the run
    runs.check_for_changes()

    # Verify config was loaded and parsed as OrchestratorConfig
    assert len(runs.config) == 1
    run_idx = runs.id_2_idx["run_test123"]
    assert run_idx in runs.config

    # Access config as OrchestratorConfig object
    config = runs.config[run_idx]
    assert config.model.name == "test-model"
    assert config.batch_size == 32
    assert config.max_steps == 1000


def test_config_missing(tmp_path: Path) -> None:
    """Test that runs without configs are skipped and error.txt is created."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create a run directory without config
    run_dir = tmp_path / "run_noconfig"
    run_dir.mkdir()

    # Detect the run
    runs.check_for_changes()

    # Verify run was not added
    assert len(runs.config) == 0
    assert "run_noconfig" not in runs.id_2_idx

    # Verify error.txt was created
    error_path = run_dir / "configs" / "error.txt"
    assert error_path.exists()
    error_content = error_path.read_text()
    assert "No orchestrator config found" in error_content


def test_config_cleanup_on_deletion(tmp_path: Path) -> None:
    """Test that configs are cleaned up when runs are deleted."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create a run directory with valid config
    test_config = {
        "model": {"name": "test-model"},
        "batch_size": 16,
        "rollouts_per_example": 4,
        "env": [{"id": "test-env"}],
    }
    run_dir = create_run_with_config(tmp_path, "run_delete_me", config=test_config)

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


def test_config_invalid(tmp_path: Path) -> None:
    """Test that runs with invalid configs are skipped and error.txt is created."""
    runs = Runs(output_dir=tmp_path, max_runs=5)

    # Create a run directory with invalid config (invalid type for a field)
    # Invalid config - batch_size should be int, not string
    invalid_config = {
        "model": {"name": "test-model"},
        "batch_size": "not-a-number",  # Invalid type
        "rollouts_per_example": 4,
        "env": [{"id": "test-env"}],
    }
    run_dir = create_run_with_config(tmp_path, "run_invalid", config=invalid_config)
    config_dir = run_dir / "configs"

    # Detect the run
    runs.check_for_changes()

    # Verify run was not added
    assert len(runs.config) == 0
    assert "run_invalid" not in runs.id_2_idx

    # Verify error.txt was created with error details
    error_path = config_dir / "error.txt"
    assert error_path.exists()
    error_content = error_path.read_text()
    assert "Error parsing orchestrator config" in error_content
