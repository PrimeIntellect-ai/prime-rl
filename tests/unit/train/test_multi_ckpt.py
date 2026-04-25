from pathlib import Path
from typing import Generator

import pytest
import tomli_w
import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.trainer.multi_ckpt import setup_multi_checkpoint_manager
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.world import World, reset_world


@pytest.fixture(autouse=True, scope="module")
def init_process_group() -> Generator[None, None, None]:
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12357", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _create_run_with_ckpt_dirs(output_dir: Path, run_name: str, ckpt_steps: list[int]) -> Path:
    run_dir = output_dir / run_name
    run_dir.mkdir()
    control_dir = run_dir / "control"
    control_dir.mkdir()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    for step in ckpt_steps:
        (ckpt_dir / f"step_{step}").mkdir()
    config = {
        "model": {"name": "test-model"},
        "batch_size": 2,
        "rollouts_per_example": 1,
        "env": [{"id": "test-env"}],
        "sampling": {"temperature": 1.0},
        "ckpt": {"interval": 3, "keep_last": 1},
    }
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)
    return run_dir


def test_multi_ckpt_maybe_clean_keeps_ckpt_steps_in_sync_across_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: MultiCheckpointManager.maybe_clean() must update ckpt_steps
    on every rank, not just master. Otherwise non-master ranks retain stale entries
    and a later _should_save() call returns divergent values across ranks, causing
    the dist.barrier() inside save() to deadlock (master enters the save try block
    and waits on a barrier that non-master ranks - which took the `continue` path -
    never participate in).

    Repro pre-fix: master cleans step_440, master's ckpt_steps = [450].
    Non-master's maybe_clean returned early, ckpt_steps stayed [440, 450].
    Next attempt to save step 440 would diverge: True on master, False on non-master.
    """
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    manager = setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))

    _create_run_with_ckpt_dirs(tmp_path, "run_test", ckpt_steps=[440, 450])
    manager.discover_runs()
    run_idx = manager.id_2_idx["run_test"]

    multi_ckpt, _ = setup_multi_checkpoint_manager(tmp_path)
    # discover_runs ran before manager was constructed, so manually create the
    # underlying CheckpointManager for the test (mirrors the run-creation hook flow).
    multi_ckpt._run_creation_hook(run_idx, "run_test")
    inner = multi_ckpt.managers[run_idx]
    assert inner is not None
    assert inner.ckpt_steps == [440, 450]

    # Simulate a non-master rank. The bug only manifested on non-master ranks
    # because MultiCheckpointManager.maybe_clean returned early for them,
    # leaving the in-memory ckpt_steps list stale.
    monkeypatch.setattr(World, "is_master", property(lambda self: False))

    multi_ckpt.maybe_clean()

    # With keep_last=1, step 440 should be removed from ckpt_steps even on
    # non-master ranks (file deletion is master-gated inside CheckpointManager,
    # but the in-memory list update must happen on every rank).
    assert inner.ckpt_steps == [450], (
        f"non-master rank's ckpt_steps must be updated to match master. "
        f"Got {inner.ckpt_steps}, expected [450]. This divergence is what causes "
        f"the dist.barrier() deadlock inside MultiCheckpointManager.save()."
    )


def test_multi_ckpt_save_keeps_ckpt_steps_sorted(tmp_path: Path) -> None:
    """save() must insert sorted, not append. After maybe_clean trims away the
    resume step and leaves an orphan future-step (e.g. ckpt_steps=[453] when
    saving step 450), plain append would yield [453, 450] - violating the
    `assert list(ckpt_steps) == sorted(ckpt_steps)` invariant in maybe_clean
    and crashing the trainer on the next iteration.
    """
    import bisect

    from prime_rl.configs.trainer import CheckpointConfig
    from prime_rl.trainer.ckpt import CheckpointManager

    reset_world()
    runs._MULTI_RUN_MANAGER = None
    setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))

    run_dir = tmp_path / "run_orphan"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "step_453").mkdir()
    manager = CheckpointManager(run_dir, CheckpointConfig(interval=3, keep_last=1))
    assert manager.ckpt_steps == [453]

    # Mirror multi_ckpt.save's append path post-fix.
    bisect.insort(manager.ckpt_steps, 450)

    assert manager.ckpt_steps == [450, 453]
    assert list(manager.ckpt_steps) == sorted(manager.ckpt_steps)
