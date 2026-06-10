from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from prime_rl.configs.orchestrator import CheckpointConfig  # noqa: E402
from prime_rl.orchestrator.ckpt import CheckpointManager  # noqa: E402
from prime_rl.orchestrator.multi_agent_advantage import RAEState  # noqa: E402
from prime_rl.orchestrator.types import Progress  # noqa: E402


def test_rae_state_round_trips(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    progress = Progress(step=7)
    rae_state = RAEState(
        baselines={
            ("debate_v1", 1): 0.3,
            ("debate_v1", "ex-2"): -0.1,
        },
        canonical_members={
            ("debate_v1", 1): "debater_a",
            ("debate_v1", "ex-2"): "debater_a",
        },
        beta=0.85,
        n_eff=4.0,
    )

    mgr.save(progress, step=7, rae_state=rae_state)
    progress_reload = Progress()
    rae_reload = RAEState(beta=0.5, n_eff=0.0)
    mgr.load(progress_reload, step=7, rae_state=rae_reload)

    assert progress_reload.step == 7
    assert rae_reload.beta == 0.85
    assert rae_reload.n_eff == 4.0
    assert rae_reload.baselines == rae_state.baselines
    assert rae_reload.canonical_members == rae_state.canonical_members


def test_old_format_rae_state_fails_loud(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    mgr.save(Progress(step=2), step=2, rae_state=None)
    # Retired sequential-EMA format: per-member triple keys + momentum
    with open(mgr.get_ckpt_path(2) / "rae_state.pt", "wb") as f:
        torch.save({"baselines": {("debate_v1", 1, "debater_a"): 0.1}, "momentum": 0.9}, f)

    with pytest.raises(ValueError, match="retired sequential-EMA format"):
        mgr.load(Progress(), step=2, rae_state=RAEState())


def test_rae_state_load_fails_loud_when_missing(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    mgr.save(Progress(step=3), step=3, rae_state=None)

    with pytest.raises(FileNotFoundError, match="RAE state not found"):
        mgr.load(Progress(), step=3, rae_state=RAEState())


def test_save_without_rae_state_omits_file(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    mgr.save(Progress(step=1), step=1, rae_state=None)

    assert not (mgr.get_ckpt_path(1) / "rae_state.pt").exists()
