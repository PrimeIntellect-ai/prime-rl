from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from datasets import Dataset  # noqa: E402

from prime_rl.configs.orchestrator import BufferConfig, CheckpointConfig  # noqa: E402
from prime_rl.orchestrator.buffer import Buffer  # noqa: E402
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress  # noqa: E402
from prime_rl.orchestrator.multi_agent_advantage import RAEState  # noqa: E402


class _OneEnvCollection:
    names = ["dummy"]

    def __init__(self):
        self.env = SimpleNamespace(
            name="dummy",
            config=SimpleNamespace(ratio=None),
            get_dataset=lambda seed=None: Dataset.from_dict(
                {
                    "prompt": [[{"role": "user", "content": "q"}]],
                    "answer": ["a"],
                    "task": ["dummy"],
                    "info": [{}],
                    "example_id": [0],
                }
            ),
        )

    def __iter__(self):
        return iter([self.env])


def _buffer() -> Buffer:
    return Buffer(_OneEnvCollection(), BufferConfig())


def test_rae_state_round_trips(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    progress = Progress(step=7)
    buffer = _buffer()
    rae_state = RAEState(
        baselines={
            ("debate_v1", 1, "debater_a"): 0.3,
            ("debate_v1", 1, "debater_b"): -0.1,
        },
        momentum=0.85,
    )

    mgr.save(progress, buffer, step=7, rae_state=rae_state)
    progress_reload = Progress()
    rae_reload = RAEState(momentum=0.0)
    mgr.load(progress_reload, buffer, step=7, rae_state=rae_reload)

    assert progress_reload.step == 7
    assert rae_reload.momentum == 0.85
    assert rae_reload.baselines == rae_state.baselines


def test_rae_state_load_fails_loud_when_missing(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    buffer = _buffer()
    mgr.save(Progress(step=3), buffer, step=3, rae_state=None)

    with pytest.raises(FileNotFoundError, match="RAE state not found"):
        mgr.load(Progress(), buffer, step=3, rae_state=RAEState())


def test_save_without_rae_state_omits_file(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "run", CheckpointConfig(interval=1))
    mgr.save(Progress(step=1), _buffer(), step=1, rae_state=None)

    assert not (mgr.get_ckpt_path(1) / "rae_state.pt").exists()
