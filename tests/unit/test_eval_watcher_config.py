from pathlib import Path

import pytest
from pydantic import ValidationError

from prime_rl.eval.config import OfflineEvalConfig


def test_offline_eval_config_steps_requires_existing_when_not_watching(tmp_path: Path):
    (tmp_path / "step_1").mkdir()
    with pytest.raises(ValidationError) as excinfo:
        OfflineEvalConfig(weights_dir=tmp_path, steps=[2], watcher=False)
    assert "Step 2 not found in weights directory" in str(excinfo.value)


def test_offline_eval_config_steps_can_be_future_when_watching(tmp_path: Path):
    (tmp_path / "step_1").mkdir()
    cfg = OfflineEvalConfig(weights_dir=tmp_path, steps=[2], watcher=True)
    assert cfg.watcher is True
