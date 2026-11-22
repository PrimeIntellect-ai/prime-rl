from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "eval" / "reasoning_gym" / "reasoning_gym_eval.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("reasoning_gym_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocols_contain_no_machine_or_account_specific_values() -> None:
    for name in ("protocol.json", "protocol_8192.json"):
        path = SCRIPT.with_name(name)
        protocol = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = protocol["models"]["checkpoint"]
        assert not Path(checkpoint["model"]).is_absolute()
        assert "wandb_entity" not in protocol["reporting"]
        assert "/home/" not in path.read_text(encoding="utf-8")


def test_wandb_publication_is_opt_in() -> None:
    evaluator = load_evaluator()
    run = evaluator.parse_args(["run", "--model", "Qwen/Qwen3.5-4B", "--label", "base"])
    compare = evaluator.parse_args(
        ["compare", "--baseline", "before", "--checkpoint", "after", "--output", "out"]
    )
    assert run.publish_wandb is False
    assert compare.publish_wandb is False


def test_protocol_hash_uses_only_public_evaluator_files() -> None:
    evaluator = load_evaluator()
    assert len(evaluator.protocol_hash()) == 64


def test_protocol_override_can_live_outside_repository(tmp_path, monkeypatch) -> None:
    evaluator = load_evaluator()
    custom = tmp_path / "custom-protocol.json"
    custom.write_text('{}')
    monkeypatch.setattr(evaluator, "PROTOCOL_PATH", custom)
    assert len(evaluator.protocol_hash()) == 64
