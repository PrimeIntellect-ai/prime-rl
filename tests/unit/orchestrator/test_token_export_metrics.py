import json
from pathlib import Path

from prime_rl.orchestrator.token_export_metrics import (
    collect_next_token_export_metrics,
    collect_token_export_metrics,
)


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def test_collect_token_export_metrics_requires_stable_marker(tmp_path: Path):
    _write_record(
        tmp_path / "token_exports" / "step_0" / "rank_0.jsonl",
        {
            "env_name": "reverse_text",
            "loss_mask": [True],
            "mismatch_kl": [0.5],
            "entropy": [1.5],
        },
    )

    assert collect_token_export_metrics(tmp_path, 0) == {}


def test_collect_token_export_metrics_aggregates_loss_tokens(tmp_path: Path):
    step_dir = tmp_path / "token_exports" / "step_3"
    _write_record(
        step_dir / "rank_0.jsonl",
        {
            "env_name": "reverse_text",
            "loss_mask": [True, False, True],
            "mismatch_kl": [0.5, 100.0, 1.5],
            "entropy": [2.0, 200.0, 4.0],
        },
    )
    _write_record(
        step_dir / "rank_1.jsonl",
        {
            "env_name": "math",
            "loss_mask": [True, True],
            "mismatch_kl": [None, 2.0],
            "entropy": [6.0, 8.0],
        },
    )
    (step_dir / "STABLE").touch()

    metrics = collect_token_export_metrics(tmp_path, 3)

    assert metrics == {
        "entropy/max": 8.0,
        "entropy/mean": 5.0,
        "mismatch_kl/max": 2.0,
        "mismatch_kl/mean": 4.0 / 3.0,
    }


def test_collect_next_token_export_metrics_returns_oldest_unlogged_stable_step(tmp_path: Path):
    for step in [1, 2]:
        step_dir = tmp_path / "token_exports" / f"step_{step}"
        _write_record(
            step_dir / "rank_0.jsonl",
            {
                "env_name": "reverse_text",
                "loss_mask": [True],
                "mismatch_kl": [float(step)],
                "entropy": [float(step + 10)],
            },
        )
        (step_dir / "STABLE").touch()

    result = collect_next_token_export_metrics(tmp_path, last_logged_step=0, max_step=2)

    assert result is not None
    assert result.step == 1
    assert result.metrics["mismatch_kl/mean"] == 1.0
