import json
from pathlib import Path

from prime_rl.sweep.metrics import read_final_summary, read_intermediate_metric


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_metrics_jsonl(run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_read_final_summary_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_final_summary(tmp_path / "missing", "val/loss") is None
    (tmp_path / "no_summary").mkdir()
    assert read_final_summary(tmp_path / "no_summary", "val/loss") is None


def test_read_final_summary_finds_value_under_run_subdir(tmp_path: Path) -> None:
    write_summary(tmp_path / "run-abc123" / "final_summary.json", {"val/loss": 0.42, "step": 100})
    assert read_final_summary(tmp_path, "val/loss") == 0.42


def test_read_final_summary_picks_latest_when_multiple_runs(tmp_path: Path) -> None:
    older = tmp_path / "run-old" / "final_summary.json"
    newer = tmp_path / "run-new" / "final_summary.json"
    write_summary(older, {"val/loss": 1.0})
    write_summary(newer, {"val/loss": 2.0})
    import os

    os.utime(older, (1.0, 1.0))
    os.utime(newer, (10.0, 10.0))

    assert read_final_summary(tmp_path, "val/loss") == 2.0


def test_read_final_summary_returns_none_for_non_scalar(tmp_path: Path) -> None:
    write_summary(tmp_path / "run-x" / "final_summary.json", {"val/loss": "nope", "flag": True})
    assert read_final_summary(tmp_path, "val/loss") is None
    assert read_final_summary(tmp_path, "flag") is None
    assert read_final_summary(tmp_path, "missing.key") is None


def test_read_final_summary_rejects_non_finite(tmp_path: Path) -> None:
    summary_path = tmp_path / "run-x" / "final_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('{"val/loss": NaN, "reward": Infinity, "neg": -Infinity, "ok": 1.5}')
    assert read_final_summary(tmp_path, "val/loss") is None
    assert read_final_summary(tmp_path, "reward") is None
    assert read_final_summary(tmp_path, "neg") is None
    assert read_final_summary(tmp_path, "ok") == 1.5


def test_read_final_summary_rejects_integer_float_overflow(tmp_path: Path) -> None:
    write_summary(tmp_path / "run-x" / "final_summary.json", {"reward": 10**1000})
    assert read_final_summary(tmp_path, "reward") is None


def test_read_final_summary_reads_latest_step_from_metrics_jsonl(tmp_path: Path) -> None:
    """metrics.jsonl is the canonical source: take the value at the largest step."""
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": 1, "reward": 0.1},
            {"step": 2, "reward": 0.4},
            {"step": 3, "reward": 0.3},
        ],
    )
    assert read_final_summary(tmp_path, "reward") == 0.3


def test_read_final_summary_uses_later_row_when_steps_tie(tmp_path: Path) -> None:
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": 2, "reward": 0.2},
            {"step": 2, "reward": 0.6},
        ],
    )
    assert read_final_summary(tmp_path, "reward") == 0.6


def test_read_final_summary_prefers_metrics_jsonl_over_legacy(tmp_path: Path) -> None:
    """If both exist, the sidecar wins; final_summary.json is fallback only."""
    write_metrics_jsonl(tmp_path, [{"step": 1, "reward": 0.9}])
    write_summary(tmp_path / "run-old" / "final_summary.json", {"reward": 0.1})
    assert read_final_summary(tmp_path, "reward") == 0.9


def test_read_final_summary_falls_back_when_metric_missing_from_sidecar(tmp_path: Path) -> None:
    """Sidecar without the metric falls through to final_summary.json."""
    write_metrics_jsonl(tmp_path, [{"step": 1, "loss": 1.5}])
    write_summary(tmp_path / "run-x" / "final_summary.json", {"reward": 0.7})
    assert read_final_summary(tmp_path, "reward") == 0.7


def test_read_final_summary_skips_malformed_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"step": 1, "reward": 0.1}) + "\n"
        + "garbage line\n"
        + json.dumps({"step": 2, "reward": 0.5}) + "\n"
    )
    assert read_final_summary(tmp_path, "reward") == 0.5


def test_read_final_summary_returns_none_for_malformed_legacy_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "run-x" / "final_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{not valid json\n")
    assert read_final_summary(tmp_path, "reward") is None

    summary_path.write_text(json.dumps(["not", "a", "mapping"]))
    assert read_final_summary(tmp_path, "reward") is None


def test_read_final_summary_ignores_malformed_step_for_ordering(tmp_path: Path) -> None:
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": 1, "reward": 0.1},
            {"step": "2", "reward": 0.9},
            {"step": 3, "reward": 0.5},
        ],
    )
    assert read_final_summary(tmp_path, "reward") == 0.5


def test_read_final_summary_ignores_rows_with_invalid_steps(tmp_path: Path) -> None:
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": "5", "reward": 0.9},
            {"step": True, "reward": 0.8},
            {"step": -1, "reward": 0.7},
        ],
    )

    assert read_final_summary(tmp_path, "reward") is None


def test_read_final_summary_falls_back_when_sidecar_steps_invalid(tmp_path: Path) -> None:
    write_metrics_jsonl(tmp_path, [{"step": -1, "reward": 0.9}])
    write_summary(tmp_path / "run-x" / "final_summary.json", {"reward": 0.4})

    assert read_final_summary(tmp_path, "reward") == 0.4


def test_read_intermediate_metric_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_intermediate_metric(tmp_path, "reward") is None
    write_metrics_jsonl(tmp_path, [{"step": 1, "loss": 1.5}])
    # Metric not present in any row.
    assert read_intermediate_metric(tmp_path, "reward") is None


def test_read_intermediate_metric_returns_latest_step_value(tmp_path: Path) -> None:
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": 1, "reward": 0.1},
            {"step": 5, "reward": 0.7},
            {"step": 3, "reward": 0.3},
        ],
    )
    assert read_intermediate_metric(tmp_path, "reward") == (5, 0.7)


def test_read_intermediate_metric_uses_later_row_when_steps_tie(tmp_path: Path) -> None:
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": 2, "reward": 0.2},
            {"step": 2, "reward": 0.6},
        ],
    )
    assert read_intermediate_metric(tmp_path, "reward") == (2, 0.6)


def test_read_intermediate_metric_rejects_non_finite_values(tmp_path: Path) -> None:
    """The polling loop must not feed NaN/Inf to optuna_trial.report()."""
    path = tmp_path / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"step": 1, "reward": NaN}\n{"step": 0, "reward": 0.5}\n')
    # Latest step (1) is NaN => filtered => returns None rather than the older
    # step's good value, since reporting an outdated step would be misleading.
    assert read_intermediate_metric(tmp_path, "reward") is None


def test_read_intermediate_metric_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"step": 1, "reward": 0.1}) + "\n"
        + "{not valid json\n"
        + json.dumps({"step": 2, "reward": 0.5}) + "\n"
        + "\n"  # blank line
    )
    assert read_intermediate_metric(tmp_path, "reward") == (2, 0.5)


def test_read_intermediate_metric_ignores_malformed_step_for_ordering(tmp_path: Path) -> None:
    write_metrics_jsonl(
        tmp_path,
        [
            {"step": "5", "reward": 0.9},
            {"step": 4, "reward": 0.7},
        ],
    )
    assert read_intermediate_metric(tmp_path, "reward") == (4, 0.7)


def test_read_intermediate_metric_rejects_negative_step(tmp_path: Path) -> None:
    write_metrics_jsonl(tmp_path, [{"step": -1, "reward": 0.9}])

    assert read_intermediate_metric(tmp_path, "reward") is None
