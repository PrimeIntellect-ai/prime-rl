import json

import pytest

from tools.ngu_difficulty import bucket_for, count_outcomes, count_solves, split


def episode(task, index, score, *, ok=True):
    return {
        "id": f"{task}-{index}",
        "ok": ok,
        "env": {"name": "swerebench-1k"},
        "task": {"data": {"name": task}, "hash": f"hash-{task}"},
        "traces": [{"rewards": {"solved": {"score": score, "weight": 1}}}],
    }


def test_difficulty_boundaries():
    assert [bucket_for(n) for n in range(9)] == [
        "extra-hard",
        "hard",
        "hard",
        "medium",
        "medium",
        "medium",
        "easy",
        "easy",
        "easy",
    ]
    assert bucket_for(3, 4) == "easy"
    assert bucket_for(2, 4) == "medium"
    assert bucket_for(1, 4) == "hard"
    assert bucket_for(0, 4) == "extra-hard"
    with pytest.raises(ValueError):
        bucket_for(0, 0)
    with pytest.raises(ValueError):
        bucket_for(9)


def test_resume_duplicates_and_errors_do_not_change_counts():
    rows = [episode("a", n, int(n < 6)) for n in range(8)]
    assert count_solves([*rows, *rows, episode("a", 9, 0, ok=False)], ["a"]) == {"a": 6}
    with pytest.raises(ValueError, match="exactly 8"):
        count_solves(rows[:7] + [episode("a", 9, 0, ok=False)], ["a"])
    with pytest.raises(ValueError, match="Conflicting"):
        count_solves(rows + [episode("a", 0, 0)], ["a"])
    with pytest.raises(ValueError, match="exactly 8"):
        count_solves(rows + [episode("a", 9, 0)], ["a"])


def test_split_is_disjoint_exhaustive_and_preserves_eval(tmp_path):
    import tomllib

    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({"task_ids": ["a", "b", "c", "d"], "revision": "fixed"}))
    stream = tmp_path / "run/monitors/file/traces/stream"
    stream.mkdir(parents=True)
    rows = [episode(task, n, int(n < solves)) for task, solves in zip("abcd", [8, 4, 2, 0]) for n in range(8)]
    (stream / "00000.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "split"
    split(manifest, tmp_path / "run", output)
    result = json.loads((output / "results.json").read_text())
    assert result["bucket_sizes"] == {"easy": 1, "medium": 1, "hard": 1, "extra-hard": 1}
    assert result["avg_at_8"] == 14 / 32
    ids = [json.loads((output / f"{bucket}.json").read_text())["task_ids"][0] for bucket in result["bucket_sizes"]]
    assert ids == list("abcd")
    overlay = tomllib.loads((output / "online-eval.toml").read_text())
    assert len(overlay["orchestrator"]["eval"]["source"]) == 5
    assert overlay["orchestrator"]["eval"]["source"][0]["name"] == "swebench-verified"


def test_partial_split_uses_valid_denominators(tmp_path):
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({"task_ids": ["a", "b", "c"], "revision": "fixed"}))
    stream = tmp_path / "run/monitors/file/traces/stream"
    stream.mkdir(parents=True)
    rows = [episode("a", n, int(n < 3)) for n in range(4)] + [episode("b", 0, 0), episode("c", 0, 0, ok=False)]
    (stream / "00000.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    split(manifest, tmp_path / "run", tmp_path / "split", allow_partial=True)
    result = json.loads((tmp_path / "split/results.json").read_text())
    assert result["task_mean_pass_rate"] == 0.375
    assert result["valid_attempt_pass_rate"] == 0.6
    assert result["unclassified_tasks"] == ["c"]
    assert result["bucket_sizes"] == {"easy": 1, "medium": 0, "hard": 0, "extra-hard": 1}
    assert result["avg_at_8"] is None
    with pytest.raises(ValueError, match="More than 8"):
        count_outcomes([episode("a", n, 1) for n in range(9)], ["a"], allow_partial=True)
