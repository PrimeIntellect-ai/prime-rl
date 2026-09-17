import json

import pytest

from tools.ngu_difficulty import bucket_for, count_solves, split


def episode(task, index, score, *, ok=True):
    return {
        "id": f"{task}-{index}",
        "ok": ok,
        "env": {"name": "swerebench-profile"},
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
