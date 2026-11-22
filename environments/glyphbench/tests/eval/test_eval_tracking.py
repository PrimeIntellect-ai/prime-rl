from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from glyphbench.eval_tracking import load_pro_episodes, main


@pytest.mark.parametrize(
    "task_ids,override,expected_task,expected_suites",
    [
        (["glyphbench/minigrid-empty-5x5-v0"], None,
         "glyphbench/minigrid-empty-5x5-v0", {"minigrid"}),
        (["glyphbench/procgen-maze-v0", "glyphbench/minigrid-empty-5x5-v0"], None,
         ["glyphbench/minigrid-empty-5x5-v0", "glyphbench/procgen-maze-v0"],
         {"minigrid", "procgen"}),
        (["glyphbench/minigrid-empty-5x5-v0"], "glyphbench/craftaxfull-v0",
         "glyphbench/craftaxfull-v0", {"craftaxfull"}),
    ],
)
def test_sync_labels_match_results_or_explicit_override(
    tmp_path: Path, monkeypatch, task_ids, override, expected_task, expected_suites,
) -> None:
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps({"info": {"env_id": task_id}, "reward": 0.5})
                  for task_id in task_ids),
        encoding="utf-8",
    )
    captured = {}
    logs = []
    finished = []
    run = SimpleNamespace(
        url="https://wandb.example/run", summary={}, log=logs.append,
        finish=lambda: finished.append(True),
    )

    def init(**kwargs):
        captured.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, Settings=lambda **kwargs: kwargs,
    ))
    args = [
        "--results-root", str(tmp_path), "--harness", "normal",
        "--model", "test-model", "--run-name", "test-run",
    ]
    if override:
        args.extend(["--task", override])

    assert main(args) == 0

    assert captured["config"]["task"] == expected_task
    assert set(captured["tags"]) & {"minigrid", "procgen", "craftaxfull"} == expected_suites
    assert str(tmp_path) not in json.dumps(captured["config"])
    assert [row["episode/task"] for row in logs[:-1]] == task_ids
    assert finished == [True]


def test_pro_results_without_config_do_not_invent_a_task(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps([{"episode": 0, "seed": 42, "episode_return": 0.5}]),
        encoding="utf-8",
    )

    assert load_pro_episodes(tmp_path)[0]["task"] == "glyphbench"
