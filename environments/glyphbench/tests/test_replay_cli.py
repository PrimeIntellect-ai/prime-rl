from __future__ import annotations

import argparse
import json

from glyphbench.cli import _cmd_replay


def test_replay_lists_pro_transcript_as_source_file(tmp_path, capsys) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"task_id": "glyphbench/minigrid-empty-5x5-v0", "seed": 42})
    )
    row = {
        "turn": 1,
        "observation": "[Grid]\n→",
        "action": "MOVE_FORWARD",
        "reward": 0.0,
        "total_return": 0.0,
    }
    (tmp_path / "transcript_0.jsonl").write_text(json.dumps(row) + "\n")
    args = argparse.Namespace(
        runs_dir=tmp_path,
        env=None,
        suite=None,
        model=None,
        seed=None,
        list=True,
        episode=None,
        pause=False,
        delay=0.0,
        reasoning_lines=None,
    )

    assert _cmd_replay(args) == 0
    output = capsys.readouterr().out
    assert "glyphbench/minigrid-empty-5x5-v0" in output
    assert "1 rollouts matched across 1 source file(s)." in output
