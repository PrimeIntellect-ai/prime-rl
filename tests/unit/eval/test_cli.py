import json

import pytest

from prime_rl.entrypoints.eval import expand_shorthands


def test_expand_shorthands_folds_taskset_into_a_source_and_env_into_the_group() -> None:
    argv = [
        "gsm8k",
        "-n",
        "4",
        "--env.agent.harness.id",
        "bash",
        "--env.agent.max-turns=5",
        "--env.taskset.tasks",
        '["fix-git"]',
        "-c",
        "8",
        "--run.name",
        "smoke",
    ]
    expanded = expand_shorthands(argv)
    assert json.loads(expanded[1]) == [{"env": {"taskset": {"id": "gsm8k"}}}]
    assert json.loads(expanded[-1]) == {
        "agent": {"harness": {"id": "bash"}, "max_turns": "5"},
        "taskset": {"tasks": ["fix-git"]},
    }
    assert expanded[0] == "--source"
    assert expanded[2:-2] == [
        "-n",
        "4",
        "--concurrency.min_inflight",
        "8",
        "--concurrency.max_inflight",
        "8",
        "--run.name",
        "smoke",
    ]
    assert expanded[-2] == "--env"


def test_expand_shorthands_passes_through_without_shorthands() -> None:
    argv = ["@", "eval.toml", "--model", "x", "--resume"]
    assert expand_shorthands(argv) == argv


def test_expand_shorthands_requires_a_value() -> None:
    with pytest.raises(SystemExit, match="needs a value"):
        expand_shorthands(["gsm8k", "--env.agent.harness.id"])
    assert "--env.agent.max_turns" not in expand_shorthands(["gsm8k", "--env.agent.max-turns", "-1"])
