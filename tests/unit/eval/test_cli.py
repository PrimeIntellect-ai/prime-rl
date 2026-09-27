import json

import pytest

from prime_rl.entrypoints.eval import expand_shorthands


def test_expand_shorthands_folds_taskset_and_env_into_the_env_block() -> None:
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
    assert json.loads(expanded[expanded.index("--env") + 1]) == {
        "taskset": {"id": "gsm8k", "tasks": ["fix-git"]},
        "agent": {"harness": {"id": "bash"}, "max_turns": "5"},
    }
    assert expanded[: expanded.index("--env")] == [
        "-n",
        "4",
        "--concurrency.min_inflight",
        "8",
        "--concurrency.max_inflight",
        "8",
        "--run.name",
        "smoke",
    ]


def test_expand_shorthands_passes_through_without_shorthands() -> None:
    argv = ["@", "eval.toml", "--model", "x", "--resume"]
    assert expand_shorthands(argv) == argv


def test_expand_shorthands_refuses_taskset_next_to_source_toml(tmp_path) -> None:
    toml = tmp_path / "eval.toml"
    toml.write_text('[[source]]\nenv.taskset.id = "gsm8k"\n')
    with pytest.raises(SystemExit, match="cannot be combined"):
        expand_shorthands(["wordle", "@", toml.as_posix()])
    assert expand_shorthands(["@", toml.as_posix(), "--env.agent.harness.id", "bash"])[-2] == "--env"


def test_expand_shorthands_requires_a_value() -> None:
    with pytest.raises(SystemExit, match="needs a value"):
        expand_shorthands(["gsm8k", "--env.agent.harness.id"])
    assert "--env.agent.max_turns" not in expand_shorthands(["gsm8k", "--env.agent.max-turns", "-1"])
