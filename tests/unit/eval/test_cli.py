import json
from unittest.mock import Mock

import pytest

from prime_rl.entrypoints.eval import expand_shorthands


@pytest.mark.parametrize("configured", [False, True])
def test_gsm8k_eval_requires_credentials(monkeypatch, tmp_path, configured) -> None:
    from tests.integration import test_gsm8k_eval

    monkeypatch.setattr(test_gsm8k_eval, "resolve_api_key", lambda config: "test-key" if configured else "EMPTY")
    run_process = Mock(return_value=object())
    if configured:
        assert test_gsm8k_eval.eval_process.__wrapped__(run_process, tmp_path) is run_process.return_value
        run_process.assert_called_once()
    else:
        with pytest.raises(pytest.skip.Exception, match="PRIME_API_KEY"):
            test_gsm8k_eval.eval_process.__wrapped__(run_process, tmp_path)
        run_process.assert_not_called()


def test_expand_shorthands_folds_taskset_and_env_into_one_source() -> None:
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
    source = json.loads(expanded[expanded.index("--source") + 1])
    assert source == [
        {
            "env": {
                "taskset": {"id": "gsm8k", "tasks": ["fix-git"]},
                "agent": {"harness": {"id": "bash"}, "max_turns": "5"},
            }
        }
    ]
    assert expanded[: expanded.index("--source")] == [
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


def test_expand_shorthands_refuses_shorthand_next_to_source_toml(tmp_path) -> None:
    toml = tmp_path / "eval.toml"
    toml.write_text('[[source]]\nenv.taskset.id = "gsm8k"\n')
    with pytest.raises(SystemExit, match="cannot be combined"):
        expand_shorthands(["wordle", "@", toml.as_posix()])
    with pytest.raises(SystemExit, match="cannot be combined"):
        expand_shorthands(["wordle", "@", toml.as_posix(), "--env.retries.max-retries", "3"])


def test_expand_shorthands_passes_env_flags_to_the_shared_block_next_to_source_toml(tmp_path) -> None:
    toml = tmp_path / "eval.toml"
    toml.write_text('[[source]]\nenv.taskset.id = "gsm8k"\n')
    argv = ["@", toml.as_posix(), "--env.retries.max-retries", "3", "--env.timeout.episode=60"]
    assert expand_shorthands(argv) == [
        "@",
        toml.as_posix(),
        "--env.retries.max-retries",
        "3",
        "--env.timeout.episode",
        "60",
    ]


def test_expand_shorthands_requires_a_value() -> None:
    with pytest.raises(SystemExit, match="needs a value"):
        expand_shorthands(["gsm8k", "--env.agent.harness.id"])
    assert "--env.agent.max_turns" not in expand_shorthands(["gsm8k", "--env.agent.max-turns", "-1"])
