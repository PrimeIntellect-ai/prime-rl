import json

import pytest
import verifiers.v1 as vf

from prime_rl.configs.eval import EvalConfig
from prime_rl.entrypoints.eval import expand_shorthands, resolve_rollout_targets


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


def test_expand_shorthands_requires_a_value() -> None:
    with pytest.raises(SystemExit, match="needs a value"):
        expand_shorthands(["gsm8k", "--env.agent.harness.id"])
    assert "--env.agent.max_turns" not in expand_shorthands(["gsm8k", "--env.agent.max-turns", "-1"])


def test_rollout_target_uses_selected_task_count(monkeypatch) -> None:
    loads = []

    class Taskset:
        INFINITE = False

        def __iter__(self):
            return iter(range(500))

    def load_taskset(config):
        loads.append(config.id)
        return Taskset()

    monkeypatch.setattr(vf, "load_taskset", load_taskset)
    config = EvalConfig.model_validate(
        {
            "min_rollouts_per_source": 1000,
            "source": [
                {"name": "all-bash", "env": {"taskset": {"id": "gsm8k"}}, "group_size": 34},
                {"name": "first-200", "env": {"taskset": {"id": "gsm8k"}}, "num_examples": 200},
                {"name": "first-30", "env": {"taskset": {"id": "gsm8k"}}, "num_examples": 30},
                {"name": "all-rlm", "env": {"taskset": {"id": "gsm8k"}}},
            ],
        }
    )

    resolve_rollout_targets(config)

    assert [source.group_size for source in config.source] == [2, 5, 34, 2]
    assert loads == ["gsm8k"] * 3


def test_rollout_target_requires_bound_for_infinite_taskset(monkeypatch) -> None:
    class Taskset:
        INFINITE = True

        def __iter__(self):
            return iter(range(500))

    monkeypatch.setattr(vf, "load_taskset", lambda config: Taskset())
    config = EvalConfig.model_validate(
        {"min_rollouts_per_source": 1000, "source": [{"env": {"taskset": {"id": "gsm8k"}}}]}
    )

    with pytest.raises(ValueError, match="infinite taskset needs num_examples"):
        resolve_rollout_targets(config)
