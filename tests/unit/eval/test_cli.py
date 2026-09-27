import pytest

from prime_rl.entrypoints.eval import expand_shorthands


def test_expand_shorthands_expands_taskset_and_concurrency() -> None:
    argv = ["gsm8k", "-n", "4", "--env.agent.harness.id", "bash", "-c", "8", "--run.name", "smoke"]
    assert expand_shorthands(argv) == [
        "--env.taskset.id",
        "gsm8k",
        "-n",
        "4",
        "--env.agent.harness.id",
        "bash",
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
    argv = ["@", toml.as_posix(), "--env.agent.harness.id", "bash"]
    assert expand_shorthands(argv) == argv


def test_expand_shorthands_requires_a_value() -> None:
    with pytest.raises(SystemExit, match="needs a value"):
        expand_shorthands(["gsm8k", "-c"])
    assert expand_shorthands(["-c=4"]) == ["--concurrency.min_inflight", "4", "--concurrency.max_inflight", "4"]
