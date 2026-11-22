from __future__ import annotations

import itertools
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage
from verifiers.v1.utils.loaders import (
    environment_class,
    harness_class,
    load_taskset,
    taskset_class,
)

from glyphbench import make_env
from glyphbench.protocol import parse_action_response
from glyphbench.termination import is_context_limit_stop
from glyphbench.verifiers_harness import GlyphBenchHarness, _conversation
from glyphbench.verifiers_v1 import (
    GlyphBenchEnv,
    GlyphBenchEnvConfig,
    GlyphBenchTaskset,
    GlyphBenchTasksetConfig,
    _context_capacity_stats,
    rollout_seed,
)

EASY = "glyphbench/classics-snake-easy-v0"
MEDIUM = "glyphbench/classics-snake-medium-v0"
CONFIG_DIR = Path("configs/rl/qwen35-4b-glyphbench")


def test_plugin_exports_native_v1_taskset_and_environment() -> None:
    assert taskset_class("glyphbench") is GlyphBenchTaskset
    assert environment_class("glyphbench") is GlyphBenchEnv
    assert harness_class("glyphbench") is GlyphBenchHarness


def test_context_limit_classification_uses_framework_stop_not_exact_usage() -> None:
    assert is_context_limit_stop("context_length")
    assert is_context_limit_stop("prompt_too_long")
    assert not is_context_limit_stop("max_total_tokens")
    assert not is_context_limit_stop("is_done")


def test_environment_defaults_to_the_bundled_in_process_harness() -> None:
    config = GlyphBenchEnvConfig(
        taskset=GlyphBenchTasksetConfig(id="glyphbench", tasks=[EASY])
    )
    environment = GlyphBenchEnv(config)
    assert isinstance(environment._harnesses["player"], GlyphBenchHarness)


def test_harness_appends_turns_to_one_conversation_without_repeating_system() -> None:
    history = [
        SystemMessage(content="system-once"),
        UserMessage(content="observation-zero"),
        AssistantMessage(
            content="<action>RIGHT</action>",
            reasoning_content="reasoning-zero",
        ),
    ]
    trace = SimpleNamespace(branches=[SimpleNamespace(messages=history)])
    conversation = _conversation(
        None,  # type: ignore[arg-type]
        trace,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        [UserMessage(content="observation-one")],
    )

    assert sum(isinstance(message, SystemMessage) for message in conversation) == 1
    assert [message.content for message in conversation] == [
        "system-once",
        "observation-zero",
        "<action>RIGHT</action>",
        "observation-one",
    ]
    assert conversation[2].reasoning_content == "reasoning-zero"  # type: ignore[union-attr]


def test_taskset_balances_tasks_before_advancing_seed() -> None:
    config = GlyphBenchTasksetConfig(
        id="glyphbench",
        tasks=[EASY, MEDIUM],
        seed=42,
        num_seeds=2,
        mixed_seeds=False,
    )
    tasks = list(itertools.islice(load_taskset(config), 5))
    assert [(task.data.task_id, task.data.seed) for task in tasks] == [
        (EASY, 42),
        (MEDIUM, 42),
        (EASY, 43),
        (MEDIUM, 43),
        (EASY, 42),
    ]
    assert all(task.data.prompt is None for task in tasks)
    assert all(task.data.system_prompt for task in tasks)
    prompt = tasks[0].data.system_prompt or ""
    assert "Reason before acting" in prompt
    assert "replayed as part of the conversation on later turns" in prompt
    assert "Reply with exactly one XML action tag and no other text" not in prompt
    assert "for replay and memory" not in prompt


def test_production_manifest_is_100_tasks_with_all_60_craftax_tasks() -> None:
    config = GlyphBenchTasksetConfig(
        id="glyphbench",
        task_file=CONFIG_DIR / "multitask_craftax100_tasks.txt",
    )
    tasks = config.resolved_tasks()
    assert len(tasks) == 100
    assert sum("/craftax-" in task for task in tasks) == 60
    # Manifest order drives issuance order for task-homogeneous groups. Keep
    # the first half of the first 16x64 step suite-diverse and avoid long
    # Craftax-only bursts.
    assert {
        task.split("/", 1)[1].split("-", 1)[0] for task in tasks[:8]
    } == {"craftax", "classics", "miniatari", "minigrid", "minihack", "procgen"}
    assert max(
        len(list(run))
        for is_craftax, run in itertools.groupby(
            tasks,
            key=lambda task: "/craftax-" in task,
        )
        if is_craftax
    ) <= 2


def test_hero_manifest_is_100_unique_suite_interleaved_tasks() -> None:
    config = GlyphBenchTasksetConfig(
        id="glyphbench",
        task_file=CONFIG_DIR / "hero100_tasks.txt",
    )
    tasks = config.resolved_tasks()
    assert len(tasks) == len(set(tasks)) == 100
    suites = [task.split("/", 1)[1].split("-", 1)[0] for task in tasks]
    assert {suite: suites.count(suite) for suite in set(suites)} == {
        "classics": 18,
        "craftax": 18,
        "miniatari": 18,
        "minigrid": 18,
        "minihack": 19,
        "procgen": 9,
    }
    assert suites[:6] == [
        "classics",
        "craftax",
        "miniatari",
        "minigrid",
        "minihack",
        "procgen",
    ]


def test_pilot_and_periodic_eval_manifest_sizes() -> None:
    pilot = GlyphBenchTasksetConfig(
        id="glyphbench", task_file=CONFIG_DIR / "multitask_pilot_tasks.txt"
    )
    evaluation = GlyphBenchTasksetConfig(
        id="glyphbench", task_file=CONFIG_DIR / "multitask_fixed_eval_tasks.txt"
    )
    assert len(pilot.resolved_tasks()) == 8
    assert len(evaluation.resolved_tasks()) == 50


def test_production_configs_do_not_export_raw_inference_metrics() -> None:
    expected = {
        "single_task.toml": (50, 65_536),
        "multitask.toml": (1_000, 65_536),
    }
    for name, (max_steps, seq_len) in expected.items():
        with (CONFIG_DIR / name).open("rb") as config_file:
            config = tomllib.load(config_file)
        assert config["max_steps"] == max_steps
        assert config["seq_len"] == seq_len
        assert config["inference"]["vllm"]["max_model_len"] == seq_len
        assert (
            config["orchestrator"]["train"]["sampling"]["max_completion_tokens"]
            == 4_096
        )
        expected_ac = (
            {"mode": "full", "freq": 1}
            if name == "multitask.toml"
            else {
                "mode": "selective",
                "freq": 1,
                "targets": ["norm", "attn_proj", "mlp", "linear_attn"],
            }
        )
        assert config["trainer"]["model"]["ac"] == expected_ac
        assert config["orchestrator"]["collect_inference_metrics"] is False
        assert config["orchestrator"]["renderer"] == {
            "name": "qwen3.5",
            "enable_thinking": True,
            "thinking_retention": "all",
        }
        assert config["orchestrator"]["max_off_policy_steps"] <= 3
        assert config["slurm"]["cleanup_grace_period"] == 120
        for phase in ("train", "eval"):
            for source in config["orchestrator"][phase]["source"]:
                assert "n_frames" not in source["env"]["taskset"]
                player = source["env"]["player"]
                assert "max_turns" not in player
                assert "max_input_tokens" not in player
                assert "max_output_tokens" not in player
                if name == "multitask.toml":
                    assert player["max_total_tokens"] == 65_535
                else:
                    assert "max_total_tokens" not in player
        assert "file" not in config["monitors"]
        assert config["trainer"]["monitors"]["file"] == "None"
        assert config["orchestrator"]["monitors"]["file"] == "None"

    with (CONFIG_DIR / "multitask.toml").open("rb") as config_file:
        multitask = tomllib.load(config_file)
    assert multitask["deployment"]["num_infer_nodes"] == 3
    assert multitask["deployment"]["num_train_nodes"] == 1
    assert "64k-context" in multitask["monitors"]["wandb"]["tags"]
    assert "full-activation-checkpointing" in multitask["monitors"]["wandb"]["tags"]
    assert multitask["orchestrator"]["group_size"] == 64
    assert multitask["orchestrator"]["batch_size"] == 1_024
    assert multitask["env_vars"]["HF_HUB_OFFLINE"] == "1"
    assert multitask["env_vars"]["TRANSFORMERS_OFFLINE"] == "1"
    for phase in ("train", "eval"):
        assert all(
            source["env"]["player"]["max_total_tokens"] == 65_535
            for source in multitask["orchestrator"][phase]["source"]
        )
    assert multitask["ckpt"] == {
        "interval": 25,
        "keep_last": 2,
        "keep_interval": 250,
    }


def test_multitask_eval_panel_is_fixed_and_matches_manifest() -> None:
    with (CONFIG_DIR / "multitask.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
    sources = config["orchestrator"]["eval"]["source"]
    assert len(sources) == 1
    configured_manifest = Path(sources[0]["env"]["taskset"]["task_file"])
    manifest = GlyphBenchTasksetConfig(
        id="glyphbench",
        task_file=CONFIG_DIR / "multitask_fixed_eval_tasks.txt",
    ).resolved_tasks()
    assert configured_manifest == CONFIG_DIR / "multitask_fixed_eval_tasks.txt"
    assert len(manifest) == 50
    assert set(manifest) < set(
        GlyphBenchTasksetConfig(
            id="glyphbench",
            task_file=CONFIG_DIR / "multitask_craftax100_tasks.txt",
        ).resolved_tasks()
    )
    assert config["orchestrator"]["eval"]["interval"] == 50
    assert config["orchestrator"]["eval"]["num_examples"] == 50 * 4
    assert all(source["env"]["taskset"]["mixed_seeds"] is False for source in sources)
    assert all(source["env"]["taskset"]["seed"] == 1_000_000 for source in sources)
    assert all(source["env"]["taskset"]["num_seeds"] == 4 for source in sources)


def test_v1_training_contract_forbids_memory_mode() -> None:
    with pytest.raises(ValidationError):
        GlyphBenchTasksetConfig(id="glyphbench", tasks=[EASY], use_memory=True)


def test_v1_training_contract_has_no_second_frame_history_channel() -> None:
    with pytest.raises(ValidationError):
        GlyphBenchTasksetConfig(  # type: ignore[call-arg]
            id="glyphbench",
            tasks=[EASY],
            n_frames=0,
        )


def test_mixed_seed_is_trace_specific_and_fixed_eval_is_exact() -> None:
    assert rollout_seed(42, "trace-a", mixed=False) == 42
    first = rollout_seed(42, "trace-a", mixed=True)
    assert first == rollout_seed(42, "trace-a", mixed=True)
    assert first != rollout_seed(42, "trace-b", mixed=True)


def test_context_capacity_logging_counts_the_pre_engine_guard_as_truncation() -> None:
    below = SimpleNamespace(
        branches=[SimpleNamespace(num_total_tokens=65_534)],
        stop_condition=None,
    )
    at_guard = SimpleNamespace(
        branches=[SimpleNamespace(num_total_tokens=65_535)],
        stop_condition="max_total_tokens",
    )
    refused = SimpleNamespace(
        branches=[SimpleNamespace(num_total_tokens=65_000)],
        stop_condition="context_length",
    )

    assert _context_capacity_stats(below, 65_535)[2] is False
    assert _context_capacity_stats(at_guard, 65_535) == (
        65_535,
        pytest.approx(65_535 / 65_536),
        True,
    )
    assert _context_capacity_stats(refused, 65_535)[2] is True


def test_protocol_parser_drives_the_public_action_spec() -> None:
    game = make_env(EASY)
    try:
        parsed = parse_action_response(
            "consider left\n<action>UP</action>",
            game.action_spec,
            noop=game.noop_action_name,
        )
        assert not parsed.failed
        assert parsed.name == "UP"

        failed = parse_action_response(
            "not an action",
            game.action_spec,
            noop=game.noop_action_name,
        )
        assert failed.failed
        assert 0 <= failed.index < game.action_spec.n
    finally:
        game.close()
