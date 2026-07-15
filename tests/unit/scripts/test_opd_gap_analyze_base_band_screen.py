import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts" / "opd_gap_analyze_base_band_screen.py"
SPEC = importlib.util.spec_from_file_location("base_band_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(task, reward, start, *, errors=None, stop="agent_completed"):
    return {
        "task": {"name": task},
        "rewards": {"solved": reward},
        "timing": {"start": start},
        "errors": errors or [],
        "stop_condition": stop,
        "nodes": [],
    }


def manifest():
    return {
        "rollouts_per_task": 16,
        "group_size": 8,
        "bands": [{"name": "hard"}],
        "tasks": [{"task": "task_a", "tier": 4, "metadata_pass_rate": 0.2, "band": "hard"}],
    }


def test_observed_groups_drive_eligibility():
    rewards = [0, 1] * 8
    rows = [row("task_a", reward, start) for start, reward in enumerate(rewards)]
    summary = MODULE.summarize_backend(rows, manifest())
    task = summary["tasks"][0]
    assert task["mixed_groups"] == 2
    assert task["all_fail_groups"] == 0
    assert task["all_pass_groups"] == 0
    assert task["eligible_for_group_relative_rl"]


def test_errors_disqualify_otherwise_mixed_task():
    rewards = [0, 1] * 8
    rows = [row("task_a", reward, start) for start, reward in enumerate(rewards)]
    rows[-1]["errors"] = [{"type": "ProviderError"}]
    summary = MODULE.summarize_backend(rows, manifest())
    assert not summary["tasks"][0]["eligible_for_group_relative_rl"]


def test_truncation_is_a_preference_not_reward_variance_disqualification():
    rewards = [0, 1] * 8
    rows = [row("task_a", reward, start) for start, reward in enumerate(rewards)]
    for trace in rows[:5]:
        trace["stop_condition"] = "harness_timeout"
    task = MODULE.summarize_backend(rows, manifest(), max_preferred_truncation_rate=0.25)["tasks"][0]
    assert task["eligible_for_group_relative_rl"]
    assert not task["preferred_for_training"]


def test_all_fail_and_all_pass_groups_are_counted():
    rows = [row("task_a", 0, start) for start in range(8)]
    rows += [row("task_a", 1, start) for start in range(8, 16)]
    task = MODULE.summarize_backend(rows, manifest())["tasks"][0]
    assert task["all_fail_groups"] == 1
    assert task["all_pass_groups"] == 1
    assert task["mixed_groups"] == 0


def test_uniform_partial_reward_is_not_group_variance():
    rows = [row("task_a", 0.8, start) for start in range(16)]
    task = MODULE.summarize_backend(rows, manifest())["tasks"][0]
    assert task["uniform_partial_groups"] == 2
    assert task["mixed_groups"] == 0
    assert not task["eligible_for_group_relative_rl"]


def test_manifest_can_require_one_mixed_group_for_pass_at_8_screen():
    one_group_manifest = manifest()
    one_group_manifest["rollouts_per_task"] = 8
    one_group_manifest["minimum_mixed_groups"] = 1
    rows = [row("task_a", reward, start) for start, reward in enumerate([0, 1] * 4)]
    task = MODULE.summarize_backend(rows, one_group_manifest)["tasks"][0]
    assert task["eligible_for_group_relative_rl"]
