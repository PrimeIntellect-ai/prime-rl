"""End-to-end smoke test of the Pro harness loop against the real CraftaxFull
env, using a scripted fake client (no network)."""

from __future__ import annotations

import importlib.util
import json

import pytest

from glyphbench.core.registry import make_env
from glyphbench.envs import _import_all_suites
from glyphbench.pro_harness import ProConfig, ProHarness
from glyphbench.pro_harness.adapters import get_adapter
from glyphbench.pro_harness.clients import CompletionResult, LLMClient
from glyphbench.pro_harness.navigation import ExplorationTracker

_import_all_suites()


class _FakeClient(LLMClient):
    """Scripted client: valid actions on action turns, a memory block on
    memory turns. Optionally make every action unparseable to test forfeits."""

    def __init__(self, actions, *, break_actions=False, memory_text=None):
        super().__init__(ProConfig(task_id="glyphbench/craftaxfull-v0"))
        self.actions = list(actions)
        self.i = 0
        self.break_actions = break_actions
        self.memory_text = memory_text or (
            "<memory>\nTACTICAL: exploring east toward water.\n"
            "MAP_ADD: area=0, type=water, row=1, col=1, note=pool\n</memory>"
        )
        self.labels: list[str] = []
        self.users: list[object] = []

    def describe(self) -> str:
        return "fake"

    def _attempt(self, system, user, *, drop_sampling):  # pragma: no cover
        raise NotImplementedError

    def complete(self, system, user, *, label=""):
        self.labels.append(label)
        self.users.append(user)
        if label.startswith("action"):
            if self.break_actions:
                return CompletionResult(
                    text="hmm, not sure what to do here.",
                    reasoning="think",
                    finish_reason="stop",
                    input_tokens=10,
                    output_tokens=5,
                )
            a = self.actions[self.i % len(self.actions)]
            self.i += 1
            return CompletionResult(
                text=f"Looking at the grid, I will act.\n<action>{a}</action>",
                reasoning="some reasoning",
                finish_reason="stop",
                input_tokens=10,
                output_tokens=5,
            )
        return CompletionResult(
            text=self.memory_text, finish_reason="stop", input_tokens=8, output_tokens=4
        )


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_full_loop_runs_and_records():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0", max_turns=6, save_results=False, verbose=False
    )
    env = make_env(cfg.task_id, max_turns=6)
    client = _FakeClient(["MOVE_RIGHT", "MOVE_DOWN", "DO", "MOVE_LEFT", "MOVE_UP", "DRINK_WATER"])
    h = ProHarness(cfg, client=client)
    res, interactions = h.run_episode(env, get_adapter(cfg.task_id), seed=42, episode_idx=0)

    assert res.steps >= 1
    assert res.forfeits == 0
    assert res.num_action_turns >= 1
    assert res.num_memory_turns >= 1
    assert isinstance(res.episode_return, float)
    roles = {i["role"] for i in interactions}
    assert "action" in roles and "memory" in roles
    # Memory turns added landmarks.
    assert any(i.get("landmark_count", 0) >= 1 for i in interactions)
    # Both turn types alternate: action then memory.
    assert client.labels[0] == "action"


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_normal_harness_uses_one_call_per_environment_turn():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        harness_variant="normal",
        max_turns=4,
        save_results=False,
        verbose=False,
    )
    env = make_env(cfg.task_id, max_turns=4)
    client = _FakeClient(["MOVE_RIGHT", "MOVE_DOWN", "DO", "MOVE_LEFT"])
    result, interactions = ProHarness(cfg, client=client).run_episode(
        env, get_adapter(cfg.task_id), seed=42, episode_idx=0
    )

    assert result.steps >= 1
    assert result.num_memory_turns == 0
    assert result.num_action_turns == result.steps
    assert set(client.labels) == {"action"}
    assert {row["role"] for row in interactions} == {"action"}


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_forfeit_path_and_feedback():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        max_turns=3,
        save_results=False,
        verbose=False,
        parse_retries=0,
        forfeit_mode="noop",
    )
    env = make_env(cfg.task_id, max_turns=3)
    client = _FakeClient([], break_actions=True)
    h = ProHarness(cfg, client=client)
    res, interactions = h.run_episode(env, get_adapter(cfg.task_id), seed=1, episode_idx=0)
    assert res.forfeits >= 1
    assert res.parse_failures >= 1
    # FORFEIT recorded in the action interactions.
    assert any(i["role"] == "action" and i["action"] == "FORFEIT" for i in interactions)


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_run_writes_outputs(tmp_path):
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        max_turns=4,
        num_episodes=2,
        output_dir=str(tmp_path),
        save_results=True,
        verbose=False,
        model="fake",
    )
    client = _FakeClient(["MOVE_RIGHT", "DO", "MOVE_UP", "MOVE_DOWN"])
    out = ProHarness(cfg, client=client).run()
    assert out["summary"]["episodes"] == 2
    # Files were written.
    run_dirs = list(tmp_path.glob("craftaxfull-v0-*"))
    assert run_dirs, "expected a run directory"
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
    assert len(metrics) == 2
    # Secrets are not persisted.
    cfg_dump = json.loads((run_dirs[0] / "config.json").read_text())
    assert "api_key" not in cfg_dump
    # Every native Craftax frame is streamed to MP4 (one per episode).
    videos = sorted(run_dirs[0].glob("episode_*.mp4"))
    assert len(videos) == 2, f"expected 2 MP4s, got {videos}"
    assert all(video.stat().st_size > 0 for video in videos)
    # Raw (un-normalized) score is reported; for craftax it's achievement count.
    assert metrics[0]["raw_score_label"] == "achievements"
    assert metrics[0]["raw_score"] is not None
    sm = out["summary"]
    assert "mean_raw_achievements" in sm and "mean_episode_return" in sm
    # Per-turn replay transcript is written and is ingestible by `gb replay`.
    transcripts = sorted(run_dirs[0].glob("transcript_*.jsonl"))
    assert len(transcripts) == 2
    rows = [json.loads(ln) for ln in transcripts[0].read_text().splitlines() if ln.strip()]
    # Craftax obs is the beautified Unicode render: grid + rich HUD + legend.
    assert rows and "--- Legend ---" in rows[0]["observation"]
    assert "[Status]" in rows[0]["observation"]
    assert rows[0]["raw_score_label"] == "achievements"
    # The `gb replay` integration converts pro transcripts into the shared
    # renderer's rollout shape (discovery + filter + build_turns).
    from glyphbench.cli import _build_turns, _discover_pro_transcripts, _pro_rollouts

    found = _discover_pro_transcripts(run_dirs[0])
    assert transcripts[0] in found
    pro = _pro_rollouts([transcripts[0]])
    assert len(pro) == 1
    _, rollout = pro[0]
    assert rollout["info"]["env_id"] == "glyphbench/craftaxfull-v0"
    turns = _build_turns(rollout)
    assert turns and "--- Legend ---" in turns[0]["user"]
    assert rollout["trajectory"][0]["extras"]["glyphbench_step_role"] == "action"


def test_max_output_tokens_default_is_uncapped():
    # The whole point: by default reasoning is never truncated.
    cfg = ProConfig(task_id="glyphbench/craftaxfull-v0")
    assert cfg.max_output_tokens is None


def test_max_reasoning_effort_is_supported_by_config():
    cfg = ProConfig(task_id="glyphbench/craftaxfull-v0", reasoning_effort="max")
    assert cfg.reasoning_effort == "max"


def test_pixel_observations_are_restricted_to_craftaxfull_azure_pro():
    with pytest.raises(ValueError, match="Pro harness"):
        ProConfig(
            task_id="glyphbench/craftaxfull-v0",
            backend="azure",
            harness_variant="normal",
            observation_mode="pixels",
        )
    with pytest.raises(ValueError, match="Azure Responses"):
        ProConfig(
            task_id="glyphbench/craftaxfull-v0",
            backend="openai",
            harness_variant="pro",
            observation_mode="pixels",
        )


def test_native_text_observations_are_restricted_to_craftaxfull_pro():
    with pytest.raises(ValueError, match="Pro harness"):
        ProConfig(
            task_id="glyphbench/craftaxfull-v0",
            backend="azure",
            harness_variant="normal",
            observation_mode="native_text",
        )
    with pytest.raises(ValueError, match="craftaxfull-v0"):
        ProConfig(
            task_id="glyphbench/nethack-full-v0",
            backend="azure",
            harness_variant="pro",
            observation_mode="native_text",
        )


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_pixel_mode_sends_native_image_on_action_and_memory_turns():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        backend="azure",
        harness_variant="pro",
        observation_mode="pixels",
        image_detail="original",
        max_turns=2,
        save_results=False,
        save_video=False,
        verbose=False,
    )
    env = make_env(cfg.task_id, max_turns=2)
    client = _FakeClient(["NOOP"])
    result, interactions = ProHarness(cfg, client=client).run_episode(
        env, get_adapter(cfg.task_id), seed=42, episode_idx=0
    )

    assert result.num_image_inputs == len(client.users)
    assert result.num_image_inputs >= result.num_action_turns
    assert {row["model_observation_mode"] for row in interactions if row["role"] == "action"} == {
        "pixels"
    }
    for request in client.users:
        assert isinstance(request, list)
        content = request[0]["content"]
        text = next(part["text"] for part in content if part["type"] == "input_text")
        image = next(part for part in content if part["type"] == "input_image")
        assert "--- Legend ---" not in text
        assert image["image_url"].startswith("data:image/png;base64,iVBOR")
        assert image["detail"] == "original"


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_native_text_mode_uses_upstream_craftax_renderer():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        backend="azure",
        harness_variant="pro",
        observation_mode="native_text",
        max_turns=2,
        save_results=False,
        save_video=False,
        verbose=False,
    )
    env = make_env(cfg.task_id, max_turns=2)
    result_client = _FakeClient(["NOOP"])
    result, interactions = ProHarness(cfg, client=result_client).run_episode(
        env, get_adapter(cfg.task_id), seed=42, episode_idx=0
    )

    action_rows = [row for row in interactions if row["role"] == "action"]
    action_inputs = [
        user
        for label, user in zip(result_client.labels, result_client.users, strict=True)
        if label == "action"
    ]
    assert result.num_image_inputs == 0
    assert action_rows
    assert {row["model_observation_mode"] for row in action_rows} == {"native_text"}
    assert all(isinstance(user, str) and "Map:\n" in user for user in action_inputs)
    assert all("\nInventory:\n" in user for user in action_inputs)
    assert all("--- Legend ---" not in user for user in action_inputs)


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_run_can_batch_multiple_episode_seeds():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        harness_variant="normal",
        num_episodes=3,
        max_concurrent_episodes=3,
        max_turns=1,
        save_results=False,
        save_video=False,
        verbose=False,
    )
    output = ProHarness(cfg, client=_FakeClient(["NOOP"])).run()
    assert [episode["episode"] for episode in output["episodes"]] == [0, 1, 2]
    assert [episode["seed"] for episode in output["episodes"]] == [42, 43, 44]


def test_grounded_blocked_movement_feedback_requires_same_facing_and_position():
    before = "Pos:(36,27)  Facing:up\n"
    blocked = ProHarness._blocked_movement_feedback("MOVE_UP", before, before)
    assert "GROUNDED NAVIGATION FAILURE" in blocked
    assert "Do NOT repeat MOVE_UP" in blocked
    # Turning in place is not yet proof of a blocked edge, and a changed
    # position is a successful move.
    assert not ProHarness._blocked_movement_feedback(
        "MOVE_LEFT", before, "Pos:(36,27)  Facing:left\n"
    )
    assert not ProHarness._blocked_movement_feedback(
        "MOVE_UP", before, "Pos:(35,27)  Facing:up\n"
    )


def test_exploration_tracker_uses_only_visible_coordinates():
    tracker = ExplorationTracker()
    tracker.observe(
        "0",
        "·····\n··↓▼·\n·····\n\n"
        "HP:9/9 Pos:(20,30) Facing:down\n",
    )
    text = tracker.render("0")
    assert "Current (20, 30)" in text
    assert "ladder_down@(20,31)" in text
    assert "frontier tiles" in text


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_pro_memory_updates_can_be_periodic_and_event_driven():
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        max_turns=4,
        memory_update_every=4,
        save_results=False,
        save_video=False,
        verbose=False,
    )
    env = make_env(cfg.task_id, max_turns=4)
    client = _FakeClient(["NOOP"])
    result, interactions = ProHarness(cfg, client=client).run_episode(
        env, get_adapter(cfg.task_id), seed=42, episode_idx=0
    )
    assert result.num_action_turns == result.steps
    assert result.num_memory_turns <= result.steps
    assert result.memory_updates_skipped >= 1
    assert sum(row["role"] == "memory" for row in interactions) == result.num_memory_turns


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="craftax/jax extra not installed",
)
def test_env_crash_isolated_to_one_episode(monkeypatch):
    # An env-internal crash in one episode must not abort the whole sweep.
    cfg = ProConfig(
        task_id="glyphbench/craftaxfull-v0",
        max_turns=3,
        num_episodes=2,
        save_results=False,
        verbose=False,
        model="fake",
    )
    client = _FakeClient(["MOVE_RIGHT", "DO", "MOVE_UP"])
    h = ProHarness(cfg, client=client)

    real_run_episode = h.run_episode
    calls = {"n": 0}

    def flaky_run_episode(game, adapter, seed, episode_idx):
        calls["n"] += 1
        if episode_idx == 0:
            raise RuntimeError("simulated JAX/NLE backend crash")
        return real_run_episode(game, adapter, seed, episode_idx)

    monkeypatch.setattr(h, "run_episode", flaky_run_episode)
    out = h.run()
    # Both episodes recorded; episode 0 marked errored, episode 1 ran fine.
    assert out["summary"]["episodes"] == 2
    assert out["summary"]["errored_episodes"] == 1
    assert calls["n"] == 2
    assert out["episodes"][0]["error"] is not None
    assert out["episodes"][1]["error"] is None
