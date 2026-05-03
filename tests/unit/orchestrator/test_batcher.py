"""Tests for TrainBatcher and its pure helpers.

Covers the data-transformation helpers (_rollout_metrics, _eval_metrics,
_rollout_timing) directly, plus the routing/lifecycle methods of
TrainBatcher (_handle_eval, _flush_eval, _maybe_save_ckpt, _wait_barrier).

The full run() loop is exercised by the reverse-text smoke test, not here —
it would need stubs for everything (advantage, post-processor, sender,
buffer) and the value/cost ratio is poor.
"""

import asyncio
from dataclasses import dataclass

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
from prime_rl.orchestrator.batcher import (
    BatcherInputs,
    StepStrategy,
    TrainBatcher,
    _eval_metrics,
    _rollout_metrics,
    _rollout_timing,
)
from prime_rl.orchestrator.engine import Group


def _run(coro):
    return asyncio.run(coro)


# --- _rollout_timing ---------------------------------------------------------


def test_rollout_timing_empty_when_no_timing_block():
    assert _rollout_timing({}) == {}
    assert _rollout_timing({"reward": 1.0}) == {}


def test_rollout_timing_extracts_duration_from_dicts_and_scalars():
    """`total`/`overhead` are scalars; the rest carry a `duration` derived
    from start/end timestamps (so they show up as dicts in the rollout)."""
    r = {
        "timing": {
            "total": 1.5,
            "overhead": 0.05,
            "setup": {"duration": 0.1, "start": 0, "end": 0.1},
            "generation": {"duration": 1.0, "start": 0.1, "end": 1.1},
            "scoring": {"duration": 0.2},
            "model": {"duration": 0.05},
            "env": {"duration": 0.1},
        }
    }
    out = _rollout_timing(r)
    assert out == {
        "total": 1.5,
        "overhead": 0.05,
        "setup": 0.1,
        "generation": 1.0,
        "scoring": 0.2,
        "model": 0.05,
        "env": 0.1,
    }


def test_rollout_timing_skips_missing_fields():
    r = {"timing": {"total": 0.5, "generation": {"duration": 0.4}}}
    assert _rollout_timing(r) == {"total": 0.5, "generation": 0.4}


# --- _rollout_metrics --------------------------------------------------------


def test_rollout_metrics_empty_returns_empty():
    assert _rollout_metrics("", []) == {}
    assert _rollout_metrics("rollouts", []) == {}


def _rollout(reward: float, completion_len: int = 1) -> dict:
    """Build a minimal vf.RolloutOutput-shaped dict for metric tests.
    `seq_len` is computed by get_seq_len from prompt_ids + completion_ids on
    the last trajectory step, and prompt_len from prompt_ids on the first
    step. With one step both are the same step → completion_len = len(completion_ids)."""
    return {
        "reward": reward,
        "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(completion_len))}}],
    }


def test_rollout_metrics_basic_stats_no_filters_no_timing():
    rollouts = [
        _rollout(reward=0.0, completion_len=3),
        _rollout(reward=1.0, completion_len=2),
        _rollout(reward=0.0, completion_len=1),
    ]
    m = _rollout_metrics("", rollouts)
    assert m["reward/mean"] == 1.0 / 3
    assert m["seq_len/mean"] == 2.0  # (3+2+1)/3
    assert m["pass_rate"] == 1.0 / 3  # one reward > 0
    assert m["n_rollouts"] == 3


def test_rollout_metrics_emits_filter_rates_when_filter_keys_present():
    rollouts = [
        {"reward": 0.0, "trajectory": [], "is_filtered": True, "filters": {"gibberish": True, "repetition": False}},
        {"reward": 1.0, "trajectory": [], "is_filtered": False, "filters": {"gibberish": False, "repetition": False}},
    ]
    m = _rollout_metrics("", rollouts)
    assert m["filters/drop_rate"] == 0.5
    assert m["filters/gibberish/rate"] == 0.5
    assert m["filters/repetition/rate"] == 0.0


def test_rollout_metrics_keys_under_prefix():
    rollouts = [{"reward": 0.0, "trajectory": []}]
    m = _rollout_metrics("rollouts", rollouts)
    assert "rollouts/reward/mean" in m
    assert "rollouts/n_rollouts" in m
    assert "reward/mean" not in m  # no top-level


def test_rollout_metrics_emits_timing_means_when_present():
    rollouts = [
        {
            "reward": 0.0,
            "trajectory": [],
            "timing": {"total": 1.0, "overhead": 0.1, "generation": {"duration": 0.5}},
        },
        {
            "reward": 0.0,
            "trajectory": [],
            "timing": {"total": 3.0, "overhead": 0.3, "generation": {"duration": 1.5}},
        },
    ]
    m = _rollout_metrics("", rollouts)
    assert m["timing/total/mean"] == 2.0
    assert m["timing/overhead/mean"] == 0.2
    assert m["timing/generation/mean"] == 1.0


# --- _eval_metrics -----------------------------------------------------------


def _eval_group(rewards: list[float], env_id: str = "env_a") -> Group:
    return Group(
        example={"example_id": 0},
        env_id=env_id,
        kind="eval",
        rollouts=[{"reward": r, "trajectory": []} for r in rewards],
        policy_version=0,
        eval_step=0,
    )


def test_eval_metrics_empty_when_all_groups_empty():
    """All groups timed out (no rollouts) — return empty so the batcher
    can short-circuit logging."""
    groups = [
        Group(example={}, env_id="e", kind="eval", rollouts=[], policy_version=0, eval_step=0),
    ]
    assert _eval_metrics("eval", groups) == {}


def test_eval_metrics_pass_at_k_any_rollout_passed():
    """pass_at_k = fraction of examples where ≥1 rollout had reward > 0."""
    groups = [
        _eval_group([0.0, 0.0, 1.0]),  # passed
        _eval_group([0.0, 0.0, 0.0]),  # didn't pass
        _eval_group([1.0]),  # passed
    ]
    m = _eval_metrics("eval", groups)
    assert m["eval/pass_at_k"] == 2 / 3
    assert m["eval/n_examples"] == 3


def test_eval_metrics_avg_at_k_is_mean_of_per_example_means():
    groups = [
        _eval_group([0.0, 1.0]),  # mean 0.5
        _eval_group([1.0, 1.0]),  # mean 1.0
    ]
    m = _eval_metrics("eval", groups)
    assert m["eval/avg_at_k"] == 0.75


# --- TrainBatcher fixtures ---------------------------------------------------


@dataclass
class _StubPolicy:
    policy_version: int = 0

    def max_off_policy_level(self) -> int:
        return 0


@dataclass
class _StubEvalCounter:
    last_eval_step: int = 0
    expected: dict[int, int] | None = None

    def expected_eval_count(self, step: int) -> int | None:
        return (self.expected or {}).get(step)


class _StubPostProcessor:
    """Records calls to process(); doesn't actually convert/send."""

    def __init__(self):
        self.calls: list[tuple[list, list, int]] = []

    async def process(self, trainable, filtered, step):
        self.calls.append((list(trainable), list(filtered), step))


class _StubCkptManager:
    """Records save calls."""

    def __init__(self):
        self.saves: list[tuple[int, dict]] = []

    def save(self, state, step):
        self.saves.append((step, dict(state.__dict__)))


def _build(
    *,
    policy: _StubPolicy | None = None,
    eval_counter: _StubEvalCounter | None = None,
    ckpt_manager: _StubCkptManager | None = None,
    ckpt_interval: int | None = None,
    max_steps: int | None = None,
    max_training_batches_ahead: int = 1,
    strict: bool = False,
) -> TrainBatcher:
    return TrainBatcher(
        BatcherInputs(
            in_q=asyncio.Queue(),
            post=_StubPostProcessor(),  # type: ignore[arg-type]
            policy=policy or _StubPolicy(),  # type: ignore[arg-type]
            strategy=StepStrategy(size=1),
            advantage_cfg=DefaultAdvantageConfig(type="default"),
            filters=[],
            max_steps=max_steps,
            max_training_batches_ahead=max_training_batches_ahead,
            strict_async_level=strict,
            eval_counter=eval_counter,  # type: ignore[arg-type]
            ckpt_manager=ckpt_manager,  # type: ignore[arg-type]
            ckpt_interval=ckpt_interval,
        )
    )


# --- _handle_eval ------------------------------------------------------------


def test_handle_eval_ignored_without_eval_counter():
    """No eval_counter => not an eval-aware run => drop on the floor."""
    b = _build(eval_counter=None)
    g = _eval_group([1.0])
    b._handle_eval(g)
    assert b._eval_buf == {}


def test_handle_eval_accumulates_groups_by_trigger_step():
    counter = _StubEvalCounter(expected={5: 3})  # need 3 groups before flush
    b = _build(eval_counter=counter)

    g1 = _eval_group([1.0])
    g1.eval_step = 5
    g2 = _eval_group([0.5])
    g2.eval_step = 5
    b._handle_eval(g1)
    b._handle_eval(g2)

    # Only 2 of 3 expected; still buffered, not flushed
    assert len(b._eval_buf[5]) == 2


def test_handle_eval_flushes_when_expected_count_reached(monkeypatch):
    """Once the expected count is hit, the trigger step's buffer is consumed
    and flushed (popped from _eval_buf)."""
    counter = _StubEvalCounter(expected={5: 2})
    b = _build(eval_counter=counter)

    flushed_steps: list[int] = []
    monkeypatch.setattr(b, "_flush_eval", lambda step, groups: flushed_steps.append(step))

    g1 = _eval_group([1.0])
    g1.eval_step = 5
    g2 = _eval_group([0.5])
    g2.eval_step = 5
    b._handle_eval(g1)
    b._handle_eval(g2)

    assert flushed_steps == [5]
    assert 5 not in b._eval_buf  # popped


def test_handle_eval_does_not_flush_when_eval_step_unknown():
    """If expected_eval_count returns None (e.g. a stale eval_step), keep
    accumulating without flushing."""
    counter = _StubEvalCounter(expected={})
    b = _build(eval_counter=counter)
    g = _eval_group([1.0])
    g.eval_step = 99
    b._handle_eval(g)
    assert len(b._eval_buf[99]) == 1


# --- _maybe_save_ckpt --------------------------------------------------------


def test_maybe_save_ckpt_noop_without_manager():
    b = _build(ckpt_manager=None, ckpt_interval=10)
    b.step = 10
    _run(b._maybe_save_ckpt())  # should not raise


def test_maybe_save_ckpt_noop_without_interval():
    mgr = _StubCkptManager()
    b = _build(ckpt_manager=mgr, ckpt_interval=None)
    b.step = 10
    _run(b._maybe_save_ckpt())
    assert mgr.saves == []


def test_maybe_save_ckpt_noop_at_step_zero():
    """Don't save at step 0 (haven't actually run anything)."""
    mgr = _StubCkptManager()
    b = _build(ckpt_manager=mgr, ckpt_interval=5)
    b.step = 0
    _run(b._maybe_save_ckpt())
    assert mgr.saves == []


def test_maybe_save_ckpt_skips_off_interval_steps():
    mgr = _StubCkptManager()
    b = _build(ckpt_manager=mgr, ckpt_interval=5)
    b.step = 7  # not a multiple of 5
    _run(b._maybe_save_ckpt())
    assert mgr.saves == []


def test_maybe_save_ckpt_fires_at_interval_boundary():
    mgr = _StubCkptManager()
    b = _build(ckpt_manager=mgr, ckpt_interval=5)
    b.step = 10
    _run(b._maybe_save_ckpt())
    assert len(mgr.saves) == 1
    saved_step, saved_state = mgr.saves[0]
    assert saved_step == 10
    assert saved_state["step"] == 10


def test_maybe_save_ckpt_skips_when_at_max_steps():
    """Don't save on the final step — exit follows immediately."""
    mgr = _StubCkptManager()
    b = _build(ckpt_manager=mgr, ckpt_interval=5, max_steps=10)
    b.step = 10
    _run(b._maybe_save_ckpt())
    assert mgr.saves == []


# --- _wait_barrier -----------------------------------------------------------


def test_wait_barrier_returns_immediately_when_within_target():
    """Lead = step - policy_version <= max_training_batches_ahead → return."""
    policy = _StubPolicy(policy_version=10)
    b = _build(policy=policy, max_training_batches_ahead=2)
    b.step = 11  # lead = 1, within 2

    async def go():
        await asyncio.wait_for(b._wait_barrier(), timeout=0.5)

    _run(go())  # should not block


def test_wait_barrier_returns_when_lead_equals_target_in_strict_mode():
    policy = _StubPolicy(policy_version=10)
    b = _build(policy=policy, max_training_batches_ahead=2, strict=True)
    b.step = 12  # lead = 2, exact

    async def go():
        await asyncio.wait_for(b._wait_barrier(), timeout=0.5)

    _run(go())


def test_wait_barrier_blocks_when_lead_exceeds_target():
    """Lead beyond target → block until policy advances."""
    policy = _StubPolicy(policy_version=0)
    b = _build(policy=policy, max_training_batches_ahead=1)
    b.step = 5  # lead = 5, way over

    async def go():
        # Should time out; the barrier would block forever without policy bump
        try:
            await asyncio.wait_for(b._wait_barrier(), timeout=0.3)
        except asyncio.TimeoutError:
            return "blocked"
        return "returned"

    assert _run(go()) == "blocked"


def test_wait_barrier_unblocks_when_policy_catches_up():
    policy = _StubPolicy(policy_version=0)
    b = _build(policy=policy, max_training_batches_ahead=1)
    b.step = 5

    async def go():
        async def bump_later():
            await asyncio.sleep(0.05)
            policy.policy_version = 5  # lead drops to 0

        bumper = asyncio.create_task(bump_later())
        try:
            await asyncio.wait_for(b._wait_barrier(), timeout=1.0)
        finally:
            await bumper

    _run(go())  # should not raise
