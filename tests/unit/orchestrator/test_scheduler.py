"""Tests for Scheduler — env selection, eval epoch triggering, dataset cycling.

We construct Scheduler with SchedulerInputs directly here, bypassing
setup_scheduler so we don't need to touch verifiers or the config layer.
"""

import asyncio
from collections import Counter
from itertools import islice

from prime_rl.orchestrator.scheduler import Dispatch, Scheduler, SchedulerInputs, Task, _cycle_forever


def _run(coro):
    """Drive an async function in a sync test — pytest-asyncio isn't wired."""
    return asyncio.run(coro)


def _task(id: str, kind: str = "train", rollouts_per_group: int = 1) -> Task:
    """Stub Task — env=None is fine since Scheduler only stores it."""
    return Task(id=id, env=None, sampling_args={}, kind=kind, rollouts_per_group=rollouts_per_group)  # type: ignore[arg-type]


def _stream(rows: list[dict]):
    """Wrap a finite list in an infinite cycle so next() never raises."""
    return iter(_cycle_forever(rows, env_id=None, buffer=None))


def _build(**overrides) -> Scheduler:
    """Build a Scheduler with sensible defaults, overrides supplied as kwargs."""
    return Scheduler(
        SchedulerInputs(
            train_tasks=overrides.pop("train_tasks", [_task("a")]),
            train_datasets=overrides.pop("train_datasets", [_stream([{"x": 1}])]),
            eval_tasks=overrides.pop("eval_tasks", []),
            eval_datasets=overrides.pop("eval_datasets", []),
            env_weights=overrides.pop("env_weights", None),
            eval_interval=overrides.pop("eval_interval", None),
            eval_at_zero=overrides.pop("eval_at_zero", False),
            seed=overrides.pop("seed", None),
        )
    )


# --- _cycle_forever ----------------------------------------------------------


def test_cycle_forever_loops_back():
    rows = [{"a": 1}, {"a": 2}]
    seen = list(islice(_cycle_forever(rows), 5))
    assert [r["a"] for r in seen] == [1, 2, 1, 2, 1]


def test_cycle_forever_assigns_monotonic_example_id():
    rows = [{"a": 1}, {"a": 2}]
    seen = list(islice(_cycle_forever(rows), 5))
    assert [r["example_id"] for r in seen] == [0, 1, 2, 3, 4]


def test_cycle_forever_does_not_mutate_source_rows():
    rows = [{"a": 1}]
    list(islice(_cycle_forever(rows), 3))
    assert rows == [{"a": 1}], "source list mutated"


class _StubBuffer:
    """Minimal buffer surface: just is_evicted(env_id, example) -> bool."""

    def __init__(self, evicted_ids: set[int]):
        self._evicted = evicted_ids

    def is_evicted(self, env_id: str, example: dict) -> bool:
        return example["example_id"] in self._evicted


def test_cycle_forever_skips_evicted_examples():
    """Eviction is keyed on example_id (assigned by the cycler), so once a
    given id is evicted it stays evicted across passes."""
    rows = [{"a": 1}, {"a": 2}, {"a": 3}]
    # Evict middle of pass 1 (id=1) and middle of pass 2 (id=4).
    buf = _StubBuffer(evicted_ids={1, 4})
    seen = list(islice(_cycle_forever(rows, env_id="env", buffer=buf), 4))
    assert [r["example_id"] for r in seen] == [0, 2, 3, 5]


def test_cycle_forever_falls_back_when_fully_evicted():
    """If a full pass yields nothing (all evicted), yield raw rows so the
    engine doesn't deadlock."""
    rows = [{"a": 1}, {"a": 2}]
    # Evict ids 0 and 1 (the first pass) but not 2+ (so the fallback yields
    # all subsequent rows raw).
    buf = _StubBuffer(evicted_ids={0, 1})
    seen = list(islice(_cycle_forever(rows, env_id="env", buffer=buf), 4))
    assert [r["example_id"] for r in seen] == [2, 3, 4, 5]


# --- next_task: train round-robin --------------------------------------------


def test_next_task_round_robin_cycles_through_envs():
    sched = _build(
        train_tasks=[_task("a"), _task("b"), _task("c")],
        train_datasets=[_stream([{"x": "a1"}]), _stream([{"x": "b1"}]), _stream([{"x": "c1"}])],
    )
    seen_ids = [sched.next_task().task.id for _ in range(6)]  # type: ignore[union-attr]
    assert seen_ids == ["a", "b", "c", "a", "b", "c"]


def test_next_task_returns_dispatch_with_train_marker():
    """Train dispatches have eval_step=None; that's how the engine distinguishes."""
    sched = _build()
    d = sched.next_task()
    assert isinstance(d, Dispatch)
    assert d.eval_step is None
    assert d.task.kind == "train"


# --- next_task: train weighted -----------------------------------------------


def test_next_task_weighted_distribution_matches_ratios():
    """With weights [0.9, 0.1], env 'a' should dominate over many draws."""
    sched = _build(
        train_tasks=[_task("a"), _task("b")],
        train_datasets=[_stream([{"x": "a"}]), _stream([{"x": "b"}])],
        env_weights=[0.9, 0.1],
        seed=42,
    )
    counts = Counter(sched.next_task().task.id for _ in range(1000))  # type: ignore[union-attr]
    # 90/10 split; allow ±5pp slack at n=1000
    assert 850 < counts["a"] < 950, counts
    assert 50 < counts["b"] < 150, counts


def test_next_task_weighted_is_deterministic_under_same_seed():
    def run_once():
        sched = _build(
            train_tasks=[_task("a"), _task("b"), _task("c")],
            train_datasets=[_stream([{"x": "a"}]), _stream([{"x": "b"}]), _stream([{"x": "c"}])],
            env_weights=[0.5, 0.3, 0.2],
            seed=123,
        )
        return [sched.next_task().task.id for _ in range(20)]  # type: ignore[union-attr]

    assert run_once() == run_once()


# --- next_task: eval queue exclusivity ---------------------------------------


def _eval_setup(eval_at_zero: bool = True, eval_interval: int | None = None):
    return _build(
        eval_tasks=[_task("eval_env", kind="eval", rollouts_per_group=2)],
        eval_datasets=[[{"prompt": "p1"}, {"prompt": "p2"}]],
        eval_interval=eval_interval,
        eval_at_zero=eval_at_zero,
    )


def test_eval_queue_drained_exclusively_before_train():
    """While the eval queue has items, every dispatch must be eval — never
    interleave train into the middle of an eval epoch."""
    sched = _eval_setup(eval_at_zero=True)
    d1 = sched.next_task()
    d2 = sched.next_task()
    d3 = sched.next_task()  # eval drained, should now return train
    assert d1.task.kind == "eval" and d1.eval_step == 0  # type: ignore[union-attr]
    assert d2.task.kind == "eval" and d2.eval_step == 0  # type: ignore[union-attr]
    assert d3.task.kind == "train" and d3.eval_step is None  # type: ignore[union-attr]


def test_eval_dispatching_step_clears_when_queue_empties():
    sched = _eval_setup(eval_at_zero=True)
    assert sched._dispatching_eval_step == 0
    sched.next_task()
    assert sched._dispatching_eval_step == 0  # still draining
    sched.next_task()  # last eval entry
    assert sched._dispatching_eval_step is None


def test_eval_entries_have_example_id_per_position():
    sched = _eval_setup(eval_at_zero=True)
    d1 = sched.next_task()
    d2 = sched.next_task()
    assert d1.example["example_id"] == 0  # type: ignore[union-attr]
    assert d2.example["example_id"] == 1  # type: ignore[union-attr]


def test_eval_entries_are_cartesian_product_env_x_examples():
    """Two eval envs × two examples each = 4 dispatches, ordered by env then
    example."""
    sched = _build(
        eval_tasks=[_task("e1", kind="eval"), _task("e2", kind="eval")],
        eval_datasets=[
            [{"p": "a"}, {"p": "b"}],
            [{"p": "c"}, {"p": "d"}],
        ],
        eval_at_zero=True,
    )
    drained = [sched.next_task() for _ in range(4)]
    assert [d.task.id for d in drained] == ["e1", "e1", "e2", "e2"]  # type: ignore[union-attr]
    assert [d.example["p"] for d in drained] == ["a", "b", "c", "d"]  # type: ignore[union-attr]


def test_eval_at_zero_false_does_not_populate_queue():
    sched = _eval_setup(eval_at_zero=False)
    d = sched.next_task()
    assert d.task.kind == "train"  # type: ignore[union-attr]


# --- on_new_version ----------------------------------------------------------


def test_on_new_version_noop_without_eval_tasks():
    sched = _build(eval_interval=2)
    _run(sched.on_new_version(10))
    assert sched.last_eval_step == 0
    assert not sched._eval_queue


def test_on_new_version_noop_without_interval():
    sched = _eval_setup(eval_at_zero=False, eval_interval=None)
    _run(sched.on_new_version(10))
    assert not sched._eval_queue


def test_on_new_version_skips_when_step_below_interval():
    sched = _eval_setup(eval_at_zero=False, eval_interval=5)
    _run(sched.on_new_version(3))
    assert not sched._eval_queue
    assert sched.last_eval_step == 0


def test_on_new_version_triggers_at_exact_boundary():
    sched = _eval_setup(eval_at_zero=False, eval_interval=5)
    _run(sched.on_new_version(5))
    assert sched.last_eval_step == 5
    assert len(sched._eval_queue) == 2  # 2 eval examples


def test_on_new_version_triggers_when_step_jumps_over_boundary():
    """Watcher may go 0 → 7 (skipping 5). The trigger should still fire."""
    sched = _eval_setup(eval_at_zero=False, eval_interval=5)
    _run(sched.on_new_version(7))
    assert sched.last_eval_step == 7
    assert len(sched._eval_queue) == 2


def test_on_new_version_skips_when_previous_epoch_still_dispatching():
    """If an eval epoch is mid-dispatch (queue non-empty), don't start a new
    one even if we've passed another interval boundary."""
    sched = _eval_setup(eval_at_zero=True, eval_interval=5)
    assert sched._dispatching_eval_step == 0
    _run(sched.on_new_version(10))  # would normally trigger
    # still dispatching the original epoch — no new entries appended
    assert len(sched._eval_queue) == 2
    assert sched.last_eval_step == 0


def test_on_new_version_triggers_again_after_previous_epoch_drained():
    sched = _eval_setup(eval_at_zero=True, eval_interval=5)
    sched.next_task()
    sched.next_task()  # drain initial epoch
    assert sched._dispatching_eval_step is None
    _run(sched.on_new_version(7))
    assert sched.last_eval_step == 7
    assert len(sched._eval_queue) == 2


def test_on_new_version_anchors_next_interval_on_trigger_step():
    """After triggering at step 7 (with interval 5), next trigger is at 12+,
    not 10."""
    sched = _eval_setup(eval_at_zero=False, eval_interval=5)
    _run(sched.on_new_version(7))
    sched.next_task()
    sched.next_task()  # drain
    _run(sched.on_new_version(11))  # 11 < 7 + 5, should NOT trigger
    assert sched.last_eval_step == 7
    _run(sched.on_new_version(12))  # 12 == 7 + 5, should trigger
    assert sched.last_eval_step == 12


# --- expected_eval_count -----------------------------------------------------


def test_expected_eval_count_returns_count_for_triggered_step():
    sched = _eval_setup(eval_at_zero=True)
    assert sched.expected_eval_count(0) == 2


def test_expected_eval_count_returns_none_for_untriggered_step():
    sched = _eval_setup(eval_at_zero=False, eval_interval=5)
    assert sched.expected_eval_count(99) is None
