"""Tests for WeightWatcher — observer fan-out, missing-dir tolerance,
raced-cleanup tolerance, observer-raise recovery."""

import asyncio
from pathlib import Path

from prime_rl.orchestrator.watcher import WatcherInputs, WeightWatcher


def _run(coro):
    return asyncio.run(coro)


class _RecordingObserver:
    def __init__(self, name: str, raise_at: int | None = None):
        self.name = name
        self.calls: list[int] = []
        self._raise_at = raise_at

    async def on_new_version(self, step: int) -> None:
        self.calls.append(step)
        if self._raise_at == step:
            raise RuntimeError(f"{self.name} failed at step {step}")


def _make_step(broadcast_dir: Path, step: int) -> None:
    """Create a step_N directory with the STABLE marker the watcher looks for."""
    step_dir = broadcast_dir / f"step_{step}"
    step_dir.mkdir(parents=True)
    (step_dir / "STABLE").touch()


def _build(broadcast_dir: Path, observers: list[_RecordingObserver]) -> WeightWatcher:
    return WeightWatcher(WatcherInputs(broadcast_dir=broadcast_dir, observers=observers))  # type: ignore[arg-type]


def test_tick_noop_when_broadcast_dir_missing(tmp_path: Path):
    """Pre-startup state: trainer hasn't created the dir yet."""
    obs = _RecordingObserver("a")
    w = _build(tmp_path / "missing", [obs])
    _run(w.tick())
    assert obs.calls == []
    assert w.current_step == 0


def test_tick_noop_when_no_new_step(tmp_path: Path):
    """Dir exists but only old steps are present."""
    _make_step(tmp_path, 0)
    obs = _RecordingObserver("a")
    w = _build(tmp_path, [obs])
    w.current_step = 5  # already past anything in dir
    _run(w.tick())
    assert obs.calls == []
    assert w.current_step == 5


def test_tick_fires_observers_in_order_on_new_step(tmp_path: Path):
    """All observers called with the new step, in declared order."""
    _make_step(tmp_path, 3)
    a = _RecordingObserver("a")
    b = _RecordingObserver("b")
    c = _RecordingObserver("c")
    w = _build(tmp_path, [a, b, c])

    _run(w.tick())

    assert a.calls == [3]
    assert b.calls == [3]
    assert c.calls == [3]
    assert w.current_step == 3


def test_tick_jumps_to_latest_step_skipping_intermediates(tmp_path: Path):
    """Trainer pruned step 1 + 2; watcher should call observers with 3 only."""
    _make_step(tmp_path, 1)
    _make_step(tmp_path, 2)
    _make_step(tmp_path, 3)
    obs = _RecordingObserver("a")
    w = _build(tmp_path, [obs])

    _run(w.tick())

    assert obs.calls == [3]
    assert w.current_step == 3


def test_tick_advances_current_step_on_observer_raise(tmp_path: Path):
    """If an observer raises, log + advance current_step anyway so we don't
    loop forever retrying the same broken step."""
    _make_step(tmp_path, 5)
    a = _RecordingObserver("a")
    b = _RecordingObserver("b", raise_at=5)
    c = _RecordingObserver("c")
    w = _build(tmp_path, [a, b, c])

    _run(w.tick())

    # a was called, b raised, c skipped
    assert a.calls == [5]
    assert b.calls == [5]
    assert c.calls == []
    # but current_step still advances so we don't retry
    assert w.current_step == 5


def test_tick_skips_when_step_path_disappears_mid_check(tmp_path: Path, monkeypatch):
    """Race: latest_step returned but the step dir was just cleaned up by the
    trainer. Watcher should skip silently and try again next tick."""
    _make_step(tmp_path, 1)
    obs = _RecordingObserver("a")
    w = _build(tmp_path, [obs])

    # Force get_latest_ckpt_step to claim step 99 exists; then the step path
    # check inside tick() will fail since step_99 isn't real.
    import prime_rl.orchestrator.watcher as watcher_mod

    monkeypatch.setattr(watcher_mod, "get_latest_ckpt_step", lambda _d: 99)
    _run(w.tick())

    assert obs.calls == []
    assert w.current_step == 0  # didn't advance


def test_consecutive_ticks_only_fire_once_per_step(tmp_path: Path):
    """current_step gating: a second tick at the same latest step is a no-op."""
    _make_step(tmp_path, 1)
    obs = _RecordingObserver("a")
    w = _build(tmp_path, [obs])

    _run(w.tick())
    _run(w.tick())
    _run(w.tick())

    assert obs.calls == [1]  # only fired once


def test_tick_picks_up_subsequent_step_after_first(tmp_path: Path):
    """First tick fires for step 1, second tick fires for step 2."""
    _make_step(tmp_path, 1)
    obs = _RecordingObserver("a")
    w = _build(tmp_path, [obs])

    _run(w.tick())
    assert obs.calls == [1]

    _make_step(tmp_path, 2)
    _run(w.tick())
    assert obs.calls == [1, 2]
    assert w.current_step == 2
