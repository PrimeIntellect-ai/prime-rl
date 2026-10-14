"""Evaluation-side episode, group, and epoch assembly.

A bounded env's epoch is complete once every rollout it expects arrived. A streaming
env's epoch ends only with its taskset, which may run for days, so the sink cuts it
into windows instead of holding every episode until the end: a window closes once it
holds ``STREAM_WINDOW_ATTEMPTS`` attempts or is ``STREAM_WINDOW_S`` old (the runner
asks ``due_windows``), and its episodes are dropped once it is logged."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import DispatchFailure, EvalBatch, GroupCancellation
from prime_rl.orchestrator.utils import episode_env_name, eval_work

if TYPE_CHECKING:
    import verifiers.v1 as vf

STREAM_WINDOW_ATTEMPTS = 100
"""Attempts (episodes, failures, cancellations) after which a streaming env's window closes."""
STREAM_WINDOW_S = 600.0
"""Age after which a streaming env's window closes, if it holds any attempt."""


class EvalSink:
    """Collect completed evaluation episodes into per-environment epochs."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_batches: dict[tuple[str, int], list[vf.Episode]] = defaultdict(list)
        self.pending_batch_failures: dict[tuple[str, int], list[DispatchFailure]] = defaultdict(list)
        self.pending_batch_cancellations: dict[tuple[str, int], int] = defaultdict(int)
        self.window_opened: dict[tuple[str, int], float] = {}
        """When each streaming epoch's open window took its first attempt."""
        self.windows_closed: dict[tuple[str, int], int] = defaultdict(int)
        """Windows each streaming epoch has closed so far."""
        self.closed_attempts: dict[tuple[str, int], int] = defaultdict(int)
        """Attempts in each streaming epoch's closed windows."""

    def add(self, episode: vf.Episode) -> EvalBatch | None:
        key = (episode_env_name(episode), eval_work(episode).step)
        self.pending_batches[key].append(episode)
        return self._complete(key)

    def fail(self, failure: DispatchFailure) -> EvalBatch | None:
        """Count a request failure toward its eval epoch without manufacturing a
        verifier episode."""
        if failure.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {failure.kind} dispatch failure")
        key = (failure.env_name, failure.step)
        self.pending_batch_failures[key].append(failure)
        return self._complete(key)

    def cancel(self, cancellation: GroupCancellation) -> EvalBatch | None:
        """Count a dispatcher-cancelled group toward its superseded eval epoch."""
        if cancellation.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {cancellation.kind} group cancellation")
        key = (cancellation.env_name, cancellation.step)
        self.pending_batch_cancellations[key] += cancellation.count
        return self._complete(key)

    def _complete(self, key: tuple[str, int]) -> EvalBatch | None:
        expected = self.batch_size_for(key[0])
        if expected is None:
            self.window_opened.setdefault(key, time.monotonic())
            if self._batch_size(key) >= STREAM_WINDOW_ATTEMPTS:
                return self.close_window(key)
            return None
        if self._batch_size(key) >= expected:
            return self.process_batch(key)
        return None

    def due_windows(self) -> list[EvalBatch]:
        """Close the streaming windows that are ``STREAM_WINDOW_S`` old."""
        now = time.monotonic()
        return [
            self.close_window(key)
            for key, opened in list(self.window_opened.items())
            if now - opened >= STREAM_WINDOW_S
        ]

    def close_window(self, key: tuple[str, int]) -> EvalBatch:
        """Close the open window of a streaming epoch that goes on."""
        self.window_opened.pop(key, None)
        self.closed_attempts[key] += self._batch_size(key)
        batch = self._take(key, window=self.windows_closed[key], final=False)
        self.windows_closed[key] += 1
        return batch

    def _batch_size(self, key: tuple[str, int]) -> int:
        return (
            len(self.pending_batches[key])
            + len(self.pending_batch_failures[key])
            + self.pending_batch_cancellations[key]
        )

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int | None:
        """Every rollout of an env's epoch: its examples times its group size. ``None``
        for a streaming env, whose epoch ends with its taskset."""
        env = self.eval_envs.get(env_name)
        if env.feed is not None:
            return None
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int | None]]:
        """``(env, step, arrived, expected)`` per epoch in progress."""
        keys = (
            set(self.pending_batches)
            | set(self.pending_batch_failures)
            | set(self.pending_batch_cancellations)
            | set(self.closed_attempts)
        )
        return [
            (
                env_name,
                step,
                self.closed_attempts.get((env_name, step), 0) + self._batch_size((env_name, step)),
                self.batch_size_for(env_name),
            )
            for env_name, step in keys
        ]

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        """The epoch's final batch: a bounded epoch whole, or a stream's last window."""
        self.window_opened.pop(key, None)
        self.closed_attempts.pop(key, None)
        window = self.windows_closed.pop(key, 0) if self.batch_size_for(key[0]) is None else None
        return self._take(key, window=window, final=True)

    def _take(self, key: tuple[str, int], *, window: int | None, final: bool) -> EvalBatch:
        env_name, step = key
        return EvalBatch(
            env_name=env_name,
            step=step,
            episodes=EvalEpisodes(self.pending_batches.pop(key, []), group_size=self.group_size_for(env_name)),
            failures=self.pending_batch_failures.pop(key, []),
            cancelled=self.pending_batch_cancellations.pop(key, 0),
            window=window,
            final=final,
        )
