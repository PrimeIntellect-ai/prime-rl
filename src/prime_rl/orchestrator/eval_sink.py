"""Evaluation-side episode, group, and epoch assembly."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import DispatchFailure, EvalBatch, GroupCancellation
from prime_rl.orchestrator.utils import episode_env_name, eval_work

if TYPE_CHECKING:
    import verifiers.v1 as vf


class EvalSink:
    """Collect completed evaluation episodes into per-environment epochs."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_batches: dict[tuple[str, int], list[vf.Episode]] = defaultdict(list)
        self.pending_batch_failures: dict[tuple[str, int], list[DispatchFailure]] = defaultdict(list)
        self.pending_batch_cancellations: dict[tuple[str, int], int] = defaultdict(int)

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
        if self._batch_size(key) >= self.batch_size_for(key[0]):
            return self.process_batch(key)
        return None

    def _batch_size(self, key: tuple[str, int]) -> int:
        return (
            len(self.pending_batches[key])
            + len(self.pending_batch_failures[key])
            + self.pending_batch_cancellations[key]
        )

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        """Every rollout of an env's epoch: its examples times its group size."""
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int]]:
        """``(env, step, arrived, expected)`` per epoch in progress."""
        keys = set(self.pending_batches) | set(self.pending_batch_failures) | set(self.pending_batch_cancellations)
        return [
            (env_name, step, self._batch_size((env_name, step)), self.batch_size_for(env_name))
            for env_name, step in keys
        ]

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        env_name, step = key
        return EvalBatch(
            env_name=env_name,
            step=step,
            episodes=EvalEpisodes(self.pending_batches.pop(key, []), group_size=self.group_size_for(env_name)),
            failures=self.pending_batch_failures.pop(key, []),
            cancelled=self.pending_batch_cancellations.pop(key, 0),
        )
