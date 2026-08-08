"""EvalSink: three-level rollout sink for eval epochs.

Same shape as ``TrainSink``, but no tokenization / advantages / filters:

1. ``process_rollout`` — no-op.
2. ``process_group`` — at ``group_size`` episodes, move the episode results
   (errored ones included) into the ``(env, eval_step)`` bucket.
3. ``process_batch`` — at ``num_tasks × group_size`` episodes, return an
   ``EvalBatch`` with the full returned cohort (metrics are computed downstream).

``add()`` takes one ``EpisodeResult`` and returns ``EvalBatch | None``;
all accounting counts episodes, never loose traces.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalRollouts
from prime_rl.orchestrator.types import EpisodeResult, EvalBatch, Rollout
from prime_rl.utils.logger import get_logger


class EvalSink:
    """Constructed only when eval is configured."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[uuid.UUID, list[EpisodeResult]] = defaultdict(list)
        self.pending_batches: dict[tuple[str, int], list[EpisodeResult]] = defaultdict(list)

    def add(self, result: EpisodeResult) -> EvalBatch | None:
        """Process one episode arrival; finalize the group on the ``group_size``-th
        episode and the per-env epoch on the ``num_tasks × group_size``-th."""
        env_name = result.env_name
        group_id = result.group_id
        for rollout in result.rollouts:
            self.process_rollout(rollout)
        assert result.eval_step is not None, "eval episode missing eval_step"
        bkey = (env_name, result.eval_step)
        self.pending_groups[group_id].append(result)
        if len(self.pending_groups[group_id]) >= self.group_size_for(env_name):
            self.process_group(group_id)
        if len(self.pending_batches[bkey]) >= self.batch_size_for(env_name):
            return self.process_batch(bkey)
        return None

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        """``num_tasks × group_size`` — total episodes expected for one
        epoch of ``env_name``."""
        env = self.eval_envs.get(env_name)
        return len(env.eval_tasks) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        """One entry per accumulating ``(env, eval_step)`` batch:
        ``(env_name, eval_step, batch_count, expected, buffered)``.
        ``batch_count`` is finalized-group episodes in ``pending_batches``;
        ``buffered`` is partial-group episode arrivals."""
        batch_counts = {key: len(episodes) for key, episodes in self.pending_batches.items()}
        buffered: dict[tuple[str, int], int] = {}
        for episodes in self.pending_groups.values():
            if not episodes:
                continue
            first = episodes[0]
            assert first.eval_step is not None
            bkey = (first.env_name, first.eval_step)
            buffered[bkey] = buffered.get(bkey, 0) + len(episodes)
        return [
            (
                env_name,
                eval_step,
                batch_counts.get((env_name, eval_step), 0),
                self.batch_size_for(env_name),
                buffered.get((env_name, eval_step), 0),
            )
            for (env_name, eval_step) in set(batch_counts) | set(buffered)
        ]

    # ── level 1: per-rollout (no-op for eval) ─────────────────────────────

    def process_rollout(self, rollout: Rollout) -> None:
        """No-op. Eval rollouts don't need trainer-bound tokenization; the
        method exists to keep the three-level structure uniform with
        ``TrainSink``.
        """
        return None

    # ── level 2: per-group (move into batch bucket) ───────────────────────

    def process_group(self, group_id: uuid.UUID) -> None:
        episodes = self.pending_groups.pop(group_id, [])
        if not episodes:
            return
        group = [rollout for result in episodes for rollout in result.rollouts]
        env_name = episodes[0].env_name
        task_idx = episodes[0].task_data.idx
        eval_step = episodes[0].eval_step
        assert eval_step is not None
        bucket = self.pending_batches[(env_name, eval_step)]
        bucket.extend(episodes)

        survivors = [r for r in group if r.episode_ok and not r.has_error]
        num_failed_episodes = sum(not result.episode.ok for result in episodes)
        num_errored_traces = sum(r.has_error for r in group)
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} eval_step={eval_step} | "
            f"episodes={len(episodes)} (failed={num_failed_episodes}) | "
            f"traces={len(group)} (errored={num_errored_traces}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        """Pop the finished ``(env, eval_step)`` epoch and return the ``EvalBatch`` with its full
        returned cohort (errored rollouts included — the ``all`` set). Metrics are computed
        downstream via ``EvalBatch.rollouts.metrics`` over the all/effective subsets, so the sink
        does no aggregation."""
        env_name, step = key
        episodes = self.pending_batches.pop(key, [])
        rollouts = [rollout for result in episodes for rollout in result.rollouts]
        return EvalBatch(
            env_name=env_name,
            step=step,
            episodes=episodes,
            rollouts=EvalRollouts(rollouts, episode_results=episodes),
        )
