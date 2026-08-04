"""EvalSink: three-level rollout sink for eval epochs.

Same shape as ``TrainSink``, but no tokenization / advantages / filters:

1. ``process_rollout`` — no-op.
2. ``process_group`` — at ``group_size`` episodes, move the rollouts
   (errored ones included) into the ``(env, eval_step)`` bucket.
3. ``process_batch`` — at ``num_examples × group_size`` episodes, return an
   ``EvalBatch`` with the full returned cohort (metrics are computed downstream).

``add()`` takes one ``Episode`` and returns ``EvalBatch | None``;
all accounting counts episodes, never loose traces.
"""

from __future__ import annotations

from collections import defaultdict

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalRollouts
from prime_rl.orchestrator.types import Episode, EvalBatch, Rollout, env_name_of, group_id_of, rollouts_of
from prime_rl.utils.logger import get_logger


def eval_step_of(episode: Episode) -> int:
    """The eval epoch an episode belongs to, off the run the dispatcher recorded when it landed.
    Only the eval path has one, which is why this lives here and not on ``Episode``."""
    assert isinstance(episode.run, vf.EvalRunInfo) and episode.run.step is not None
    return episode.run.step


class EvalSink:
    """Constructed only when eval is configured."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[str, list[Episode]] = defaultdict(list)
        # Episodes arrived per group / per batch bucket — the finalization counts.
        self.pending_group_episodes: dict[str, int] = defaultdict(int)
        self.pending_batches: dict[tuple[str, int], list[Episode]] = defaultdict(list)
        self.pending_batch_episodes: dict[tuple[str, int], int] = defaultdict(int)

    def add(self, episode: Episode) -> EvalBatch | None:
        """Process one episode arrival; finalize the group on the ``group_size``-th
        episode and the per-env epoch on the ``num_examples × group_size``-th. A failed
        episode brings no rollouts but still counts toward both."""
        env_name = env_name_of(episode)
        group_id = group_id_of(episode)
        for rollout in rollouts_of(episode):
            self.process_rollout(rollout)
        bkey = (env_name, eval_step_of(episode))
        self.pending_groups[group_id].append(episode)
        self.pending_group_episodes[group_id] += 1
        if self.pending_group_episodes[group_id] >= self.group_size_for(env_name):
            self.process_group(group_id)
        if self.pending_batch_episodes[bkey] >= self.batch_size_for(env_name):
            return self.process_batch(bkey)
        return None

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        """``num_examples × group_size`` — total episodes expected for one
        epoch of ``env_name``."""
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        """One entry per accumulating ``(env, eval_step)`` batch:
        ``(env_name, eval_step, batch_count, expected, buffered)``.
        ``batch_count`` is finalized-group episodes in ``pending_batches``;
        ``buffered`` is partial-group episode arrivals from non-group-scoring envs."""
        batch_counts: dict[tuple[str, int], int] = dict(self.pending_batch_episodes)
        buffered: dict[tuple[str, int], int] = {}
        for group_id, episodes in self.pending_groups.items():
            if not episodes:
                continue
            env_name = episodes[0].env_name
            if self.eval_envs.get(env_name).requires_group_scoring:
                continue
            bkey = (env_name, eval_step_of(episodes[0]))
            buffered[bkey] = buffered.get(bkey, 0) + self.pending_group_episodes.get(group_id, 0)
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

    def process_group(self, group_id: str) -> None:
        finished = self.pending_groups.pop(group_id, [])
        episodes = self.pending_group_episodes.pop(group_id, 0)
        if not finished:
            return
        # Read the group's facts off an episode, not a trace: every episode in it may have
        # produced none (a whole group cancelled off-policy).
        env_name = finished[0].env_name
        eval_step = eval_step_of(finished[0])
        group = [t for e in finished for t in rollouts_of(e)]
        task_idx = group[0].task.data.idx if group else -1
        bucket = self.pending_batches[(env_name, eval_step)]
        bucket.extend(finished)
        self.pending_batch_episodes[(env_name, eval_step)] += episodes

        survivors = [r for r in group if not r.has_error]
        num_errored = len(group) - len(survivors)
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} eval_step={eval_step} | "
            f"rollouts={len(group)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        """Pop the finished ``(env, eval_step)`` epoch and return the ``EvalBatch`` with its full
        returned cohort (errored rollouts included — the ``all`` set). Metrics are computed
        downstream via ``EvalBatch.rollouts.metrics`` over the all/effective subsets, so the sink
        does no aggregation."""
        env_name, step = key
        episodes = self.pending_batches.pop(key, [])
        self.pending_batch_episodes.pop(key, None)
        return EvalBatch(env_name=env_name, step=step, rollouts=EvalRollouts(episodes))
