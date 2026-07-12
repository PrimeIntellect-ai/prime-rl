"""EvalSink: three-level rollout sink for eval epochs.

Same shape as ``TrainSink``, but no tokenization / advantages / filters:

1. ``process_rollout`` — no-op.
2. ``process_group`` — at ``group_size`` arrivals, move the rollouts
   (errored ones included) into the ``(env, eval_step)`` bucket.
3. ``process_batch`` — at ``num_examples × group_size`` arrivals, return an
   ``EvalBatch`` with the full returned cohort (metrics are computed downstream).

``add()`` returns ``EvalBatch | None``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalRollouts
from prime_rl.orchestrator.types import EvalBatch, Rollout
from prime_rl.utils.logger import get_logger


class EvalSink:
    """Constructed only when eval is configured."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        # Bucket size IS the arrival count — ``process_group`` flushes
        # everything in without filtering
        self.pending_batches: dict[tuple[str, int], list[Rollout]] = defaultdict(list)

    def add(self, rollout: Rollout) -> EvalBatch | None:
        """Process one arrival; finalize the group on the ``group_size``-th
        arrival and the per-env epoch on the ``num_examples × group_size``-th."""
        env_name = rollout.env_name
        self.process_rollout(rollout)
        bkey = (env_name, rollout.eval_step)
        self.pending_groups[rollout.group_id].append(rollout)
        if len(self.pending_groups[rollout.group_id]) >= self.group_size_for(env_name):
            self.process_group(rollout.group_id)
        if len(self.pending_batches[bkey]) >= self.batch_size_for(env_name):
            return self.process_batch(bkey)
        return None

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        """``num_examples × group_size`` — total rollouts expected for one
        epoch of ``env_name``."""
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        """One entry per accumulating ``(env, eval_step)`` batch:
        ``(env_name, eval_step, batch_count, expected, buffered)``.
        ``batch_count`` is finalized-group survivors in ``pending_batches``;
        ``buffered`` is partial-group arrivals from non-group-scoring envs."""
        batch_counts: dict[tuple[str, int], int] = {bkey: len(bucket) for bkey, bucket in self.pending_batches.items()}
        buffered: dict[tuple[str, int], int] = {}
        for rollouts in self.pending_groups.values():
            if not rollouts:
                continue
            env_name = rollouts[0].env_name
            if self.eval_envs.get(env_name).requires_group_scoring:
                continue
            bkey = (env_name, rollouts[0].eval_step)
            buffered[bkey] = buffered.get(bkey, 0) + len(rollouts)
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
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return
        env_name = group[0].env_name
        task_idx = group[0].task.data.idx
        eval_step = group[0].eval_step
        bucket = self.pending_batches[(env_name, eval_step)]
        bucket.extend(group)

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
        rollouts = self.pending_batches.pop(key, [])
        return EvalBatch(env_name=env_name, step=step, rollouts=EvalRollouts(rollouts))
