"""EvalSink: three-level rollout sink for eval epochs.

Same processing structure as ``TrainSink`` — three ``process_*`` methods —
but the logic at each level is different (eval doesn't tokenize for the
trainer, doesn't compute advantages, doesn't apply filters):

1. ``process_rollout(rollout)`` — no-op. Eval rollouts don't need
   trainer-bound tokenization.
2. ``process_group(key)`` — finalize one per-example group. Moves
   surviving rollouts into the (env, eval_step) batch bucket. Per-example
   ``pass@k`` aggregation happens at the batch level.
3. ``process_batch(key)`` — runs when ``is_batch_done=True`` arrives (the
   dispatcher signals epoch completion on the last expected rollout for an
   (env, eval_step) pair). Builds per-env metrics (reward, completion_len,
   pass@k, no_response, truncation) and returns an ``EvalBatchResult``
   the orchestrator hands to the monitor.

Filters do not apply to eval — filters are a train-only concept.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import verifiers as vf

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_utils import compute_pass_at_k
from prime_rl.orchestrator.vf_utils import get_seq_len
from prime_rl.orchestrator_v2.dispatcher import Rollout


@dataclass
class EvalBatchResult:
    """One env's eval epoch — fully aggregated and ready to log.

    ``rollouts`` is the raw cohort (used for ``monitor.log_eval_samples`` +
    saving to disk). ``metrics`` is the pre-built W&B dict.
    """

    env_name: str
    eval_step: int
    rollouts: list[vf.RolloutOutput]
    metrics: dict[str, Any]


class EvalSink:
    """Three-level eval sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(self, *, eval_envs: EvalEnvs | None) -> None:
        self.eval_envs = eval_envs
        # Per (env_name, example_id, eval_step) accumulation — finalized into
        # the batch bucket on ``is_group_done``.
        self.pending_groups: dict[tuple[str, int, int], list[Rollout]] = defaultdict(list)
        # Per (env_name, eval_step) accumulation — emits an ``EvalBatchResult``
        # on ``is_batch_done``.
        self.pending_batches: dict[tuple[str, int], list[vf.RolloutOutput]] = defaultdict(list)

    # ── ingest ────────────────────────────────────────────────────────────

    def add(self, rollout: Rollout) -> EvalBatchResult | None:
        """Run the per-rollout level (always), per-group level (on
        ``is_group_done``), and per-batch level (on ``is_batch_done``).
        Returns an ``EvalBatchResult`` when an env's epoch is complete.
        """
        assert rollout.kind == "eval", "EvalSink only handles eval rollouts"
        assert rollout.eval_step is not None, "eval Rollout missing eval_step"
        self.process_rollout(rollout)
        gkey = (rollout.env_name, rollout.example_id, rollout.eval_step)
        self.pending_groups[gkey].append(rollout)
        if rollout.is_group_done:
            self.process_group(gkey)
        if rollout.is_batch_done:
            bkey = (rollout.env_name, rollout.eval_step)
            return self.process_batch(bkey)
        return None

    # ── level 1: per-rollout (no-op for eval) ─────────────────────────────

    def process_rollout(self, rollout: Rollout) -> None:
        """No-op. Eval rollouts don't need trainer-bound tokenization; the
        method exists to keep the three-level structure uniform with
        ``TrainSink``.
        """
        return None

    # ── level 2: per-group (move into batch bucket) ───────────────────────

    def process_group(self, key: tuple[str, int, int]) -> None:
        group = self.pending_groups.pop(key, [])
        if not group:
            return
        env_name, _example_id, eval_step = key
        bucket = self.pending_batches[(env_name, eval_step)]
        bucket.extend(r.raw for r in group)

    # ── level 3: per-batch (per-env metrics) ──────────────────────────────

    def process_batch(self, key: tuple[str, int]) -> EvalBatchResult:
        env_name, eval_step = key
        rollouts = self.pending_batches.pop(key, [])
        metrics = self.build_metrics(env_name, eval_step, rollouts)
        return EvalBatchResult(env_name=env_name, eval_step=eval_step, rollouts=rollouts, metrics=metrics)

    def build_metrics(self, env_name: str, eval_step: int, rollouts: list[vf.RolloutOutput]) -> dict[str, Any]:
        """Build the per-env metrics dict. Pass@k computed when the env's
        reward set is binary (subset of {0.0, 1.0})."""
        to_log: dict[str, Any] = {"step": eval_step}
        if not rollouts:
            return to_log

        rewards = [r.get("reward", 0.0) for r in rollouts]
        lens = [get_seq_len(r) for r in rollouts]
        no_response_rate = sum(1 for r in rollouts if not r.get("completion")) / len(rollouts)
        truncation_rate = sum(1 for r in rollouts if r.get("is_truncated")) / len(rollouts)

        group_size = 1
        if self.eval_envs is not None:
            try:
                group_size = self.eval_envs.get(env_name).config.group_size
            except KeyError:
                pass

        prefix = f"eval/{env_name}"
        to_log.update(
            {
                f"{prefix}/avg@{group_size}": float(sum(rewards) / len(rewards)),
                f"{prefix}/reward/mean": float(sum(rewards) / len(rewards)),
                f"{prefix}/completion_len/mean": float(sum(lens) / len(lens)),
                f"{prefix}/completion_len/max": float(max(lens)),
                f"{prefix}/completion_len/min": float(min(lens)),
                f"{prefix}/is_truncated/mean": float(truncation_rate),
                f"{prefix}/no_response/mean": float(no_response_rate),
                f"{prefix}/n_rollouts": float(len(rollouts)),
            }
        )

        # pass@k: reconstruct per-example reward sets from ``example_id``.
        by_example: dict[int, list[float]] = {}
        for r in rollouts:
            by_example.setdefault(r["example_id"], []).append(float(r.get("reward", 0.0)))
        to_log[f"{prefix}/n_examples"] = float(len(by_example))
        unique_rewards = {float(r) for r in rewards}
        if unique_rewards.issubset({0.0, 1.0}) and by_example:
            pass_at_k_per_example = [compute_pass_at_k(rs) for rs in by_example.values()]
            keys = set().union(*(d.keys() for d in pass_at_k_per_example))
            for k in keys:
                values = [d.get(k, 0.0) for d in pass_at_k_per_example]
                to_log[f"{prefix}/{k}"] = float(sum(values) / len(values))

        return to_log
