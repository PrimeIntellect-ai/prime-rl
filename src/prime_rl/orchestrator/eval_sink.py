"""EvalSink: three-level rollout sink for eval epochs.

Same processing structure as ``TrainSink`` — three ``process_*`` methods —
but the logic at each level is different (eval doesn't tokenize for the
trainer, doesn't compute advantages, doesn't apply filters):

1. ``process_rollout(rollout)`` — no-op. Eval rollouts don't need
   trainer-bound tokenization.
2. ``process_group(key)`` — runs when ``group_size`` arrivals are
   accumulated for ``(env, example_id, eval_step)``. Moves the rollouts
   (including errored ones — eval metrics surface error rates) into the
   ``(env, eval_step)`` batch bucket.
3. ``process_batch(key)`` — runs when total arrivals for ``(env, eval_step)``
   reach ``num_examples * group_size`` (full epoch). Returns an ``EvalBatch``
   holding the raw rollouts and the pre-built per-env W&B metrics dict
   (reward, completion_len, pass@k, cancelled/errored counts, ...). All
   per-batch derivation lives here so the orchestrator's role is just to log
   + persist.

The sink owns the boundary signals: ``add()`` returns an ``EvalBatch`` once
the full epoch has arrived, derived purely from counting. The dispatcher
emits every dispatched rollout (success or error) exactly once so the
count-based finalization always fires.

Filters do not apply to eval — filters are a train-only concept.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import verifiers as vf

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_utils import compute_pass_at_k
from prime_rl.orchestrator.types import EvalBatch, Rollout
from prime_rl.orchestrator.vf_utils import get_seq_len
from prime_rl.utils.logger import get_logger


class EvalSink:
    """Three-level eval sink. Constructed once, fed via ``add(rollout)``.

    Construct only when eval is configured — the orchestrator gates this on
    ``config.eval is not None`` so ``eval_envs`` is always present here.
    """

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        # Per (env_name, example_id, eval_step) accumulation — finalized
        # into the batch bucket when arrivals reach ``group_size``.
        self.pending_groups: dict[tuple[str, int, int], list[Rollout]] = defaultdict(list)
        # Per (env_name, eval_step) accumulation — emits an ``EvalBatch``
        # when ``len(bucket) >= num_examples * group_size``. ``process_group``
        # flushes every member of a finalized group into here without
        # filtering, so the bucket size IS the arrival count.
        self.pending_batches: dict[tuple[str, int], list[vf.RolloutOutput]] = defaultdict(list)

    # ── ingest ────────────────────────────────────────────────────────────

    def add(self, rollout: Rollout) -> EvalBatch | None:
        """Process one arrival. Runs the per-rollout step always; finalizes
        the per-example group on the ``group_size``-th arrival for the group,
        and the per-env epoch when total arrivals reach the expected count.
        Returns the ``EvalBatch`` when an env's epoch is complete.
        """
        assert rollout.kind == "eval", "EvalSink only handles eval rollouts"
        assert rollout.eval_step is not None, "eval Rollout missing eval_step"
        self.process_rollout(rollout)
        gkey = (rollout.env_name, rollout.example_id, rollout.eval_step)
        bkey = (rollout.env_name, rollout.eval_step)
        self.pending_groups[gkey].append(rollout)
        if len(self.pending_groups[gkey]) >= self.group_size_for(rollout.env_name):
            self.process_group(gkey)
        if len(self.pending_batches[bkey]) >= self.batch_size_for(rollout.env_name):
            return self.process_batch(bkey)
        return None

    # ── helpers for sink-owned boundary detection ─────────────────────────

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        """Total rollouts the sink expects for one epoch of ``env_name``
        (= ``num_examples * group_size``)."""
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

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
        env_name, example_id, eval_step = key
        bucket = self.pending_batches[(env_name, eval_step)]
        bucket.extend(r.raw for r in group)

        # Per-group summary (eval). Mirrors the train side: one info line
        # per finalized group with error / reward counts. Eval doesn't run
        # filters, so the filter slot is always empty.
        all_raws = [r.raw for r in group]
        survivors = [raw for raw in all_raws if raw.get("error") is None]
        num_errored = len(all_raws) - len(survivors)
        rewards = [raw.get("reward", 0.0) for raw in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().info(
            f"Group | env={env_name} example_id={example_id} eval_step={eval_step} | "
            f"rollouts={len(all_raws)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    # ── level 3: per-batch (per-env metrics) ──────────────────────────────

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        """Build the per-env metrics dict and return the finalized
        ``EvalBatch``. This is the natural per-batch processing hook — all
        per-batch derivations happen here so the orchestrator just has to
        log + persist.

        Errored rollouts (``raw["error"] is not None`` — env-side failures,
        cancellations, task exceptions) are excluded from reward / seq_len /
        pass@k aggregation and surfaced separately as ``cancelled_count`` /
        ``errored_count`` (an errored rollout doesn't represent a real
        evaluation attempt and including it as reward=0 would silently bias
        the score down).
        """
        env_name, step = key
        rollouts = self.pending_batches.pop(key, [])

        to_log: dict[str, Any] = {"step": step}
        prefix = f"eval/{env_name}"
        n_total = len(rollouts)
        n_cancelled = sum(1 for r in rollouts if (r.get("error") or {}).get("error") == "Cancelled")
        n_errored = sum(1 for r in rollouts if r.get("error") is not None) - n_cancelled
        valid = [r for r in rollouts if r.get("error") is None]
        to_log[f"{prefix}/cancelled_count"] = float(n_cancelled)
        to_log[f"{prefix}/errored_count"] = float(n_errored)
        to_log[f"{prefix}/n_rollouts"] = float(n_total)
        to_log[f"{prefix}/valid_rate"] = float(len(valid) / max(n_total, 1))

        if valid:
            rewards = [r.get("reward", 0.0) for r in valid]
            lens = [get_seq_len(r) for r in valid]
            no_response_rate = sum(1 for r in valid if not r.get("completion")) / len(valid)
            truncation_rate = sum(1 for r in valid if r.get("is_truncated")) / len(valid)
            group_size = self.group_size_for(env_name)
            to_log.update(
                {
                    f"{prefix}/avg@{group_size}": float(sum(rewards) / len(rewards)),
                    f"{prefix}/reward/mean": float(sum(rewards) / len(rewards)),
                    f"{prefix}/completion_len/mean": float(sum(lens) / len(lens)),
                    f"{prefix}/completion_len/max": float(max(lens)),
                    f"{prefix}/completion_len/min": float(min(lens)),
                    f"{prefix}/is_truncated/mean": float(truncation_rate),
                    f"{prefix}/no_response/mean": float(no_response_rate),
                }
            )

            # pass@k: reconstruct per-example reward sets from ``example_id``,
            # ignoring errored attempts (they don't count toward k tries).
            by_example: dict[int, list[float]] = {}
            for r in valid:
                by_example.setdefault(r["example_id"], []).append(float(r.get("reward", 0.0)))
            to_log[f"{prefix}/n_examples"] = float(len(by_example))
            unique_rewards = {float(r) for r in rewards}
            if unique_rewards.issubset({0.0, 1.0}) and by_example:
                pass_at_k_per_example = [compute_pass_at_k(rs) for rs in by_example.values()]
                keys = set().union(*(d.keys() for d in pass_at_k_per_example))
                for k in keys:
                    values = [d.get(k, 0.0) for d in pass_at_k_per_example]
                    to_log[f"{prefix}/{k}"] = float(sum(values) / len(values))

        return EvalBatch(env_name=env_name, step=step, rollouts=rollouts, metrics=to_log)
