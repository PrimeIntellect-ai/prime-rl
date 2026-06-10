"""EvalSink: three-level rollout sink for eval epochs.

Same shape as ``TrainSink``, but no tokenization / advantages / filters:

1. ``process_rollout`` — no-op.
2. ``process_group`` — at ``group_size`` arrivals, move the rollouts
   (errored ones included) into the ``(env, eval_step)`` bucket.
3. ``process_batch`` — at ``num_examples × group_size`` arrivals, build
   the ``EvalBatchMetrics`` and return an ``EvalBatch``.

``add()`` returns ``EvalBatch | None``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

import verifiers as vf

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_utils import compute_pass_at_k
from prime_rl.orchestrator.types import EvalBatch, EvalBatchMetrics, EvalRollout
from prime_rl.utils.logger import get_logger


def aggregate_mar_panel(rollouts: list[vf.RolloutOutput]) -> tuple[dict[str, float], dict[str, int]]:
    """Multi-agent MARScore panel for one eval batch.

    Returns ``(mar_metrics, winner_counts)``: per-key means of
    ``MARScore.to_metrics_flat()`` over the rollouts carrying a ``mar_score``
    (per-member rewards, parse errors, member metrics, episode metrics), and
    the judge-winner distribution from ``episode_categorical["winner"]``
    (ties included; ``None`` buckets as ``"none"``). Single-agent rollouts
    carry no ``mar_score`` and contribute nothing — both dicts come back
    empty for non-multi-agent envs."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    winner_counts: dict[str, int] = {}
    for rollout in rollouts:
        mar_raw = rollout.get("mar_score")
        if mar_raw is None:
            continue
        mar = vf.MARScore.model_validate(mar_raw)
        for key, value in mar.to_metrics_flat().items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
        if "winner" in mar.episode_categorical:
            winner = mar.episode_categorical["winner"] or "none"
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}, winner_counts


class EvalSink:
    """Constructed only when eval is configured."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[uuid.UUID, list[EvalRollout]] = defaultdict(list)
        # Bucket size IS the arrival count — ``process_group`` flushes
        # everything in without filtering
        self.pending_batches: dict[tuple[str, int], list[EvalRollout]] = defaultdict(list)

    def add(self, rollout: EvalRollout) -> EvalBatch | None:
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

    def process_rollout(self, rollout: EvalRollout) -> None:
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
        example_id = group[0].example_id
        eval_step = group[0].eval_step
        bucket = self.pending_batches[(env_name, eval_step)]
        bucket.extend(group)

        survivors = [r for r in group if r.error is None]
        num_errored = len(group) - len(survivors)
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={env_name} example_id={example_id} eval_step={eval_step} | "
            f"rollouts={len(group)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        """Build ``EvalBatchMetrics`` and return the finalized ``EvalBatch``.
        Errored rollouts (env failures, cancellations, task exceptions) are
        excluded from reward / pass@k / seq_len aggregation (including them
        at reward=0 would bias the score down) and surfaced separately as
        ``n_cancelled`` / ``n_errored``."""
        env_name, step = key
        rollouts = self.pending_batches.pop(key, [])

        n_total = len(rollouts)
        n_cancelled = sum(1 for r in rollouts if (r.error or {}).get("error") == "Cancelled")
        n_errored = sum(1 for r in rollouts if r.error is not None) - n_cancelled
        valid = [r for r in rollouts if r.error is None]
        metrics = EvalBatchMetrics(
            n_rollouts=n_total,
            n_cancelled=n_cancelled,
            n_errored=n_errored,
        )

        if valid:
            rewards = [r.reward for r in valid]
            lens = [r.raw["token_usage"]["final_output_tokens"] for r in valid]
            metrics.group_size = self.group_size_for(env_name)
            metrics.reward_mean = float(sum(rewards) / len(rewards))
            metrics.completion_len_mean = float(sum(lens) / len(lens))
            metrics.completion_len_max = float(max(lens))
            metrics.completion_len_min = float(min(lens))
            metrics.truncation_rate = float(sum(1 for r in valid if r.is_truncated) / len(valid))
            metrics.no_response_rate = float(sum(1 for r in valid if not r.raw.get("completion")) / len(valid))
            num_turns = [len(r.raw.get("trajectory") or []) for r in valid]
            metrics.num_turns_mean = float(sum(num_turns) / len(num_turns))
            metrics.num_turns_min = float(min(num_turns))
            metrics.num_turns_max = float(max(num_turns))

            # MARScore panel (multi-agent envs only; no-op otherwise).
            # Inert-scalar is the env's structural declaration (symmetric
            # zero-sum debate ⇒ episode scalar ≡ 0.0 by construction), NOT a
            # runtime all-zero observation — a truth_member pack can honestly
            # score 0.0 on a batch, and that zero must stay visible.
            # ``to_wandb_dict`` omits avg@k / pass@k when inert.
            metrics.mar_metrics, metrics.winner_counts = aggregate_mar_panel([r.raw for r in valid])
            metrics.inert_scalar = self.eval_envs.get(env_name).has_inert_episode_scalar

            # Env-emitted rollout metrics (e.g. debate diagnostics),
            # forwarded key-agnostically: mean per numeric key over the
            # rollouts carrying that key. Non-numeric values are not metrics
            # and are skipped.
            metric_sums: dict[str, float] = {}
            metric_counts: dict[str, int] = {}
            for r in valid:
                for key, value in (r.raw.get("metrics") or {}).items():
                    if isinstance(value, (bool, int, float)):
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                        metric_counts[key] = metric_counts.get(key, 0) + 1
            metrics.env_metrics = {key: metric_sums[key] / metric_counts[key] for key in metric_sums}

            # pass@k: errored attempts don't count toward k tries
            by_example: dict[int | str, list[float]] = {}
            for r in valid:
                by_example.setdefault(r.example_id, []).append(r.reward)
            metrics.n_examples = len(by_example)
            unique_rewards = {float(r) for r in rewards}
            if unique_rewards.issubset({0.0, 1.0}) and by_example:
                pass_at_k_per_example = [compute_pass_at_k(rs) for rs in by_example.values()]
                keys = set().union(*(d.keys() for d in pass_at_k_per_example))
                for k in keys:
                    values = [d[k] for d in pass_at_k_per_example if k in d]
                    metrics.pass_at_k[k] = float(sum(values) / len(values))

        return EvalBatch(env_name=env_name, step=step, rollouts=rollouts, metrics=metrics)
