"""MetricsBuilder: assembles the per-step W&B dict.

The only I/O / state is the trainer's token-export metrics, which lag the
orchestrator: ``build`` folds in the oldest unlogged stable step it finds on
disk and tracks the last step logged so it never re-logs one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.token_export_metrics import collect_next_token_export_metrics
from prime_rl.orchestrator.types import Progress, TrainBatchMetrics, TrainRollout


class MetricsBuilder:
    def __init__(self, config: OrchestratorConfig, start_step: int = 0) -> None:
        self.config = config
        # Token exports are read back one step behind the orchestrator; track the
        # last run step whose metrics we've folded in so we never re-log a step.
        self._last_token_export_step_logged = start_step - 1

    def build(
        self,
        *,
        step: int,
        rollouts: list[TrainRollout],
        metrics: TrainBatchMetrics,
        progress: Progress,
        step_time: float,
        save_ckpt_time: float,
        teacher_logprobs_time: float,
        pre_filter_seen: int,
        pre_filter_dropped: int,
        pre_filter_dropped_by_name: dict[str, int],
        bridge_metrics: Mapping[str, int],
    ) -> dict[str, Any]:
        """Builds the per-step W&B dict. Stable metric names so
        existing dashboards / alerts keep working.

        ``bridge_metrics`` is a snapshot of the verifiers renderer client's
        process-global TITO bridge counters (attempts/successes/failures)."""
        num_rollouts = len(rollouts)
        num_unique_examples = len({r.group_id for r in rollouts})
        num_tokens = sum(
            r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"] for r in rollouts
        )

        results_df = pd.DataFrame(
            {
                "group_id": [r.group_id for r in rollouts],
                "example_id": [r.example_id for r in rollouts],
                "env_name": [r.env_name for r in rollouts],
                "reward": [r.reward for r in rollouts],
                "is_truncated": [r.is_truncated for r in rollouts],
                "is_filtered": [r.is_filtered for r in rollouts],
                "stop_condition": [r.raw.get("stop_condition") for r in rollouts],
                "seq_len": [
                    r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
                    for r in rollouts
                ],
                "prefill_len": metrics.rollout_prefill_lens,
                "decode_len": metrics.rollout_decode_lens,
                "samples_per_rollout": metrics.samples_per_rollout,
                "num_turns": [len(r.raw["trajectory"]) for r in rollouts],
                "member_id": [r.raw.get("member_id") for r in rollouts],
                "advantage": [r.advantage for r in rollouts],
            }
        )
        metrics_df = pd.DataFrame([(r.raw.get("metrics") or {}) for r in rollouts])
        filter_df = pd.DataFrame([r.filter_results for r in rollouts])
        length_penalty_df = pd.DataFrame([(r.raw.get("length_penalty") or {}) for r in rollouts])
        timing_df = self.timing_df(rollouts)

        # Each group's full-solve threshold is its own env's group_size (envs
        # can override the top-level group_size).
        env_group_size = {env.resolved_name: env.group_size for env in self.config.train.env}

        def compute_solve_rates(df):
            """solve_none/solve_all/effective_batch_size are GRPO-group
            metrics: each row is an independent attempt and a group's reward
            sum reaching ``group_size`` means fully solved. Multi-agent
            member rows are correlated zero-sum projections of one episode
            (their rewards sum to ~0 by construction), so no group-solve
            semantics exists for them — they are excluded, and the keys are
            omitted entirely when a slice has only member rows."""
            df = df[df.member_id.isna()]
            if df.empty:
                return None
            grouped = df.groupby("group_id")
            reward_per_problem = grouped.reward.sum()
            solve_none = (reward_per_problem == 0).mean()
            expected = grouped.env_name.first().map(env_group_size)
            solve_all = (reward_per_problem == expected).mean()
            return solve_none, solve_all, 1 - solve_none - solve_all

        by_example = results_df.groupby("group_id")
        solve_rates = compute_solve_rates(results_df)

        to_log: dict[str, Any] = {
            "progress/tokens": num_tokens,
            "progress/prefill_tokens": metrics.num_prefill_tokens,
            "progress/decode_tokens": metrics.num_decode_tokens,
            "progress/samples": num_rollouts,
            "progress/problems": num_unique_examples,
            "progress/total_tokens": progress.total_tokens,
            "progress/total_samples": progress.total_samples,
            "progress/total_problems": progress.total_problems,
            "seq_len/all/mean": by_example.seq_len.mean().mean(),
            "seq_len/all/max": by_example.seq_len.mean().max(),
            "seq_len/all/min": by_example.seq_len.mean().min(),
            "prefill_len/all/mean": by_example.prefill_len.mean().mean(),
            "prefill_len/all/max": by_example.prefill_len.mean().max(),
            "prefill_len/all/min": by_example.prefill_len.mean().min(),
            "decode_len/all/mean": by_example.decode_len.mean().mean(),
            "decode_len/all/max": by_example.decode_len.mean().max(),
            "decode_len/all/min": by_example.decode_len.mean().min(),
            "is_truncated/all/mean": by_example.is_truncated.mean().mean(),
            "is_truncated/all/max": by_example.is_truncated.mean().max(),
            "stop_condition/all/generation_truncated": (
                results_df.is_truncated & (results_df.stop_condition != "prompt_too_long")
            ).mean(),
            **{
                f"stop_condition/all/{sc}": rate
                for sc, rate in results_df.stop_condition.dropna().value_counts(normalize=True).items()
            },
            "samples_per_rollout/all/mean": by_example.samples_per_rollout.mean().mean(),
            "samples_per_rollout/all/max": by_example.samples_per_rollout.mean().max(),
            "samples_per_rollout/all/min": by_example.samples_per_rollout.mean().min(),
            "num_turns/all/mean": by_example.num_turns.mean().mean(),
            "num_turns/all/max": by_example.num_turns.mean().max(),
            "num_turns/all/min": by_example.num_turns.mean().min(),
            **{
                f"timing/all/{key}/{stat}": getattr(
                    timing_df[key].groupby(results_df.group_id).mean(),
                    stat,
                )()
                for key in timing_df.columns
                for stat in ("mean", "max", "min")
            },
            "reward/all/mean": by_example.reward.mean().mean(),
            "reward/all/max": by_example.reward.mean().max(),
            "reward/all/min": by_example.reward.mean().min(),
            **{f"batch/{env}": r for env, r in results_df.env_name.value_counts(normalize=True).items()},
            "time/step": step_time,
            "time/teacher_logprobs": teacher_logprobs_time,
            "time/save_ckpt": save_ckpt_time,
            "filters/all/is_filtered": results_df.is_filtered.astype(float).mean(),
            **{f"filters/all/{name}": filter_df[name].astype(float).mean() for name in filter_df.columns},
            "step": step,
        }
        if solve_rates is not None:
            solve_none, solve_all, effective_batch_size = solve_rates
            to_log["solve_none/all"] = solve_none
            to_log["solve_all/all"] = solve_all
            to_log["effective_batch_size/all"] = effective_batch_size

        if not length_penalty_df.empty and "penalty" in length_penalty_df.columns:
            penalty = pd.to_numeric(length_penalty_df["penalty"], errors="coerce")
            annotated = penalty.notna()
            if annotated.any():
                aux = pd.to_numeric(length_penalty_df["aux"], errors="coerce")
                eligible = (
                    length_penalty_df.get(
                        "eligible",
                        pd.Series(False, index=length_penalty_df.index),
                    )
                    .fillna(False)
                    .astype(bool)
                )
                sign_flipped = (
                    length_penalty_df.get(
                        "sign_flipped",
                        pd.Series(False, index=length_penalty_df.index),
                    )
                    .fillna(False)
                    .astype(bool)
                )
                eligible_annotated = annotated & eligible
                to_log["length_penalty/eligible_share"] = eligible[annotated].mean()
                to_log["length_penalty/eligible_truncated_rate"] = (
                    results_df.loc[eligible_annotated, "is_truncated"].mean() if eligible_annotated.any() else 0.0
                )
                to_log["length_penalty/penalty_mean"] = penalty[annotated].mean()
                to_log["length_penalty/penalty_max"] = penalty[annotated].max()
                to_log["length_penalty/aux_abs_max"] = aux[annotated].abs().max()
                to_log["length_penalty/sign_flip_rate"] = sign_flipped[annotated].mean()

        # Per-env metrics
        per_env_columns = [
            "seq_len",
            "prefill_len",
            "decode_len",
            "is_truncated",
            "samples_per_rollout",
            "num_turns",
        ]
        for env, env_df in results_df.groupby("env_name"):
            env_by_example = env_df.groupby("group_id")
            for col in per_env_columns:
                to_log[f"{col}/{env}/mean"] = env_by_example[col].mean().mean()
                to_log[f"{col}/{env}/max"] = env_by_example[col].mean().max()
                if col != "is_truncated":
                    to_log[f"{col}/{env}/min"] = env_by_example[col].mean().min()
            env_timing_df = timing_df.loc[env_df.index]
            for key in timing_df.columns:
                per_example = env_timing_df.groupby(env_df["group_id"])[key].mean()
                to_log[f"timing/{env}/{key}/mean"] = per_example.mean()
                to_log[f"timing/{env}/{key}/max"] = per_example.max()
                to_log[f"timing/{env}/{key}/min"] = per_example.min()
            to_log[f"reward/{env}/mean"] = env_by_example.reward.mean().mean()
            to_log[f"reward/{env}/max"] = env_by_example.reward.mean().max()
            to_log[f"reward/{env}/min"] = env_by_example.reward.mean().min()
            env_solve_rates = compute_solve_rates(env_df)
            if env_solve_rates is not None:
                sn, sa, eb = env_solve_rates
                to_log[f"solve_none/{env}"] = sn
                to_log[f"solve_all/{env}"] = sa
                to_log[f"effective_batch_size/{env}"] = eb
            to_log[f"stop_condition/{env}/generation_truncated"] = (
                env_df.is_truncated & (env_df.stop_condition != "prompt_too_long")
            ).mean()
            for sc, rate in env_df.stop_condition.dropna().value_counts(normalize=True).items():
                to_log[f"stop_condition/{env}/{sc}"] = rate
            env_metrics_df = metrics_df.loc[env_df.index] if not metrics_df.empty else metrics_df
            for metric in metrics_df.columns:
                to_log[f"metrics/{env}/{metric}"] = env_metrics_df.groupby(env_df["group_id"])[metric].mean().mean()
            to_log[f"filters/{env}/is_filtered"] = env_df.is_filtered.astype(float).mean()
            env_filter_df = filter_df.loc[env_df.index] if not filter_df.empty else filter_df
            for name in filter_df.columns:
                to_log[f"filters/{env}/{name}"] = env_filter_df[name].astype(float).mean()

        # Multi-agent member rows (fan-out products carrying ``member_id``):
        # per-member reward/advantage and the trainable-row balance across
        # members (shares sum to 1 — seat imbalance shows up directly)
        ma_df = results_df[results_df.member_id.notna()]
        if not ma_df.empty:
            num_episodes = len({r.source_rollout_id for r in rollouts if r.raw.get("member_id") is not None})
            to_log["multi_agent/member_rows"] = len(ma_df)
            to_log["multi_agent/fan_out_factor"] = len(ma_df) / num_episodes
            trainable_ma = ma_df[~ma_df.is_filtered]
            for member_id, member_df in ma_df.groupby("member_id"):
                to_log[f"multi_agent/{member_id}/reward_mean"] = member_df.reward.mean()
                to_log[f"multi_agent/{member_id}/advantage_mean"] = member_df.advantage.astype(float).mean()
                if len(trainable_ma) > 0:
                    to_log[f"multi_agent/{member_id}/trainable_row_share"] = float(
                        (trainable_ma.member_id == member_id).mean()
                    )

        rae_stats = metrics.rae_stats
        if rae_stats is not None and rae_stats.updates > 0:
            to_log["multi_agent/rae/cold_key_fraction"] = rae_stats.cold_updates / rae_stats.updates
            to_log["multi_agent/rae/baseline_drift_mean"] = rae_stats.baseline_abs_delta_sum / rae_stats.updates
            to_log["multi_agent/rae/baseline_keys_total"] = rae_stats.baseline_keys_total
            for member_id, baseline_sum in rae_stats.baseline_sum_by_member.items():
                to_log[f"multi_agent/{member_id}/rae_baseline_mean"] = (
                    baseline_sum / rae_stats.updates_by_member[member_id]
                )

        # TITO bridge counters (process-global in the verifiers renderer
        # client) — a miss silently degrades to a full O(T²) re-render. The
        # window covers everything since the previous snapshot, including
        # concurrent eval traffic.
        bridge_attempts = bridge_metrics["attempts"]
        if bridge_attempts > 0:
            to_log["bridge/attempt_count"] = bridge_attempts
            to_log["bridge/hit_rate"] = bridge_metrics["successes"] / bridge_attempts
            to_log["bridge/miss_count"] = bridge_metrics["failures"]

        # Dispatcher / watcher gauges live on the ``_timestamp`` axis via
        # the periodic logger — keep this dict step-axis only
        if pre_filter_seen > 0:
            to_log["pre_filters/all/dropped_rate"] = pre_filter_dropped / pre_filter_seen
            for name, count in pre_filter_dropped_by_name.items():
                to_log[f"pre_filters/all/{name}/rate"] = count / pre_filter_seen

        # Fold in the trainer's token-export metrics for the oldest unlogged stable
        # step (exports always lag the orchestrator, so this is a past step).
        to_log.update(self.next_token_export_metrics())

        return to_log

    @property
    def last_token_export_step_logged(self) -> int:
        return self._last_token_export_step_logged

    def next_token_export_metrics(self) -> dict[str, float | int]:
        """Log-ready token-export metrics for the next unlogged stable step (empty if
        none), advancing the cursor. They arrive under trainer/, including trainer/step
        (the run step they belong to) for plotting against a lag-corrected axis. Shared
        by build() and the orchestrator's end-of-run drain."""
        token_export = collect_next_token_export_metrics(
            self.config.output_dir,
            last_logged_step=self._last_token_export_step_logged,
        )
        if token_export:
            self._last_token_export_step_logged = token_export["trainer/step"]
        return token_export

    @staticmethod
    def timing_df(rollouts: list[TrainRollout]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "total": r.raw["timing"]["total"],
                    "setup": r.raw["timing"]["setup"]["duration"],
                    "generation": r.raw["timing"]["generation"]["duration"],
                    "model": r.raw["timing"]["model"]["duration"],
                    "env": r.raw["timing"]["env"]["duration"],
                    "scoring": r.raw["timing"]["scoring"]["duration"],
                    "overhead": r.raw["timing"]["overhead"],
                }
                for r in rollouts
            ]
        )
