"""MetricsBuilder: assembles the per-step W&B dict. No I/O, no side effects."""

from __future__ import annotations

from typing import Any

import pandas as pd

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import Progress, Rollout, TrainBatchMetrics


class MetricsBuilder:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        step: int,
        rollouts: list[Rollout],
        metrics: TrainBatchMetrics,
        progress: Progress,
        step_time: float,
        save_ckpt_time: float,
        teacher_logprobs_time: float,
        pre_filter_seen: int,
        pre_filter_dropped: int,
        pre_filter_dropped_by_name: dict[str, int],
    ) -> dict[str, Any]:
        """Builds the per-step W&B dict. Stable metric names so
        existing dashboards / alerts keep working."""
        num_rollouts = len(rollouts)
        num_unique_examples = len({r.group_id for r in rollouts})
        num_tokens = sum(r.total_tokens for r in rollouts)

        results_df = pd.DataFrame(
            {
                "group_id": [r.group_id for r in rollouts],
                "task_idx": [r.task.idx for r in rollouts],
                "env_name": [r.env_name for r in rollouts],
                "reward": [r.reward for r in rollouts],
                "is_truncated": [r.is_truncated for r in rollouts],
                "is_filtered": [r.is_filtered for r in rollouts],
                "stop_condition": [r.stop_condition for r in rollouts],
                "seq_len": [r.total_tokens for r in rollouts],
                "prefill_len": metrics.rollout_prefill_lens,
                "decode_len": metrics.rollout_decode_lens,
                "samples_per_rollout": metrics.samples_per_rollout,
                "num_turns": [r.num_turns for r in rollouts],
            }
        )
        metrics_df = pd.DataFrame([r.metrics for r in rollouts])
        filter_df = pd.DataFrame([r.filter_results for r in rollouts])
        timing_df = self.timing_df(rollouts)

        # Each group's full-solve threshold is its own env's group_size (envs
        # can override the top-level group_size).
        env_group_size = {env.resolved_name: env.group_size for env in self.config.train.env}

        def compute_solve_rates(df):
            grouped = df.groupby("group_id")
            reward_per_problem = grouped.reward.sum()
            solve_none = (reward_per_problem == 0).mean()
            expected = grouped.env_name.first().map(env_group_size)
            solve_all = (reward_per_problem == expected).mean()
            return solve_none, solve_all, 1 - solve_none - solve_all

        by_example = results_df.groupby("group_id")
        solve_none, solve_all, effective_batch_size = compute_solve_rates(results_df)

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
            "solve_none/all": solve_none,
            "solve_all/all": solve_all,
            "effective_batch_size/all": effective_batch_size,
            **{f"batch/{env}": r for env, r in results_df.env_name.value_counts(normalize=True).items()},
            "time/step": step_time,
            "time/teacher_logprobs": teacher_logprobs_time,
            "time/save_ckpt": save_ckpt_time,
            "filters/all/is_filtered": results_df.is_filtered.astype(float).mean(),
            **{f"filters/all/{name}": filter_df[name].astype(float).mean() for name in filter_df.columns},
            "step": step,
        }

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
            sn, sa, eb = compute_solve_rates(env_df)
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

        # Dispatcher / watcher gauges live on the ``_timestamp`` axis via
        # the periodic logger — keep this dict step-axis only
        if pre_filter_seen > 0:
            to_log["pre_filters/all/dropped_rate"] = pre_filter_dropped / pre_filter_seen
            for name, count in pre_filter_dropped_by_name.items():
                to_log[f"pre_filters/all/{name}/rate"] = count / pre_filter_seen

        return to_log

    @staticmethod
    def timing_df(rollouts: list[Rollout]) -> pd.DataFrame:
        """Per-rollout timing from the v1 Trace (`generation`/`scoring` spans)."""
        rows = []
        for r in rollouts:
            timing = r.timing
            generation, scoring = timing.generation.duration, timing.scoring.duration
            rows.append({"total": generation + scoring, "generation": generation, "scoring": scoring})
        return pd.DataFrame(rows)
