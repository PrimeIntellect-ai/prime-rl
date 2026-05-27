import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf

from prime_rl.configs.orchestrator import AdvantageConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import backfill_rollout_tokens, interleave_rollout, offload_images_to_disk
from prime_rl.orchestrator.vf_utils import get_seq_len, save_rollouts
from prime_rl.transport import TrainingBatch, TrainingSample
from prime_rl.transport.types import TrainingMode
from prime_rl.utils.logger import get_logger

MAX_POSTPROCESS_WORKERS = 64


@dataclass(slots=True)
class RolloutPostprocessResult:
    training_batch: TrainingBatch | None
    metric_logs: dict[str, Any]
    n_trainable: int
    num_rollouts: int
    num_unique_examples: int
    num_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    parallel_preprocess_time: float
    num_offloaded_images: int
    offload_time: float
    reward_mean: float
    seq_len_mean: float
    rewards: list[float]
    advantages: list[float]


def build_training_batch_and_metrics(
    *,
    train_rollouts: list[vf.RolloutOutput],
    step: int,
    advantage_config: AdvantageConfig | None,
    rollout_filters: list[RolloutFilter],
    output_dir: Path,
    rollout_path: Path,
    tokenizer: Any,
    renderer: Any,
    training_mode: TrainingMode,
    group_size: int,
    mm_token_type_ids_mapping: dict[int, int] | None,
) -> RolloutPostprocessResult:
    """Build trainer payloads and rollout metrics for one scheduler batch."""
    compute_advantages(train_rollouts, advantage_config)
    apply_filters(rollout_filters, train_rollouts)

    num_rollouts = len(train_rollouts)
    num_unique_examples = len({(r["env_name"], r["example_id"]) for r in train_rollouts})
    n_trainable = sum(1 for r in train_rollouts if not r["is_filtered"])
    if n_trainable == 0:
        return _empty_result(
            n_trainable=n_trainable,
            num_rollouts=num_rollouts,
            num_unique_examples=num_unique_examples,
        )

    save_rollouts(train_rollouts, rollout_path, exclude_keys={"trajectory"})

    offload_start = time.perf_counter()
    num_offloaded_images = offload_images_to_disk(train_rollouts, output_dir)
    offload_time = time.perf_counter() - offload_start

    (
        training_batch,
        rollout_prefill_lens,
        rollout_decode_lens,
        rollout_samples_per_rollout,
        num_prefill_tokens,
        num_decode_tokens,
        parallel_preprocess_time,
    ) = _build_training_batch(
        train_rollouts=train_rollouts,
        step=step,
        tokenizer=tokenizer,
        renderer=renderer,
        training_mode=training_mode,
        mm_token_type_ids_mapping=mm_token_type_ids_mapping,
    )

    metric_logs, num_tokens, reward_mean, seq_len_mean = _build_metric_logs(
        train_rollouts=train_rollouts,
        rollout_prefill_lens=rollout_prefill_lens,
        rollout_decode_lens=rollout_decode_lens,
        rollout_samples_per_rollout=rollout_samples_per_rollout,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        num_rollouts=num_rollouts,
        num_unique_examples=num_unique_examples,
        group_size=group_size,
        parallel_preprocess_time=parallel_preprocess_time,
    )

    return RolloutPostprocessResult(
        training_batch=training_batch,
        metric_logs=metric_logs,
        n_trainable=n_trainable,
        num_rollouts=num_rollouts,
        num_unique_examples=num_unique_examples,
        num_tokens=num_tokens,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        parallel_preprocess_time=parallel_preprocess_time,
        num_offloaded_images=num_offloaded_images,
        offload_time=offload_time,
        reward_mean=reward_mean,
        seq_len_mean=seq_len_mean,
        rewards=[r["reward"] for r in train_rollouts],
        advantages=[r["advantage"] for r in train_rollouts],
    )


def _empty_result(
    *,
    n_trainable: int,
    num_rollouts: int,
    num_unique_examples: int,
) -> RolloutPostprocessResult:
    return RolloutPostprocessResult(
        training_batch=None,
        metric_logs={},
        n_trainable=n_trainable,
        num_rollouts=num_rollouts,
        num_unique_examples=num_unique_examples,
        num_tokens=0,
        num_prefill_tokens=0,
        num_decode_tokens=0,
        parallel_preprocess_time=0.0,
        num_offloaded_images=0,
        offload_time=0.0,
        reward_mean=0.0,
        seq_len_mean=0.0,
        rewards=[],
        advantages=[],
    )


def _build_training_batch(
    *,
    train_rollouts: list[vf.RolloutOutput],
    step: int,
    tokenizer: Any,
    renderer: Any,
    training_mode: TrainingMode,
    mm_token_type_ids_mapping: dict[int, int] | None,
) -> tuple[TrainingBatch, list[int], list[int], list[int], int, int, float]:
    parallel_preprocess_start = time.perf_counter()

    needs_backfill = any(step["tokens"] is None for rollout in train_rollouts for step in rollout["trajectory"])
    if needs_backfill:
        get_logger().info(
            "Backfilling tokens for rollout trajectories (expected for training_mode=sft against an external teacher API)"
        )
        _map_rollouts_in_threads(
            lambda rollout: backfill_rollout_tokens(
                rollout,
                tokenizer,
                renderer=renderer,
            ),
            train_rollouts,
        )

    results = _map_rollouts_in_threads(
        lambda rollout: interleave_rollout(rollout, mm_token_type_ids_mapping=mm_token_type_ids_mapping),
        train_rollouts,
    )

    train_examples: list[TrainingSample] = []
    rollout_prefill_lens: list[int] = []
    rollout_decode_lens: list[int] = []
    rollout_samples_per_rollout: list[int] = []
    num_prefill_tokens = 0
    num_decode_tokens = 0
    for rollout, samples in zip(train_rollouts, results):
        rollout_prefill_tokens = 0
        rollout_decode_tokens = 0
        if samples is None:
            samples = []
        rollout_samples_per_rollout.append(len(samples))
        for sample in samples:
            sample.advantage = rollout["advantage"]
            sample.reward = rollout["reward"]
            sample.env_name = rollout["env_name"]
            sample.training_mode = training_mode
            sample_decode_tokens = sum(sample.completion_mask)
            sample_prefill_tokens = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode_tokens
            rollout_decode_tokens += sample_decode_tokens
            rollout_prefill_tokens += sample_prefill_tokens
            if not rollout["is_filtered"]:
                train_examples.append(sample)
        rollout_prefill_lens.append(rollout_prefill_tokens)
        rollout_decode_lens.append(rollout_decode_tokens)
        num_prefill_tokens += rollout_prefill_tokens
        num_decode_tokens += rollout_decode_tokens

    parallel_preprocess_time = time.perf_counter() - parallel_preprocess_start

    return (
        TrainingBatch(
            examples=train_examples,
            step=step,
        ),
        rollout_prefill_lens,
        rollout_decode_lens,
        rollout_samples_per_rollout,
        num_prefill_tokens,
        num_decode_tokens,
        parallel_preprocess_time,
    )


def _map_rollouts_in_threads(fn, rollouts: list[vf.RolloutOutput]) -> list[Any]:
    if len(rollouts) <= 1:
        return [fn(rollout) for rollout in rollouts]

    with ThreadPoolExecutor(max_workers=min(MAX_POSTPROCESS_WORKERS, len(rollouts))) as executor:
        return list(executor.map(fn, rollouts))


def _build_metric_logs(
    *,
    train_rollouts: list[vf.RolloutOutput],
    rollout_prefill_lens: list[int],
    rollout_decode_lens: list[int],
    rollout_samples_per_rollout: list[int],
    num_prefill_tokens: int,
    num_decode_tokens: int,
    num_rollouts: int,
    num_unique_examples: int,
    group_size: int,
    parallel_preprocess_time: float,
) -> tuple[dict[str, Any], int, float, float]:
    results_df = pd.DataFrame(
        {
            "example_id": [rollout["example_id"] for rollout in train_rollouts],
            "env_name": [rollout["env_name"] for rollout in train_rollouts],
            "reward": [rollout["reward"] for rollout in train_rollouts],
            "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
            "is_filtered": [rollout["is_filtered"] for rollout in train_rollouts],
            "stop_condition": [rollout.get("stop_condition") for rollout in train_rollouts],
            "seq_len": [get_seq_len(rollout) for rollout in train_rollouts],
            "prefill_len": rollout_prefill_lens,
            "decode_len": rollout_decode_lens,
            "samples_per_rollout": rollout_samples_per_rollout,
            "num_turns": [len(rollout["trajectory"]) for rollout in train_rollouts],
        }
    )

    metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])
    filter_df = pd.DataFrame([rollout["filters"] for rollout in train_rollouts])
    timing_df = pd.DataFrame(
        [
            {
                "total": rollout["timing"]["total"],
                "setup": rollout["timing"]["setup"]["duration"],
                "generation": rollout["timing"]["generation"]["duration"],
                "model": rollout["timing"]["model"]["duration"],
                "env": rollout["timing"]["env"]["duration"],
                "scoring": rollout["timing"]["scoring"]["duration"],
                "overhead": rollout["timing"]["overhead"],
            }
            for rollout in train_rollouts
        ]
    )

    num_tokens = int(results_df.seq_len.sum())

    def compute_solve_rates(df):
        reward_per_problem = df.groupby(["env_name", "example_id"]).reward.sum()
        solve_none = (reward_per_problem == 0).mean()
        solve_all = (reward_per_problem == group_size).mean()
        return solve_none, solve_all, 1 - solve_none - solve_all

    by_example = results_df.groupby(["env_name", "example_id"])

    solve_none, solve_all, effective_batch_size = compute_solve_rates(results_df)
    to_log = {
        "progress/tokens": num_tokens,
        "progress/prefill_tokens": num_prefill_tokens,
        "progress/decode_tokens": num_decode_tokens,
        "progress/samples": num_rollouts,
        "progress/problems": num_unique_examples,
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
                timing_df[key].groupby([results_df.env_name, results_df.example_id]).mean(),
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
        "time/parallel_preprocess": parallel_preprocess_time,
        "filters/all/is_filtered": results_df.is_filtered.astype(float).mean(),
        **{f"filters/all/{name}": filter_df[name].astype(float).mean() for name in filter_df.columns},
    }

    per_env_columns = [
        "seq_len",
        "prefill_len",
        "decode_len",
        "is_truncated",
        "samples_per_rollout",
        "num_turns",
    ]

    for env, env_df in results_df.groupby("env_name"):
        env_by_example = env_df.groupby("example_id")
        for col in per_env_columns:
            to_log[f"{col}/{env}/mean"] = env_by_example[col].mean().mean()
            to_log[f"{col}/{env}/max"] = env_by_example[col].mean().max()
            if col != "is_truncated":
                to_log[f"{col}/{env}/min"] = env_by_example[col].mean().min()
        env_timing_df = timing_df.loc[env_df.index]
        for key in timing_df.columns:
            per_example = env_timing_df.groupby(env_df["example_id"])[key].mean()
            to_log[f"timing/{env}/{key}/mean"] = per_example.mean()
            to_log[f"timing/{env}/{key}/max"] = per_example.max()
            to_log[f"timing/{env}/{key}/min"] = per_example.min()
        to_log[f"reward/{env}/mean"] = env_by_example.reward.mean().mean()
        to_log[f"reward/{env}/max"] = env_by_example.reward.mean().max()
        to_log[f"reward/{env}/min"] = env_by_example.reward.mean().min()
        solve_none, solve_all, effective_batch_size = compute_solve_rates(env_df)
        to_log[f"solve_none/{env}"] = solve_none
        to_log[f"solve_all/{env}"] = solve_all
        to_log[f"effective_batch_size/{env}"] = effective_batch_size
        to_log[f"stop_condition/{env}/generation_truncated"] = (
            env_df.is_truncated & (env_df.stop_condition != "prompt_too_long")
        ).mean()
        for sc, rate in env_df.stop_condition.dropna().value_counts(normalize=True).items():
            to_log[f"stop_condition/{env}/{sc}"] = rate
        env_metrics_df = metrics_df.loc[env_df.index]
        for metric in metrics_df.columns:
            to_log[f"metrics/{env}/{metric}"] = env_metrics_df.groupby(env_df["example_id"])[metric].mean().mean()
        to_log[f"filters/{env}/is_filtered"] = env_df.is_filtered.astype(float).mean()
        env_filter_df = filter_df.loc[env_df.index]
        for name in filter_df.columns:
            to_log[f"filters/{env}/{name}"] = env_filter_df[name].astype(float).mean()

    return to_log, num_tokens, by_example.reward.mean().mean(), by_example.seq_len.mean().mean()
