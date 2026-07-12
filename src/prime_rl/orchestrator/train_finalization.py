"""Ordered persistence, shipment, and reporting for one train batch."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from prime_rl.orchestrator.utils import save_rollouts, trim_process_memory
from prime_rl.transport import TrainingBatch
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_trace_path

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.orchestrator.dispatcher import RolloutDispatcher
    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.metrics import TrainRollouts
    from prime_rl.orchestrator.train_sink import TrainSink
    from prime_rl.orchestrator.types import Progress, TrainBatch
    from prime_rl.utils.heartbeat import Heartbeat
    from prime_rl.utils.monitor.base import Monitor
    from prime_rl.utils.usage_reporter import UsageReporter

MAX_CONSECUTIVE_EMPTY_BATCHES = 10


class TrainFinalizationHost(Protocol):
    config: OrchestratorConfig
    progress: Progress
    dispatcher: RolloutDispatcher
    train_envs: TrainEnvs
    train_sink: TrainSink
    monitor: Monitor
    usage_reporter: UsageReporter | None
    heart: Heartbeat | None
    last_batch_at: float | None
    consecutive_empty_batches: int
    draining: bool
    wait_for_policy_time: float

    async def _send_to_trainer(self, batch: TrainingBatch) -> None: ...

    def update_dispatch_gate(self) -> None: ...

    async def maybe_save_ckpt(self, step: int) -> float: ...

    def maybe_trigger_eval(self, step: int) -> None: ...


@dataclass(frozen=True)
class TrainStepReport:
    metrics: dict[str, float]
    num_tokens: int
    num_input: int
    num_output: int
    num_rollouts: int
    num_unique_examples: int


async def finalize_train_batch(host: TrainFinalizationHost, batch: TrainBatch) -> None:
    """Preserve the step's persist → send → checkpoint → report transaction."""
    step = host.progress.step
    step_time = _start_step_clock(host)
    if await _skip_unshippable_batch(host, batch, step):
        return
    effective, save_ckpt_time = await _persist_and_ship(host, batch, step)
    report = _build_train_step_report(host, batch, effective, step, step_time, save_ckpt_time)
    _publish_train_step_report(host, batch, effective, report, step, step_time)


def _start_step_clock(host: TrainFinalizationHost) -> float:
    now = time.perf_counter()
    step_time = (now - host.last_batch_at) if host.last_batch_at is not None else 0.0
    host.last_batch_at = now
    return step_time


async def _skip_unshippable_batch(host: TrainFinalizationHost, batch: TrainBatch, step: int) -> bool:
    if host.config.max_steps is not None and step > host.config.max_steps:
        host.draining = True
        host.dispatcher.disable_train_scheduling()
        n_cancelled = await host.dispatcher.cancel_inflight_train_rollouts()
        get_logger().info(
            f"Draining pipeline (cancelled {n_cancelled} in-flight train rollout(s); any in-flight evals will complete)"
        )
        return True
    if not batch.samples:
        host.consecutive_empty_batches += 1
        get_logger().warning(
            f"Step {step}: empty train batch (0 of {len(batch.rollouts)} generated rollouts shipped — "
            f"all errored or filtered out) (consecutive empty batches: "
            f"{host.consecutive_empty_batches}/{MAX_CONSECUTIVE_EMPTY_BATCHES})"
        )
        if host.consecutive_empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
            raise RuntimeError(
                f"{host.consecutive_empty_batches} consecutive empty train batches — "
                "check filter config (pre_batch_filters / post_batch_filters) or task difficulty."
            )
        return True
    host.consecutive_empty_batches = 0
    n_trainable = sum(1 for rollout in batch.rollouts if rollout.is_trainable)
    if n_trainable / len(batch.rollouts) <= 0.1:
        get_logger().warning(
            f"Only {n_trainable}/{len(batch.rollouts)} generated rollouts are trainable "
            f"({n_trainable / len(batch.rollouts):.1%}) — consider reviewing task difficulty / filter config"
        )
    return False


async def _persist_and_ship(
    host: TrainFinalizationHost,
    batch: TrainBatch,
    step: int,
) -> tuple[TrainRollouts, float]:
    effective = batch.rollouts.effective
    records = [rollout.to_record() for rollout in effective]
    await asyncio.to_thread(
        save_rollouts,
        records,
        get_trace_path(host.config.output_dir, step, "train", "effective"),
    )
    await host._send_to_trainer(TrainingBatch(examples=batch.samples, step=step))
    host.progress.step += 1
    host.update_dispatch_gate()
    save_ckpt_time = await host.maybe_save_ckpt(step)
    trim_process_memory()
    return effective, save_ckpt_time


def _build_train_step_report(
    host: TrainFinalizationHost,
    batch: TrainBatch,
    effective: TrainRollouts,
    step: int,
    step_time: float,
    save_ckpt_time: float,
) -> TrainStepReport:
    metrics: dict[str, float] = {}
    for subset, pool in (("all", batch.rollouts), ("effective", effective)):
        metrics |= pool.metrics.to_wandb(prefix="train/agg", subset=subset)
        for env_name, env_pool in pool.by_env().items():
            metrics |= env_pool.metrics.to_wandb(prefix=f"train/{env_name}", subset=subset)
    num_tokens = sum(rollout.num_total_tokens for rollout in batch.rollouts)
    num_input = sum(rollout.num_input_tokens for rollout in effective)
    num_output = sum(rollout.num_output_tokens for rollout in effective)
    num_rollouts = len(batch.rollouts)
    num_unique_examples = len({rollout.group_id for rollout in batch.rollouts})
    metrics |= {
        "progress/tokens": num_tokens,
        "progress/input_tokens": num_input,
        "progress/output_tokens": num_output,
        "progress/rollouts": num_rollouts,
        "progress/tasks": num_unique_examples,
        "progress/total_tokens": host.progress.total_tokens,
        "progress/total_rollouts": host.progress.total_samples,
        "progress/total_tasks": host.progress.total_problems,
        "time/step": step_time,
        "time/save_ckpt": save_ckpt_time,
        "time/wait_for_policy": host.wait_for_policy_time,
        "step": step,
    }
    for env_name, env_pool in batch.rollouts.by_env().items():
        metrics[f"batch/{env_name}"] = len(env_pool) / num_rollouts
    if host.train_sink.pre_filter_seen > 0:
        metrics["pre_filters/all/dropped_rate"] = host.train_sink.pre_filter_dropped / host.train_sink.pre_filter_seen
        for name, count in host.train_sink.pre_filter_dropped_by_name.items():
            metrics[f"pre_filters/all/{name}/rate"] = count / host.train_sink.pre_filter_seen
    return TrainStepReport(metrics, num_tokens, num_input, num_output, num_rollouts, num_unique_examples)


def _publish_train_step_report(
    host: TrainFinalizationHost,
    batch: TrainBatch,
    effective: TrainRollouts,
    report: TrainStepReport,
    step: int,
    step_time: float,
) -> None:
    host.monitor.log(report.metrics, step=step)
    host.wait_for_policy_time = 0.0
    host.monitor.log_samples(effective.rollouts, step=step)
    host.monitor.log_distributions(
        distributions={
            "rewards": [rollout.reward for rollout in effective],
            "advantages": [value for rollout in effective if (value := rollout.scalar_advantage()) is not None],
        },
        step=step,
    )
    if host.usage_reporter is not None and (run_id := os.getenv("RUN_ID", "")):
        host.usage_reporter.report_training_usage(
            run_id=run_id,
            step=step,
            tokens=report.num_input + report.num_output,
        )
    if host.heart is not None:
        host.heart.beat()
    host.progress.total_tokens += report.num_tokens
    host.progress.total_samples += report.num_rollouts
    host.progress.total_problems += report.num_unique_examples
    _log_train_batch(host, batch, step=step, step_time=step_time)
    host.train_sink.reset_pre_filter_stats()
    host.maybe_trigger_eval(host.progress.step)
    trim_process_memory()


def _log_train_batch(host: TrainFinalizationHost, batch: TrainBatch, *, step: int, step_time: float) -> None:
    rollouts = batch.rollouts
    effective = rollouts.effective
    metrics = effective.metrics
    n_generated = len(rollouts)
    n_trainable = sum(1 for rollout in rollouts if rollout.is_trainable)
    trainable_rate = (n_trainable / n_generated) if n_generated else 0.0
    max_off_policy = max((rollout.off_policy_steps for rollout in effective), default=0)
    head = (
        f"Step {step} | {format_time(step_time):>7} | Reward {metrics.reward.mean():.4f} | "
        f"Trainable {n_trainable}/{n_generated} ({trainable_rate:.1%}) | "
        f"Turns {metrics.num_turns.mean():.1f} | Branches {metrics.num_branches.mean():.1f} | "
        f"Max Off-Policy {max_off_policy} | Error {rollouts.metrics.has_error.mean():.1%} | "
        f"Truncation {metrics.is_truncated.mean():.1%}"
    )
    if len(host.train_envs) <= 1:
        get_logger().success(head)
        return
    by_env = rollouts.by_env()
    name_width = max((len(name) for name in by_env), default=0)
    lines = [head]
    for env_name in sorted(by_env):
        pool = by_env[env_name]
        env_effective = pool.effective
        env_metrics = env_effective.metrics
        ratio = (len(pool) / n_generated) if n_generated else 0.0
        lines.append(
            f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | "
            f"Reward {env_metrics.reward.mean():.4f} | Turns {env_metrics.num_turns.mean():.1f} | "
            f"Branches {env_metrics.num_branches.mean():.1f} | "
            f"Max Off-Policy {max((r.off_policy_steps for r in env_effective), default=0)} | "
            f"Error {pool.metrics.has_error.mean():.1%} | Truncation {env_metrics.is_truncated.mean():.1%}"
        )
    get_logger().success("\n\t\t ".join(lines))
