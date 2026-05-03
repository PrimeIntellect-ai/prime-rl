import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import verifiers as vf

from prime_rl.configs.orchestrator import (
    AdvantageConfig,
    BatchingConfig,
    OrchestratorConfig,
    SamplesBatching,
    StepBatching,
    TokensBatching,
)
from prime_rl.orchestrator.advantage import AdvantageInputs, setup_advantage_fn
from prime_rl.orchestrator.buffer import DifficultyBuffer
from prime_rl.orchestrator.ckpt import CkptManager, OrchState
from prime_rl.orchestrator.engine import Group
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters, setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.transport.base import TrainingBatchSender
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor


class Done(Exception):
    """Raised by the batcher when max_steps has been reached. Caught by run()."""


class PolicyState(Protocol):
    """Read-only policy state the batcher needs for throttling + logging."""

    policy_version: int

    def max_off_policy_level(self) -> int: ...


class EvalCounter(Protocol):
    """Small surface the batcher needs from the scheduler: how many eval
    groups to expect for a given trigger step, plus the most recent eval
    trigger step for ckpt persistence."""

    last_eval_step: int

    def expected_eval_count(self, step: int) -> int | None: ...


class Advantage:
    """Scores groups in place: computes advantages and attaches them to rollouts.
    Dispatches between DefaultAdvantageConfig (built-in GRPO + optional length
    shaping) and CustomAdvantageConfig (user-imported function) via
    setup_advantage_fn."""

    def __init__(self, cfg: AdvantageConfig):
        self.advantage_fn = setup_advantage_fn(cfg)

    def score(self, group: Group) -> None:
        rewards = torch.tensor([[r.get("reward", 0.0) for r in group.rollouts]], dtype=torch.float32)
        lens = torch.tensor([[get_completion_len(r) for r in group.rollouts]], dtype=torch.int64)
        out = self.advantage_fn(AdvantageInputs(rewards=rewards, completion_lengths=lens))
        for r, a in zip(group.rollouts, out.advantages[0].tolist()):
            r["advantage"] = a


def _split(rollouts: list[vf.RolloutOutput]) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
    trainable: list[vf.RolloutOutput] = []
    filtered: list[vf.RolloutOutput] = []
    for r in rollouts:
        (filtered if r.get("is_filtered") else trainable).append(r)
    return trainable, filtered


class BatchingStrategy(Protocol):
    """Decides when a batch is ready to ship. Implementations maintain their own
    buffer and flush predicate."""

    def add(self, rollouts: list[vf.RolloutOutput]) -> None: ...
    def has_batch(self) -> bool: ...
    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]: ...


class StepStrategy:
    """Ship the first `size` rollouts produced by the engine, pre-filter. The
    trainer receives the trainable subset (filtered ones are counted toward
    `size` but dropped at ship). Matches orch1 semantics."""

    def __init__(self, size: int):
        self.size = size
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return len(self._buf) >= self.size

    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
        cohort, self._buf = self._buf[: self.size], self._buf[self.size :]
        return _split(cohort)


class SamplesStrategy:
    """Ship when `size` trainable rollouts (post-filter) have accumulated.
    Oversamples the engine: filtered rollouts are kept in the buffer for
    metric aggregation but don't count toward `size`."""

    def __init__(self, size: int):
        self.size = size
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return sum(1 for r in self._buf if not r.get("is_filtered")) >= self.size

    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
        trainable: list[vf.RolloutOutput] = []
        filtered: list[vf.RolloutOutput] = []
        cut = 0
        for i, r in enumerate(self._buf):
            if r.get("is_filtered"):
                filtered.append(r)
            else:
                trainable.append(r)
                if len(trainable) == self.size:
                    cut = i + 1
                    break
        self._buf = self._buf[cut:]
        return trainable, filtered


class TokensStrategy:
    """Ship when trainable completion tokens (post-filter) reach `size`."""

    def __init__(self, size: int):
        self.size = size
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return sum(get_completion_len(r) for r in self._buf if not r.get("is_filtered")) >= self.size

    def pop(self) -> tuple[list[vf.RolloutOutput], list[vf.RolloutOutput]]:
        trainable: list[vf.RolloutOutput] = []
        filtered: list[vf.RolloutOutput] = []
        tokens = 0
        cut = 0
        for i, r in enumerate(self._buf):
            if r.get("is_filtered"):
                filtered.append(r)
            else:
                trainable.append(r)
                tokens += get_completion_len(r)
                if tokens >= self.size:
                    cut = i + 1
                    break
        self._buf = self._buf[cut:]
        return trainable, filtered


def build_strategy(cfg: BatchingConfig) -> BatchingStrategy:
    if isinstance(cfg, StepBatching):
        return StepStrategy(cfg.size)
    if isinstance(cfg, SamplesBatching):
        return SamplesStrategy(cfg.size)
    if isinstance(cfg, TokensBatching):
        return TokensStrategy(cfg.size)
    raise ValueError(f"Unknown batching config: {cfg!r}")


def _key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


_TIMING_FIELDS = ("total", "setup", "generation", "model", "env", "scoring", "overhead")


def _rollout_timing(r: vf.RolloutOutput) -> dict[str, float]:
    """Pull a flat {field: seconds} dict out of one rollout's TimeSpan-shaped
    timing block. `total`/`overhead` are scalars; the rest carry a `duration`
    derived from start/end timestamps."""
    t = r.get("timing")
    if not t:
        return {}
    out: dict[str, float] = {}
    for k in _TIMING_FIELDS:
        v = t.get(k)
        if isinstance(v, dict):
            out[k] = float(v.get("duration", 0.0))
        elif v is not None:
            out[k] = float(v)
    return out


def _rollout_metrics(prefix: str, rollouts: list[vf.RolloutOutput]) -> dict:
    """Flat per-rollout stats, keyed under `prefix` (empty for top-level).
    Shared by train (trainable + cohort views) and eval (per-env + overall
    views). Filter rates + drop rate only emitted when filter annotations are
    present. Timing keys come out as `timing/{field}/mean`."""
    if not rollouts:
        return {}
    rewards = [r.get("reward", 0.0) for r in rollouts]
    lens = [get_completion_len(r) for r in rollouts]
    m: dict = {
        _key(prefix, "reward/mean"): sum(rewards) / len(rewards),
        _key(prefix, "seq_len/mean"): sum(lens) / len(lens),
        _key(prefix, "pass_rate"): sum(1 for r in rewards if r > 0) / len(rewards),
        _key(prefix, "n_rollouts"): len(rollouts),
    }
    if "is_filtered" in rollouts[0]:
        n_filt = sum(1 for r in rollouts if r.get("is_filtered"))
        m[_key(prefix, "filters/drop_rate")] = n_filt / len(rollouts)
    if "filters" in rollouts[0]:
        for name in rollouts[0]["filters"]:
            hits = sum(1 for r in rollouts if r["filters"].get(name))
            m[_key(prefix, f"filters/{name}/rate")] = hits / len(rollouts)
    timings = [_rollout_timing(r) for r in rollouts]
    timings = [t for t in timings if t]
    if timings:
        for field in _TIMING_FIELDS:
            vals = [t[field] for t in timings if field in t]
            if vals:
                m[_key(prefix, f"timing/{field}/mean")] = sum(vals) / len(vals)
    return m


def _eval_metrics(prefix: str, groups: list[Group]) -> dict:
    """Eval-only per-example stats: pass_at_k (any rollout passed), avg_at_k
    (mean of per-example mean rewards), n_examples."""
    non_empty = [g for g in groups if g.rollouts]
    if not non_empty:
        return {}
    passed = sum(1 for g in non_empty if any(r.get("reward", 0.0) > 0 for r in g.rollouts))
    per_ex_means = [sum(r.get("reward", 0.0) for r in g.rollouts) / len(g.rollouts) for g in non_empty]
    return {
        _key(prefix, "pass_at_k"): passed / len(non_empty),
        _key(prefix, "avg_at_k"): sum(per_ex_means) / len(per_ex_means),
        _key(prefix, "n_examples"): len(non_empty),
    }


class PostProcessor:
    """Converts rollouts -> TrainingSamples, sends the batch, and emits per-step logs/metrics."""

    def __init__(self, tokenizer, sender: TrainingBatchSender, policy: PolicyState):
        self.tokenizer = tokenizer
        self.sender = sender
        self.policy = policy
        self.logger = get_logger()
        self._last_step_t = time.perf_counter()

    async def process(
        self,
        trainable: list[vf.RolloutOutput],
        filtered: list[vf.RolloutOutput],
        step: int,
    ) -> None:
        t0 = time.perf_counter()
        samples = await asyncio.to_thread(self._convert, trainable)
        convert_time = time.perf_counter() - t0
        if not samples:
            # Trainer needs at least one sample per step. Step-mode can produce
            # fully-filtered cohorts on small models; surface clearly.
            self.logger.warning(
                f"Step {step}: shipping empty batch ({len(filtered)} filtered, 0 trainable). "
                f"Trainer may fail on this step — consider relaxing filters or batch_size."
            )

        t1 = time.perf_counter()
        batch = TrainingBatch(examples=samples, step=step)
        await asyncio.to_thread(self.sender.send, batch)
        send_time = time.perf_counter() - t1

        now = time.perf_counter()
        step_time = now - self._last_step_t
        self._last_step_t = now

        self._log(trainable, filtered, samples, step, step_time, convert_time, send_time)

    def _convert(self, rollouts: list[vf.RolloutOutput]) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        for r in rollouts:
            pretokenize_rollout_trajectory(r, self.tokenizer)
            out = interleave_rollout(r)
            if out is None:
                continue
            for s in out:
                s.advantage = r.get("advantage")
                s.reward = r.get("reward")
            samples.extend(out)
        return samples

    def _log(
        self,
        trainable: list[vf.RolloutOutput],
        filtered: list[vf.RolloutOutput],
        samples: list[TrainingSample],
        step: int,
        step_time: float,
        convert_time: float,
        send_time: float,
    ) -> None:
        cohort = trainable + filtered
        advs_t = [r.get("advantage") or 0.0 for r in trainable]
        adv_abs = sum(abs(a) for a in advs_t) / len(trainable) if trainable else 0.0
        async_level = step - self.policy.policy_version
        max_off_policy_level = self.policy.max_off_policy_level()

        metrics: dict = {
            # Trainable subset (post-filter) — top level, no kind prefix.
            **_rollout_metrics("", trainable),
            # Full pre-filter cohort — grouped under rollouts/.
            **_rollout_metrics("rollouts", cohort),
            "advantage/abs_mean": adv_abs,
            "batch_size": len(samples),
            "policy_version": self.policy.policy_version,
            "scheduler/async_level": async_level,
            "scheduler/max_off_policy_level": max_off_policy_level,
            "time/step": step_time,
            "time/convert": convert_time,
            "time/ship": send_time,
        }
        get_monitor().log(metrics, step=step)

        # reward/mean and seq_len/mean are only present when trainable is
        # non-empty (see _rollout_metrics short-circuit). Step-mode batches can
        # come out fully filtered — log placeholders instead of crashing.
        reward_str = f"{metrics['reward/mean']:.4f}" if "reward/mean" in metrics else "n/a"
        seqlen_str = f"{metrics['seq_len/mean']:.1f} tokens/sample" if "seq_len/mean" in metrics else "n/a"
        self.logger.success(
            f"Step {step} | "
            f"Time: {step_time:.2f}s | "
            f"Reward: {reward_str} | "
            f"Seq. Length: {seqlen_str} | "
            f"Async Level: {async_level} | "
            f"Max. Off-Policy Level: {max_off_policy_level} | "
            f"Filtered: {len(filtered)}/{len(cohort)}"
        )


@dataclass
class BatcherInputs:
    """Pre-built inputs for the TrainBatcher. `setup_batcher` produces this
    from config; tests construct it directly with stub policy/eval/post."""

    in_q: asyncio.Queue[Group]
    post: "PostProcessor"
    policy: PolicyState
    strategy: BatchingStrategy
    advantage_cfg: AdvantageConfig
    filters: list[RolloutFilter] = field(default_factory=list)
    max_steps: int | None = None
    max_training_batches_ahead: int = 1
    strict_async_level: bool = False
    eval_counter: EvalCounter | None = None
    ckpt_manager: CkptManager | None = None
    ckpt_interval: int | None = None
    buffer: DifficultyBuffer | None = None
    heartbeat: Heartbeat | None = None
    inference_metrics: InferenceMetricsCollector | None = None


class TrainBatcher:
    """Wires the stages: score (Advantage) → annotate (filters) → accumulate
    (BatchingStrategy) → post-process (PostProcessor). Also routes eval groups
    into an eval aggregator keyed by the eval trigger step."""

    def __init__(self, inputs: BatcherInputs):
        self.in_q = inputs.in_q
        self.policy = inputs.policy
        self.strategy = inputs.strategy
        self.advantage = Advantage(inputs.advantage_cfg)
        self.filters = inputs.filters
        self.post = inputs.post
        self.max_steps = inputs.max_steps
        self.max_training_batches_ahead = inputs.max_training_batches_ahead
        self.strict = inputs.strict_async_level
        self.step = 0
        self.eval_counter = inputs.eval_counter
        self._eval_buf: dict[int, list[Group]] = defaultdict(list)
        self.ckpt_manager = inputs.ckpt_manager
        self.ckpt_interval = inputs.ckpt_interval
        self.buffer = inputs.buffer
        self.heartbeat = inputs.heartbeat
        self.inference_metrics = inputs.inference_metrics
        self.logger = get_logger()

    async def _wait_barrier(self) -> None:
        # Don't ship more than max_training_batches_ahead of the latest policy
        # version. Stalling here cascades backpressure: the groups queue fills,
        # the engine's semaphore stops releasing. Set to a huge value to
        # benchmark orch alone (no trainer, no blocking).
        # Strict mode: wait until lead EQUALS the target (not just <=).
        # If we block here for an unusually long time the trainer is likely
        # gone or stuck. We re-warn every 60s so the stall stays visible in
        # logs; the launcher's process supervision is what hard-fails on
        # actual trainer subprocess exit.
        t0 = time.perf_counter()
        next_warn = 30.0
        while True:
            lead = self.step - self.policy.policy_version
            if self.strict:
                if lead == self.max_training_batches_ahead:
                    return
            elif lead <= self.max_training_batches_ahead:
                return
            elapsed = time.perf_counter() - t0
            if elapsed >= next_warn:
                self.logger.warning(
                    f"Batcher stalled at barrier for {int(elapsed)}s: step={self.step}, "
                    f"policy_version={self.policy.policy_version}, lead={lead} "
                    f"(max_async_level={self.max_training_batches_ahead}). "
                    f"Trainer may be stuck or down."
                )
                next_warn = elapsed + 60.0
            await asyncio.sleep(0.1)

    def _handle_eval(self, group: Group) -> None:
        if group.eval_step is None or self.eval_counter is None:
            return
        # Annotate (never drop) so we can report per-filter rates on eval. The
        # zero-advantage filter is a no-op here since eval has no advantages.
        if self.filters and group.rollouts:
            apply_filters(self.filters, group.rollouts)
        buf = self._eval_buf[group.eval_step]
        buf.append(group)
        expected = self.eval_counter.expected_eval_count(group.eval_step)
        if expected is None or len(buf) < expected:
            return
        self._flush_eval(group.eval_step, self._eval_buf.pop(group.eval_step))

    def _flush_eval(self, step: int, groups: list[Group]) -> None:
        per_env: dict[str, list[Group]] = defaultdict(list)
        for g in groups:
            per_env[g.env_id].append(g)
        metrics: dict = {}
        for env_id, env_groups in per_env.items():
            flat = [r for g in env_groups for r in g.rollouts]
            metrics.update(_rollout_metrics(f"eval/{env_id}", flat))
            metrics.update(_eval_metrics(f"eval/{env_id}", env_groups))
        all_rollouts = [r for g in groups for r in g.rollouts]
        metrics.update(_rollout_metrics("eval", all_rollouts))
        metrics.update(_eval_metrics("eval", groups))
        if "eval/reward/mean" not in metrics:
            self.logger.warning(f"Eval @ step {step}: all {len(groups)} groups timed out, skipping log")
            return
        n_timed_out = sum(1 for g in groups if not g.rollouts)
        if n_timed_out:
            metrics["eval/timed_out"] = n_timed_out
        get_monitor().log(metrics, step=step)
        envs_str = ", ".join(
            f"{e}=r:{metrics[f'eval/{e}/reward/mean']:.3f}/p@k:{metrics[f'eval/{e}/pass_at_k']:.3f}"
            for e in per_env
            if f"eval/{e}/reward/mean" in metrics
        )
        suffix = f" | Timed out: {n_timed_out}" if n_timed_out else ""
        self.logger.success(
            f"Eval @ step {step} | Reward: {metrics['eval/reward/mean']:.4f} | "
            f"Pass@k: {metrics['eval/pass_at_k']:.3f} | "
            f"N: {metrics['eval/n_rollouts']} rollouts / {metrics['eval/n_examples']} examples | "
            f"Envs: {envs_str}{suffix}"
        )

    async def _maybe_save_ckpt(self) -> None:
        if not self.ckpt_manager or not self.ckpt_interval:
            return
        if self.step <= 0 or self.step % self.ckpt_interval != 0:
            return
        if self.max_steps is not None and self.step >= self.max_steps:
            return  # don't bother saving on the final step — exit follows immediately
        last_eval = self.eval_counter.last_eval_step if self.eval_counter is not None else 0
        buffer_state = self.buffer.state_dict() if self.buffer is not None else {}
        state = OrchState(step=self.step, last_eval_step=last_eval, buffer_state=buffer_state)
        await asyncio.to_thread(self.ckpt_manager.save, state, self.step)

    async def run(self) -> None:
        while True:
            group = await self.in_q.get()
            if group.kind == "eval":
                self._handle_eval(group)
                continue
            if self.buffer is not None:
                self.buffer.observe(group)
            self.advantage.score(group)
            apply_filters(self.filters, group.rollouts)
            self.strategy.add(group.rollouts)
            while self.strategy.has_batch():
                await self._wait_barrier()
                trainable, filtered = self.strategy.pop()
                await self.post.process(trainable, filtered, self.step)
                if self.buffer is not None:
                    get_monitor().log(self.buffer.metrics(), step=self.step)
                if self.inference_metrics is not None:
                    inf_metrics = await self.inference_metrics.collect()
                    if inf_metrics:
                        get_monitor().log(inf_metrics, step=self.step)
                if self.heartbeat is not None:
                    self.heartbeat.beat()
                self.step += 1
                await self._maybe_save_ckpt()
                if self.max_steps is not None and self.step >= self.max_steps:
                    raise Done()


def setup_batcher(
    cfg: OrchestratorConfig,
    *,
    in_q: asyncio.Queue[Group],
    tokenizer: Any,
    policy: PolicyState,
    eval_counter: EvalCounter | None = None,
    ckpt_manager: CkptManager | None = None,
    buffer: DifficultyBuffer | None = None,
    inference_metrics: InferenceMetricsCollector | None = None,
) -> TrainBatcher:
    """Translate config → TrainBatcher. Builds the batch sender, filters,
    strategy, heartbeat, and PostProcessor internally. Tests should construct
    `TrainBatcher(BatcherInputs(...))` directly."""
    sender = setup_training_batch_sender(cfg.output_dir, cfg.rollout_transport)
    filters = setup_filters(cfg.filters, vocab_size=tokenizer.vocab_size)
    strategy = build_strategy(cfg.batch_size)
    heartbeat = Heartbeat(cfg.heartbeat.url) if cfg.heartbeat else None
    post = PostProcessor(tokenizer, sender, policy)
    return TrainBatcher(
        BatcherInputs(
            in_q=in_q,
            post=post,
            policy=policy,
            strategy=strategy,
            advantage_cfg=cfg.advantage,
            filters=filters,
            max_steps=cfg.max_steps,
            max_training_batches_ahead=cfg.max_async_level,
            strict_async_level=cfg.strict_async_level,
            eval_counter=eval_counter,
            ckpt_manager=ckpt_manager,
            ckpt_interval=cfg.ckpt.interval if cfg.ckpt else None,
            buffer=buffer,
            heartbeat=heartbeat,
            inference_metrics=inference_metrics,
        )
    )
