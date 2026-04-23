import asyncio
import time
from typing import Protocol

import torch
import verifiers as vf

from prime_rl.configs.orchestrator import (
    BatchingConfig,
    DefaultAdvantageConfig,
    SamplesBatching,
    StepBatching,
    TokensBatching,
)
from prime_rl.orch2.engine import Group
from prime_rl.orchestrator.advantage import AdvantageInputs, default_advantage_fn
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.transport import TrainingBatch, TrainingSample
from prime_rl.transport.base import TrainingBatchSender
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor


class Done(Exception):
    """Raised by the batcher when max_steps has been reached. Caught by run()."""


class PolicyState(Protocol):
    """Read-only policy state the batcher needs for throttling + logging."""

    policy_version: int

    def max_off_policy_level(self) -> int: ...


class Advantage:
    """Scores groups in place: computes GRPO advantages and attaches them to rollouts."""

    def __init__(self, cfg: DefaultAdvantageConfig):
        self.cfg = cfg

    def score(self, group: Group) -> None:
        rewards = torch.tensor([[r.get("reward", 0.0) for r in group.rollouts]], dtype=torch.float32)
        lens = torch.tensor([[get_completion_len(r) for r in group.rollouts]], dtype=torch.int64)
        out = default_advantage_fn(
            AdvantageInputs(rewards=rewards, completion_lengths=lens),
            length_shaping=self.cfg.length_shaping,
        )
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
        n_cohort = len(cohort)
        n_filtered = len(filtered)

        rewards_t = [r.get("reward", 0.0) for r in trainable]
        advs_t = [r.get("advantage") or 0.0 for r in trainable]
        seq_lens_t = [get_completion_len(r) for r in trainable]
        reward_mean = sum(rewards_t) / len(trainable)
        adv_abs = sum(abs(a) for a in advs_t) / len(trainable)
        seq_mean = sum(seq_lens_t) / len(trainable)

        rewards_all = [r.get("reward", 0.0) for r in cohort]
        seq_lens_all = [get_completion_len(r) for r in cohort]
        reward_mean_all = sum(rewards_all) / n_cohort
        seq_mean_all = sum(seq_lens_all) / n_cohort

        async_level = step - self.policy.policy_version
        max_off_policy_level = self.policy.max_off_policy_level()

        self.logger.success(
            f"Step {step} | "
            f"Time: {step_time:.2f}s | "
            f"Reward: {reward_mean:.4f} | "
            f"Seq. Length: {seq_mean:.1f} tokens/sample | "
            f"Async Level: {async_level} | "
            f"Max. Off-Policy Level: {max_off_policy_level} | "
            f"Filtered: {n_filtered}/{n_cohort}"
        )

        metrics: dict = {
            # What the trainer actually sees (post-filter)
            "train/reward/mean": reward_mean,
            "train/advantage/abs_mean": adv_abs,
            "train/seq_len/mean": seq_mean,
            "train/batch_size": len(samples),
            "train/policy_version": self.policy.policy_version,
            # The full cohort the engine produced for this batch (pre-filter)
            "rollouts/reward/mean": reward_mean_all,
            "rollouts/seq_len/mean": seq_mean_all,
            "rollouts/cohort_size": n_cohort,
            # Filter drop rate + per-filter detection rate over the cohort
            "filters/drop_rate": n_filtered / n_cohort,
            "scheduler/async_level": async_level,
            "scheduler/max_off_policy_level": max_off_policy_level,
            "time/step": step_time,
            "time/convert": convert_time,
            "time/ship": send_time,
        }
        # Per-filter detection rate: fraction of the cohort each filter flagged
        # (filters is the same dict on every annotated rollout; monitor-only
        # hits also count here even when they didn't cause a drop).
        if cohort and "filters" in cohort[0]:
            for name in cohort[0]["filters"]:
                hits = sum(1 for r in cohort if r["filters"].get(name))
                metrics[f"filters/{name}/rate"] = hits / n_cohort

        get_monitor().log(metrics, step=step)


class TrainBatcher:
    """Wires the stages: score (Advantage) → annotate (filters) → accumulate
    (BatchingStrategy) → post-process (PostProcessor). Also acts as the
    VersionObserver hook so step-mode strategies wake on weight updates."""

    def __init__(
        self,
        in_q: asyncio.Queue[Group],
        tokenizer,
        sender: TrainingBatchSender,
        policy: PolicyState,
        strategy: BatchingStrategy,
        advantage_cfg: DefaultAdvantageConfig,
        filters: list[RolloutFilter] | None = None,
        max_steps: int | None = None,
        max_training_batches_ahead: int = 1,
        strict_async_level: bool = False,
    ):
        self.in_q = in_q
        self.policy = policy
        self.strategy = strategy
        self.advantage = Advantage(advantage_cfg)
        self.filters = filters or []
        self.post = PostProcessor(tokenizer, sender, policy)
        self.max_steps = max_steps
        self.max_training_batches_ahead = max_training_batches_ahead
        self.strict = strict_async_level
        self.step = 0

    async def _wait_barrier(self) -> None:
        # Don't ship more than max_training_batches_ahead of the latest policy
        # version. Stalling here cascades backpressure: the groups queue fills,
        # the engine's semaphore stops releasing. Set to a huge value to
        # benchmark orch alone (no trainer, no blocking).
        # Strict mode: wait until lead EQUALS the target (not just <=).
        while True:
            lead = self.step - self.policy.policy_version
            if self.strict:
                if lead == self.max_training_batches_ahead:
                    return
            elif lead <= self.max_training_batches_ahead:
                return
            await asyncio.sleep(0.1)

    async def run(self) -> None:
        while True:
            group = await self.in_q.get()
            if group.kind == "eval":
                continue
            self.advantage.score(group)
            apply_filters(self.filters, group.rollouts)
            self.strategy.add(group.rollouts)
            while self.strategy.has_batch():
                await self._wait_barrier()
                trainable, filtered = self.strategy.pop()
                await self.post.process(trainable, filtered, self.step)
                self.step += 1
                if self.max_steps is not None and self.step >= self.max_steps:
                    raise Done()
