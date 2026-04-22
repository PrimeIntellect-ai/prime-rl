import asyncio
import time
from typing import Protocol

import torch
import verifiers as vf

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
from prime_rl.orch2.engine import Group
from prime_rl.orchestrator.advantage import AdvantageInputs, default_advantage_fn
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


class BatchBuilder:
    """Accumulates rollouts across groups; yields batches of batch_size with async-level throttling."""

    def __init__(
        self,
        batch_size: int,
        max_training_batches_ahead: int,
        policy: PolicyState,
        strict: bool = False,
    ):
        self.batch_size = batch_size
        self.max_training_batches_ahead = max_training_batches_ahead
        self.policy = policy
        self.strict = strict
        self._buf: list[vf.RolloutOutput] = []

    def add(self, rollouts: list[vf.RolloutOutput]) -> None:
        self._buf.extend(rollouts)

    def has_batch(self) -> bool:
        return len(self._buf) >= self.batch_size

    def pop(self) -> list[vf.RolloutOutput]:
        batch, self._buf = self._buf[: self.batch_size], self._buf[self.batch_size :]
        return batch

    async def wait_barrier(self, step: int) -> None:
        # Don't ship more than max_training_batches_ahead of the latest policy
        # version. Stalling here cascades backpressure: the groups queue fills,
        # the engine's semaphore stops releasing. Set to a huge value to
        # benchmark orch alone (no trainer, no blocking).
        # Strict mode: wait until lead EQUALS the target (not just <=), so the
        # policy-version-to-batch-step distance is pinned exactly.
        while True:
            lead = step - self.policy.policy_version
            if self.strict:
                if lead == self.max_training_batches_ahead:
                    return
            elif lead <= self.max_training_batches_ahead:
                return
            await asyncio.sleep(0.1)


class PostProcessor:
    """Converts rollouts -> TrainingSamples, sends the batch, and emits per-step logs/metrics."""

    def __init__(self, tokenizer, sender: TrainingBatchSender, policy: PolicyState):
        self.tokenizer = tokenizer
        self.sender = sender
        self.policy = policy
        self.logger = get_logger()
        self._last_step_t = time.perf_counter()

    async def process(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        t0 = time.perf_counter()
        samples = await asyncio.to_thread(self._convert, rollouts)
        convert_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        batch = TrainingBatch(examples=samples, step=step)
        await asyncio.to_thread(self.sender.send, batch)
        send_time = time.perf_counter() - t1

        now = time.perf_counter()
        step_time = now - self._last_step_t
        self._last_step_t = now

        self._log(rollouts, samples, step, step_time, convert_time, send_time)

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
        rollouts: list[vf.RolloutOutput],
        samples: list[TrainingSample],
        step: int,
        step_time: float,
        convert_time: float,
        send_time: float,
    ) -> None:
        rewards = [r.get("reward", 0.0) for r in rollouts]
        advs = [r.get("advantage") or 0.0 for r in rollouts]
        seq_lens = [get_completion_len(r) for r in rollouts]
        reward_mean = sum(rewards) / len(rewards)
        adv_abs = sum(abs(a) for a in advs) / len(advs)
        seq_mean = sum(seq_lens) / len(seq_lens)
        async_level = step - self.policy.policy_version
        max_off_policy_level = self.policy.max_off_policy_level()

        self.logger.success(
            f"Step {step} | "
            f"Time: {step_time:.2f}s | "
            f"Reward: {reward_mean:.4f} | "
            f"Seq. Length: {seq_mean:.1f} tokens/sample | "
            f"Async Level: {async_level} | "
            f"Max. Off-Policy Level: {max_off_policy_level}"
        )
        get_monitor().log(
            {
                "train/reward/mean": reward_mean,
                "train/advantage/abs_mean": adv_abs,
                "train/seq_len/mean": seq_mean,
                "train/batch_size": len(samples),
                "train/policy_version": self.policy.policy_version,
                "scheduler/async_level": async_level,
                "scheduler/max_off_policy_level": max_off_policy_level,
                "time/step": step_time,
                "time/convert": convert_time,
                "time/ship": send_time,
            },
            step=step,
        )


class TrainBatcher:
    """Wires the three stages: score (advantage), buffer (BatchBuilder), post-process (PostProcessor)."""

    def __init__(
        self,
        in_q: asyncio.Queue[Group],
        tokenizer,
        sender: TrainingBatchSender,
        policy: PolicyState,
        batch_size: int,
        advantage_cfg: DefaultAdvantageConfig,
        max_steps: int | None = None,
        max_training_batches_ahead: int = 1,
        strict_async_level: bool = False,
    ):
        self.in_q = in_q
        self.advantage = Advantage(advantage_cfg)
        self.builder = BatchBuilder(batch_size, max_training_batches_ahead, policy, strict=strict_async_level)
        self.post = PostProcessor(tokenizer, sender, policy)
        self.max_steps = max_steps
        self.step = 0

    async def run(self) -> None:
        while True:
            group = await self.in_q.get()
            if group.kind == "eval":
                continue
            self.advantage.score(group)
            self.builder.add(group.rollouts)
            while self.builder.has_batch():
                await self.builder.wait_barrier(self.step)
                batch = self.builder.pop()
                await self.post.process(batch, self.step)
                self.step += 1
                if self.max_steps is not None and self.step >= self.max_steps:
                    raise Done()
