import asyncio
import time

import torch
import verifiers as vf

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
from prime_rl.orch2.engine import Group, RolloutEngine
from prime_rl.orchestrator.advantage import AdvantageInputs, default_advantage_fn
from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.transport import TrainingBatch, TrainingSample
from prime_rl.transport.base import TrainingBatchSender
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor


class Done(Exception):
    """Raised by the batcher when max_steps has been reached. Caught by run()."""


class TrainBatcher:
    def __init__(
        self,
        in_q: asyncio.Queue[Group],
        tokenizer,
        sender: TrainingBatchSender,
        engine: RolloutEngine,
        batch_size: int,
        advantage_cfg: DefaultAdvantageConfig,
        max_steps: int | None = None,
        max_async_level: int = 1,
    ):
        self.in_q = in_q
        self.tokenizer = tokenizer
        self.sender = sender
        self.engine = engine
        self.batch_size = batch_size
        self.advantage_cfg = advantage_cfg
        self.max_steps = max_steps
        self.max_async_level = max_async_level
        self.step = 0
        self.logger = get_logger()
        self._last_step_t = time.perf_counter()

    async def run(self) -> None:
        buf: list[vf.RolloutOutput] = []
        while True:
            group = await self.in_q.get()
            if group.pool_id.startswith("eval"):
                continue
            self._score(group)
            buf.extend(group.rollouts)
            while len(buf) >= self.batch_size:
                rollouts, buf = buf[: self.batch_size], buf[self.batch_size :]
                # Async-level barrier: don't ship more than max_async_level batches
                # ahead of the latest policy version. This throttles orch2 when
                # weight updates fall behind, and cascades backpressure through
                # the groups queue to the engine.
                while self.step - self.engine.policy_version > self.max_async_level:
                    await asyncio.sleep(0.1)
                await self._ship(rollouts)
                if self.max_steps is not None and self.step >= self.max_steps:
                    raise Done()

    def _score(self, group: Group) -> None:
        rewards = torch.tensor([[r.get("reward", 0.0) for r in group.rollouts]], dtype=torch.float32)
        lens = torch.tensor([[get_completion_len(r) for r in group.rollouts]], dtype=torch.int64)
        out = default_advantage_fn(
            AdvantageInputs(rewards=rewards, completion_lengths=lens),
            length_shaping=self.advantage_cfg.length_shaping,
        )
        for r, a in zip(group.rollouts, out.advantages[0].tolist()):
            r["advantage"] = a

    async def _ship(self, rollouts: list[vf.RolloutOutput]) -> None:
        t0 = time.perf_counter()
        samples = await asyncio.to_thread(self._convert, rollouts)
        convert_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        batch = TrainingBatch(examples=samples, step=self.step)
        await asyncio.to_thread(self.sender.send, batch)
        send_time = time.perf_counter() - t1

        now = time.perf_counter()
        step_time = now - self._last_step_t
        self._last_step_t = now

        rewards = [r.get("reward", 0.0) for r in rollouts]
        advs = [r.get("advantage") or 0.0 for r in rollouts]
        seq_lens = [get_completion_len(r) for r in rollouts]
        reward_mean = sum(rewards) / len(rewards)
        adv_abs = sum(abs(a) for a in advs) / len(advs)
        seq_mean = sum(seq_lens) / len(seq_lens)
        async_level = self.step - self.engine.policy_version
        max_off_policy_level = self.engine.max_off_policy_level()

        self.logger.success(
            f"Step {self.step} | "
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
                "train/policy_version": self.engine.policy_version,
                "scheduler/async_level": async_level,
                "scheduler/max_off_policy_level": max_off_policy_level,
                "time/step": step_time,
                "time/convert": convert_time,
                "time/ship": send_time,
            },
            step=self.step,
        )
        self.step += 1

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
