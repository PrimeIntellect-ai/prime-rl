"""Group: produces and scores a batch of rollouts for one scheduler task.

Owns the full "rollout → advantage" pipeline. The orchestrator's default is
GRPO with optional length shaping (`GRPOGroup`). This Protocol is the seam
for swapping in alternative schemes (multi-step agents, self-play, PPO with
a critic, …) without touching the engine.

Engine calls `run()` once per dispatched task; the resulting list of rollouts
is wrapped in a `GroupOutput` and queued for the batcher.
"""

import asyncio
from typing import Protocol

import torch
import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import AdvantageConfig, OrchestratorConfig
from prime_rl.orchestrator.advantage import AdvantageInputs, setup_advantage_fn
from prime_rl.orchestrator.scheduler import Task
from prime_rl.orchestrator.vf_utils import get_completion_len


class Group(Protocol):
    """Produces a group of scored rollouts for one scheduler task.

    Implementations control how rollouts are sampled (parallel, sequential,
    multi-turn, …) and how rewards are turned into advantages. For eval tasks
    advantages are not computed — eval reports raw rewards.
    """

    model: str
    """Mutable: rollouts target this model name. The engine flips it on LoRA swap."""

    async def run(self, task: Task, example: dict) -> list[vf.RolloutOutput]: ...


class GRPOGroup:
    """Baked-in default: parallel rollouts + group-relative GRPO advantage.

    Train tasks: a per-group GRPO baseline is subtracted from each reward
    (with optional length shaping). Eval tasks: rolled out but not scored.
    """

    def __init__(
        self,
        *,
        client: vf.ClientConfig,
        model: str,
        advantage_cfg: AdvantageConfig,
        rate_limiter: AsyncLimiter | None = None,
    ):
        self.client = client
        self.model = model
        self.rate_limiter = rate_limiter
        self._advantage_fn = setup_advantage_fn(advantage_cfg)

    async def run(self, task: Task, example: dict) -> list[vf.RolloutOutput]:
        rollouts = list(await asyncio.gather(*(self._rollout(task, example) for _ in range(task.rollouts_per_group))))
        if task.kind == "train":
            self._score(rollouts)
        return rollouts

    async def _rollout(self, task: Task, example: dict) -> vf.RolloutOutput:
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        return await task.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.model,
            sampling_args=task.sampling_args,
            state_columns=["trajectory", "sampling_args"],
        )

    def _score(self, rollouts: list[vf.RolloutOutput]) -> None:
        rewards = torch.tensor([[r.get("reward", 0.0) for r in rollouts]], dtype=torch.float32)
        lens = torch.tensor([[get_completion_len(r) for r in rollouts]], dtype=torch.int64)
        out = self._advantage_fn(AdvantageInputs(rewards=rewards, completion_lengths=lens))
        for r, a in zip(rollouts, out.advantages[0].tolist()):
            r["advantage"] = a


def setup_group(cfg: OrchestratorConfig, *, client: vf.ClientConfig) -> Group:
    """Build the orchestrator's default group implementation from config."""
    rate_limiter = AsyncLimiter(max_rate=cfg.tasks_per_minute, time_period=60) if cfg.tasks_per_minute else None
    return GRPOGroup(
        client=client,
        model=cfg.model.name,
        advantage_cfg=cfg.advantage,
        rate_limiter=rate_limiter,
    )
