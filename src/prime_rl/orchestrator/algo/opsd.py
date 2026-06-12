from __future__ import annotations

import asyncio
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, OPSDAdvantageConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.utils import compute_prefill_logprobs

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import TrainRollout
    from prime_rl.utils.client import InferencePool


class OPSDAlgorithm(Algorithm):
    """On-policy self-distillation (SDFT). The teacher defaults to the policy
    itself, conditioned on an expert demonstration — no extra deployment.

    The scoring prefix is rebuilt from the rollout's first-turn prompt
    messages with the demonstration woven into the last user message; the
    returned completion logprobs are aligned back onto the sample (the
    sample's prompt positions are 0.0 and stay outside the loss mask). No
    scalar advantage is assigned."""

    action_loss_type = "ref_kl"
    model_role = "teacher"

    def __init__(self, config: AlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        assert isinstance(config.advantage, OPSDAdvantageConfig)
        assert renderer is not None, "opsd requires the renderer (validated at config time)"
        self.demo_key = config.advantage.demo_key
        self.template = config.advantage.template
        self.max_concurrent = config.advantage.max_concurrent

    def _ref_prefix_ids(self, rollout: TrainRollout) -> list[int]:
        trajectory = rollout.raw.get("trajectory") or []
        if len(trajectory) != 1:
            raise ValueError(
                f"opsd supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {len(trajectory)} steps."
            )
        info = rollout.raw.get("info") or {}
        demonstration = info.get(self.demo_key) if isinstance(info, dict) else None
        if demonstration is None:
            demonstration = rollout.raw.get(self.demo_key)
        if demonstration is None:
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the example's info dict or as a "
                f"top-level rollout field (env '{rollout.env_name}', example {rollout.example_id})."
            )

        messages = [dict(m) for m in trajectory[0]["prompt"]]
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"opsd found no user message to condition (env '{rollout.env_name}').")
        last_user = messages[user_indices[-1]]
        question = last_user.get("content")
        if not isinstance(question, str):
            raise ValueError("opsd supports text-only prompts (user content must be a string).")
        last_user["content"] = self.template.format(question=question, demonstration=demonstration)

        # Render through the policy's renderer — the same messages → token ids
        # path the policy's own prompts take, so the scoring prefix matches
        # the prompt distribution the teacher conditions on.
        assert self.renderer is not None
        return self.renderer.render_ids(messages, add_generation_prompt=True)

    async def score(self, rollouts: list[TrainRollout]) -> None:
        pool = self._reference_pool()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def score_rollout(client, rollout: TrainRollout) -> None:
            prefix_ids = self._ref_prefix_ids(rollout)
            assert len(rollout.samples) == 1  # single-step trajectory → one sample
            sample = rollout.samples[0]
            async with semaphore:
                full_logprobs = await compute_prefill_logprobs(
                    client, pool.model_name, prefix_ids + list(sample.completion_ids)
                )
            completion_logprobs = full_logprobs[-len(sample.completion_ids) :]
            sample.ref_logprobs = [0.0] * len(sample.prompt_ids) + completion_logprobs

        await asyncio.gather(
            *[score_rollout(client, rollout) for client, rollout in zip(cycle(pool.train_clients), rollouts)]
        )
