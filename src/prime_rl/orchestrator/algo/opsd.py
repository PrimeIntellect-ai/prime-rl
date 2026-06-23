from __future__ import annotations

import asyncio
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, OPSDAlgorithmConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.utils import compute_prefill_logprobs

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
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
        assert isinstance(config, OPSDAlgorithmConfig)
        assert renderer is not None, "opsd requires the renderer (validated at config time)"
        self.demo_key = config.demo_key
        self.template = config.template
        self.max_concurrent = config.max_concurrent
        self.teacher = config.model
        self.teacher_pool: InferencePool | None = None  # connected in setup()

    async def setup(self) -> None:
        self.teacher_pool = await self.connect(self.teacher)

    def _ref_prefix_ids(self, rollout: RolloutView) -> list[int]:
        branches = rollout.trace.branches
        if len(branches) != 1:
            raise ValueError(
                f"opsd supports single-branch traces only; env '{rollout.env_name}' produced {len(branches)} branches."
            )
        branch = branches[0]
        sampled_nodes = [idx for idx, node in enumerate(branch.nodes) if any(node.mask)]
        if len(sampled_nodes) != 1:
            raise ValueError(
                f"opsd supports one sampled model turn; env '{rollout.env_name}' produced {len(sampled_nodes)}."
            )

        demonstration = rollout.trace.info.get(self.demo_key)
        if demonstration is None and hasattr(rollout.trace.task, self.demo_key):
            demonstration = getattr(rollout.trace.task, self.demo_key)
        if demonstration is None:
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the example's info dict or as a "
                f"top-level rollout field (env '{rollout.env_name}', example {rollout.example_id})."
            )

        sampled_idx = sampled_nodes[0]
        messages = [node.message.model_dump(mode="json") for node in branch.nodes[:sampled_idx]]
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

    async def score_batch(self, batch: list[RolloutView]) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def score_one(client, rollout: RolloutView) -> None:
            prefix_ids = self._ref_prefix_ids(rollout)
            assert len(rollout.samples) == 1  # single-step trajectory → one sample
            sample = rollout.samples[0]
            sampled_token_ids = [
                token_id
                for token_id, sampled in zip(sample.completion_ids, sample.completion_mask, strict=True)
                if sampled
            ]
            async with semaphore:
                full_logprobs = await compute_prefill_logprobs(client, pool.model_name, prefix_ids + sampled_token_ids)
            completion_logprobs = full_logprobs[-len(sampled_token_ids) :]
            ref_logprobs: list[float] = []
            offset = 0
            for sampled in sample.completion_mask:
                if sampled:
                    ref_logprobs.append(completion_logprobs[offset])
                    offset += 1
                else:
                    ref_logprobs.append(0.0)
            sample.ref_logprobs = ref_logprobs

        await asyncio.gather(*[score_one(client, rollout) for client, rollout in zip(cycle(pool.train_clients), batch)])
