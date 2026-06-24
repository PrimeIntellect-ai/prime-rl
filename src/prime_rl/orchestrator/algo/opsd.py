from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, OPSDAlgorithmConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.client import StaticInferencePool

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
        self.teacher_pool: StaticInferencePool | None = None  # static teacher endpoint, connected in setup()

    async def setup(self) -> None:
        pool = await self.connect(self.teacher)
        if not isinstance(pool, StaticInferencePool):
            raise TypeError("opsd teacher must be a static endpoint — prefill scoring needs fixed endpoints")
        self.teacher_pool = pool

    def _ref_prefix_ids(self, rollout: RolloutView) -> list[int]:
        trace = rollout.raw
        if trace.num_turns != 1:
            raise ValueError(
                f"opsd supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {trace.num_turns} model turn(s)."
            )
        demonstration = trace.info.get(self.demo_key)
        if demonstration is None:
            demonstration = getattr(trace.task, self.demo_key, None)
        if demonstration is None:
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the trace info dict or on the task "
                f"(env '{rollout.env_name}', task {trace.task.idx})."
            )

        # The scoring prompt is the branch's leading non-sampled (input)
        # messages — the context the model conditioned on before responding.
        branch = trace.branches[0]
        messages = [node.message.model_dump(exclude_none=True) for node in branch.nodes if not node.sampled]
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

        async def score_one(rollout: RolloutView) -> None:
            prefix_ids = self._ref_prefix_ids(rollout)
            assert len(rollout.samples) == 1  # single-step trajectory → one sample
            sample = rollout.samples[0]
            completion_ids = [t for t, trains in zip(sample.token_ids, sample.mask) if trains]
            async with semaphore:
                full_logprobs = await pool.score(prefix_ids + completion_ids)
            completion_logprobs = full_logprobs[-len(completion_ids) :]
            # Scatter the demo-conditioned completion logprobs back onto the
            # sample's trainable positions; full-length-N, 0.0 elsewhere.
            ref_logprobs = [0.0] * len(sample.token_ids)
            li = 0
            for i, trains in enumerate(sample.mask):
                if trains:
                    ref_logprobs[i] = completion_logprobs[li]
                    li += 1
            sample.ref_logprobs = ref_logprobs

        await asyncio.gather(*(score_one(rollout) for rollout in batch))
