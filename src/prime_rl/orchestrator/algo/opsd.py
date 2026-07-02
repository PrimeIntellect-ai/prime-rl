from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import openai

from prime_rl.configs.algorithm import OPSDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


class OPSDAlgorithm(Algorithm):
    """On-policy self-distillation (SDFT). The teacher *is* the live policy,
    conditioned on an expert demonstration — no separate model, no extra
    deployment.

    Each sample is prefill-scored under the policy with the demonstration
    prepended as a leading system message: the teacher reads
    ``hint_block + sample.token_ids`` and the demo-conditioned logprobs over the
    sample's tokens become ``ref_logprobs`` (the trainer's ref_kl target). The
    sample is scored verbatim — no re-rendering — so the join lands on the
    message-closing special token (BPE-clean) and it's robust to tools /
    multimodal prompts and any number of turns. No scalar advantage is
    assigned."""

    action_loss_type = "ref_kl"

    def __init__(self, config: OPSDAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.demo_key = config.demo_key
        self.template = config.template
        self.renderer_config = config.renderer
        self.renderer: Renderer | None = None  # opsd builds its own in setup()
        # Self-distillation: the teacher *is* the live policy. Scoring against
        # the shared policy pool tracks its current weights, model name, and
        # endpoint churn for free.
        self.teacher_pool = self.policy_pool

    async def setup(self) -> None:
        """Build opsd's own hint-block renderer from config — it is not handed
        the policy's renderer. The tokenizer is always the live policy's
        (self-distillation has no separate model), so the hint tokenizes
        identically to the policy's own prompts."""
        from renderers.base import create_renderer, load_tokenizer

        self.renderer = create_renderer(load_tokenizer(self.policy_pool.model_name), self.renderer_config)

    def _demonstration(self, rollout: Rollout) -> str:
        demonstration = rollout.info.get(self.demo_key)
        if demonstration is None:
            demonstration = getattr(rollout.task, self.demo_key, None)
        if demonstration is None:
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the trace info dict or on the task "
                f"(env '{rollout.env_name}', task {rollout.task.idx})."
            )
        return demonstration

    async def score_rollout(self, rollout: Rollout) -> None:
        pool = self.teacher_pool
        renderer = self.renderer
        assert renderer is not None, "renderer not built — Algorithm.setup() must run first"
        hint = self.template.format(demonstration=self._demonstration(rollout))
        hint_block = renderer.render_ids([{"role": "system", "content": hint}], add_generation_prompt=False)

        async def score_sample(sample: TrainingSample) -> None:
            try:
                full_logprobs = await pool.score(hint_block + list(sample.token_ids))
            except openai.BadRequestError as e:
                if "longer than the maximum model length" not in str(e):
                    raise
                # Generation reserves no context headroom for the hint, so a
                # near-max-context trajectory + hint_block can exceed
                # max_model_len. Rare tail case: drop the rollout via the
                # pre-filter path instead of crashing the run.
                rollout.is_filtered = True
                rollout.filter_results["opsd_hint_overflow"] = True
                get_logger().warning(
                    f"OPSD hint overflow: hint ({len(hint_block)} tok) + trajectory "
                    f"({len(sample.token_ids)} tok) exceeds max_model_len; dropping rollout "
                    f"(env {rollout.env_name!r}, task {rollout.task.idx})."
                )
                return
            # Drop the hint's own logprobs; the tail aligns full-length to
            # sample.token_ids (demo-conditioned, the trainer's ref_kl target).
            sample.ref_logprobs = full_logprobs[len(hint_block) :]

        await asyncio.gather(*(score_sample(sample) for sample in rollout.samples))
