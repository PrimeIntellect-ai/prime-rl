from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import OPSDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.clients import ModelScorer
from prime_rl.orchestrator.utils import episode_env_name

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.clients import InferenceClient


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

    def __init__(self, config: OPSDAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.demo_key = config.demo_key
        self.template = config.template
        self.renderer: Renderer | None = None
        self.teacher_clients: ModelScorer | None = None

    def bind(self, client: vf.ClientConfig, model: str) -> None:
        """Self-distillation: the teacher *is* the live policy, scored on the endpoint the
        rollouts sample from. The hint block is rendered with the policy's renderer config and
        tokenizer, so it tokenizes identically to the policy's own prompts."""
        from renderers.base import create_renderer, load_tokenizer

        self.teacher_clients = ModelScorer(client, model)
        self.renderer = create_renderer(
            load_tokenizer(getattr(client, "renderer_model_name", None) or model), getattr(client, "renderer", None)
        )

    def _demonstration(self, episode: vf.Episode, trace: vf.Trace) -> str:
        demonstration = trace.info.get(self.demo_key)
        if demonstration is None:
            demonstration = getattr(trace.task.data, self.demo_key, None)
        if demonstration is None:
            env_name = episode_env_name(episode)
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the trace info dict or on the task "
                f"(env '{env_name}', task {trace.task.data.idx})."
            )
        return demonstration

    def sample_context(self, episode: vf.Episode, trace: vf.Trace) -> dict:
        return {"demonstration": self._demonstration(episode, trace)}

    async def score_sample(self, token_ids: np.ndarray, sampled: np.ndarray, context: dict) -> np.ndarray:
        teacher = self.teacher_clients
        renderer = self.renderer
        assert teacher is not None and renderer is not None, "not bound — Algorithm.bind() must run first"
        hint = self.template.format(demonstration=context["demonstration"])
        hint_block = renderer.render_ids([{"role": "system", "content": hint}], add_generation_prompt=False)
        full_logprobs = await teacher.score([*hint_block, *token_ids.tolist()])
        return np.where(sampled, np.asarray(full_logprobs[len(hint_block) :], dtype=np.float32), np.float32(0.0))
