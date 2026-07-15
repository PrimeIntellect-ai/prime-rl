from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import numpy as np
import openai

from prime_rl.configs.algorithm import OPSDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.transport.types import EncodedTensor
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
        self.demo_transform = config.demo_transform
        self.template = config.template
        self.max_score_tokens = config.max_score_tokens
        self.granularity = config.ref_logprob_granularity
        self.ref_top_k = config.ref_top_k
        self.diag_top_k = config.diag_top_k
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

    @staticmethod
    def _tool_sequence_plan(demonstration: str) -> str:
        """Reduce a validated General Agent tool-call chain to its structure.

        This keeps the plan arm scientifically distinct from full-answer OPSD:
        ordered tool and argument names survive, while every concrete argument
        value is withheld.
        """
        try:
            calls = json.loads(demonstration)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool_sequence_plan requires a JSON tool-call chain") from exc
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_sequence_plan requires a non-empty list of tool calls")

        steps: list[str] = []
        for index, call in enumerate(calls, start=1):
            if not isinstance(call, list | tuple) or len(call) != 2:
                raise ValueError(f"tool_sequence_plan call {index} must be [tool_name, arguments]")
            tool_name, arguments = call
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                raise ValueError(f"tool_sequence_plan call {index} has invalid tool name or arguments")
            argument_names = ", ".join(f"`{name}`" for name in arguments) or "no arguments"
            steps.append(f"{index}. Call `{tool_name}` with {argument_names}.")
        return "\n".join(steps)

    def _demonstration(self, rollout: Rollout) -> str:
        demonstration = rollout.info.get(self.demo_key)
        if demonstration is None:
            demonstration = getattr(rollout.task, self.demo_key, None)
        if demonstration is None:
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the trace info dict or on the task "
                f"(env '{rollout.env_name}', task {rollout.task.idx})."
            )
        if not isinstance(demonstration, str):
            raise ValueError(f"opsd demonstration '{self.demo_key}' must be a string")
        if self.demo_transform == "tool_sequence_plan":
            return self._tool_sequence_plan(demonstration)
        return demonstration

    async def score_rollout(self, rollout: Rollout) -> None:
        pool = self.teacher_pool
        renderer = self.renderer
        assert renderer is not None, "renderer not built — Algorithm.setup() must run first"
        hint = self.template.format(demonstration=self._demonstration(rollout))
        hint_block = renderer.render_ids([{"role": "system", "content": hint}], add_generation_prompt=False)

        async def score_sample(sample: TrainingSample) -> None:
            token_ids = list(sample.token_ids)
            n_scored = len(token_ids)
            if self.max_score_tokens is not None:
                # -1: the scoring request generates one token (max_tokens=1), so
                # the server needs prompt + 1 <= max_model_len.
                budget = self.max_score_tokens - len(hint_block) - 1
                if budget <= 0:
                    rollout.is_filtered = True
                    rollout.filter_results["opsd_hint_overflow"] = True
                    get_logger().warning(
                        f"OPSD hint overflow: hint block ({len(hint_block)} tok) alone exceeds "
                        f"max_score_tokens={self.max_score_tokens}; dropping rollout "
                        f"(env {rollout.env_name!r}, task {rollout.task.idx})."
                    )
                    return
                if n_scored > budget:
                    # Generation reserves no context headroom for the hint, so a
                    # near-max-context trajectory + hint can exceed the scoring
                    # window. Score the head and mask the unscored tail out of
                    # the loss — dropping the (expensive) rollout would also pay
                    # a serial backfill rollout on top.
                    n_scored = budget
                    for i in range(n_scored, len(token_ids)):
                        sample.mask[i] = False
                    get_logger().warning(
                        f"OPSD hint overflow: hint ({len(hint_block)} tok) + trajectory "
                        f"({len(token_ids)} tok) exceeds max_score_tokens={self.max_score_tokens}; "
                        f"scoring first {n_scored} tokens, masking the tail "
                        f"(env {rollout.env_name!r}, task {rollout.task.idx})."
                    )
            scores = diag_scores = None
            try:
                if self.granularity == "top_k":
                    scores = await pool.score(hint_block + token_ids[:n_scored], top_k=self.ref_top_k)
                    full_logprobs = scores.logprobs
                elif self.diag_top_k is not None:
                    # Diagnostics-only collection: the same scoring call with
                    # top_k, but the objective stays single_token — the realized
                    # logprobs below are extracted exactly as without top_k.
                    diag_scores = await pool.score(hint_block + token_ids[:n_scored], top_k=self.diag_top_k)
                    full_logprobs = diag_scores.logprobs
                else:
                    full_logprobs = await pool.score(hint_block + token_ids[:n_scored])
            except openai.BadRequestError as e:
                if "longer than the maximum model length" not in str(e):
                    raise
                # Backstop for max_score_tokens unset (or set above the server's
                # true window): drop via the pre-filter path instead of crashing.
                rollout.is_filtered = True
                rollout.filter_results["opsd_hint_overflow"] = True
                get_logger().warning(
                    f"OPSD hint overflow: hint ({len(hint_block)} tok) + trajectory "
                    f"({n_scored} tok) rejected by the inference server; dropping rollout "
                    f"(env {rollout.env_name!r}, task {rollout.task.idx})."
                )
                return
            # Drop the hint's own logprobs; the tail aligns full-length to
            # sample.token_ids (demo-conditioned, the trainer's ref_kl target).
            # A truncated sample pads 0.0 over the masked-out unscored tail to
            # keep the per-token alignment invariant.
            ref_logprobs = full_logprobs[len(hint_block) :]
            sample.ref_logprobs = ref_logprobs + [0.0] * (len(token_ids) - n_scored)
            if scores is not None:
                # Same hint slice; truncated tail rows pad with (id 0, logprob
                # -1e9) — exp(-1e9) = 0, so they carry no teacher mass.
                topk_ids = np.asarray(scores.topk_ids[len(hint_block) :], dtype=np.int32)
                topk_logprobs = np.asarray(scores.topk_logprobs[len(hint_block) :], dtype=np.float32)
                n_pad = len(token_ids) - n_scored
                if n_pad > 0:
                    topk_ids = np.concatenate([topk_ids, np.zeros((n_pad, self.ref_top_k), dtype=np.int32)])
                    topk_logprobs = np.concatenate(
                        [topk_logprobs, np.full((n_pad, self.ref_top_k), -1e9, dtype=np.float32)]
                    )
                sample.ref_topk_token_ids = EncodedTensor.from_numpy(topk_ids)
                sample.ref_topk_logprobs = EncodedTensor.from_numpy(topk_logprobs)
            if diag_scores is not None:
                # Diagnostics-only top-k (single_token objective): same hint
                # slice and truncation padding as ref_topk_* above, but shipped
                # as diag_topk_* — consumed only by token export, never the loss.
                diag_ids = np.asarray(diag_scores.topk_ids[len(hint_block) :], dtype=np.int32)
                diag_logprobs = np.asarray(diag_scores.topk_logprobs[len(hint_block) :], dtype=np.float32)
                n_pad = len(token_ids) - n_scored
                if n_pad > 0:
                    diag_ids = np.concatenate([diag_ids, np.zeros((n_pad, self.diag_top_k), dtype=np.int32)])
                    diag_logprobs = np.concatenate(
                        [diag_logprobs, np.full((n_pad, self.diag_top_k), -1e9, dtype=np.float32)]
                    )
                sample.diag_topk_token_ids = EncodedTensor.from_numpy(diag_ids)
                sample.diag_topk_logprobs = EncodedTensor.from_numpy(diag_logprobs)

        await asyncio.gather(*(score_sample(sample) for sample in rollout.samples))
