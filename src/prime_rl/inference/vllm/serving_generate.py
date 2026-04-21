"""/v1/generate endpoint — accepts pre-tokenized + pre-processed inputs.

Text-only: accepts prompt_token_ids, passes to engine.
VLM: accepts prompt_token_ids + raw images (multi_modal_data), vLLM processes
images and expands placeholders. The Renderer does all text tokenization
client-side — vLLM just handles image processing and generation.

No Jinja rendering, no server-side chat template application.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from io import BytesIO
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import Request
from PIL import Image
from pydantic import BaseModel, Field
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


# ── Request / Response schemas ───────────────────────────────────────


class RawImageData(BaseModel):
    """Raw image bytes — vLLM processes them server-side."""

    data: str = Field(description="Base64-encoded image bytes")
    media_type: str = Field(default="image/png")


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt_token_ids: list[int]
    images: list[RawImageData] | None = None

    # When unset, fill from max_model_len - prompt_len at request time so we
    # match /v1/chat/completions behavior. The previous 4096 hard default
    # silently truncated long completions on 8k+ context runs (e.g. hendrycks
    # reasoning rollouts capped at 4096 tokens, making rendered rollouts look
    # shorter than main's for the same model).
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    n: int = 1
    stop_token_ids: list[int] | None = None
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    prompt_logprobs: bool = False
    priority: int = 0


class GenerateChoiceResponse(BaseModel):
    index: int
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None = None
    routed_experts: dict | None = None


class GenerateResponse(BaseModel):
    id: str
    model: str
    prompt_token_ids: list[int]
    choices: list[GenerateChoiceResponse]
    usage: dict
    prompt_logprobs: list[float | None] | None = None


# ── Handler ──────────────────────────────────────────────────────────


class OpenAIServingGenerate:
    """Lightweight generate handler — tokens + optional images in, tokens out."""

    def __init__(self, engine_client: EngineClient, chat_handler: Any | None = None):
        self.engine_client = engine_client
        self.chat_handler = chat_handler

    async def generate(self, request: GenerateRequest, raw_request: Request) -> GenerateResponse | dict:
        engine_prompt: dict[str, Any] = {
            "prompt_token_ids": request.prompt_token_ids,
        }

        if request.images:
            pil_images = [Image.open(BytesIO(base64.b64decode(img.data))) for img in request.images]
            engine_prompt["multi_modal_data"] = {"image": pil_images}

        # Match /v1/chat/completions: if the client didn't ask for a specific
        # cap, let the model generate up to whatever room is left in context.
        # vLLM v1 AsyncLLM exposes model_config directly (no async getter).
        max_tokens = request.max_tokens
        if max_tokens is None:
            max_model_len = self.engine_client.model_config.max_model_len
            max_tokens = max(1, max_model_len - len(request.prompt_token_ids))

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            seed=request.seed,
            n=request.n,
            stop_token_ids=request.stop_token_ids or [],
            repetition_penalty=request.repetition_penalty,
            min_tokens=request.min_tokens,
            logprobs=1,
            prompt_logprobs=1 if request.prompt_logprobs else None,
            skip_special_tokens=False,
        )

        request_id = f"gen-{uuid4().hex[:16]}"
        routed_experts_map: dict[int, dict] = {}
        final_output: RequestOutput | None = None
        data_parallel_rank = None
        lora_request = None
        trace_headers = None
        if self.chat_handler is not None:
            data_parallel_rank = self.chat_handler._get_data_parallel_rank(raw_request)
            trace_headers = await self.chat_handler._get_trace_headers(raw_request.headers)
            lora_request = self.chat_handler._maybe_get_adapters(request)

        generator = self.engine_client.generate(
            engine_prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=data_parallel_rank,
        )

        # Drain the generator without polling ``raw_request.is_disconnected``
        # per decode step. That poll is one ASGI ``receive()`` await per
        # yielded token — at ~2k concurrent rollouts each producing dozens of
        # tokens, it was the single largest per-step overhead on the /generate
        # path (vLLM's own chat completions handler uses the CancelledError
        # pattern for the same reason). Starlette cancels this coroutine on
        # client disconnect, so the except branch still catches it.
        try:
            async for output in generator:
                for comp_output in output.outputs:
                    if comp_output.routed_experts is not None:
                        routed_experts_map[comp_output.index] = _encode_routed_experts(comp_output.routed_experts)
                final_output = output
        except asyncio.CancelledError:
            await self.engine_client.abort(request_id)
            raise

        if final_output is None:
            return {"error": "No output generated"}

        choices = []
        for output in final_output.outputs:
            token_ids = list(output.token_ids)
            logprobs_list: list[float] = []
            if output.logprobs:
                for i, lp_dict in enumerate(output.logprobs):
                    if i < len(token_ids) and token_ids[i] in lp_dict:
                        logprobs_list.append(lp_dict[token_ids[i]].logprob)
                    else:
                        logprobs_list.append(0.0)

            choices.append(
                GenerateChoiceResponse(
                    index=output.index,
                    token_ids=token_ids,
                    logprobs=logprobs_list,
                    finish_reason=output.finish_reason,
                    routed_experts=routed_experts_map.get(output.index),
                )
            )

        prompt_len = len(final_output.prompt_token_ids)
        completion_len = sum(len(c.token_ids) for c in choices)
        prompt_logprobs = _extract_prompt_logprobs(final_output.prompt_logprobs)

        return GenerateResponse(
            id=request_id,
            model=request.model or "",
            prompt_token_ids=list(final_output.prompt_token_ids),
            choices=choices,
            usage={
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            },
            prompt_logprobs=prompt_logprobs,
        )


def _encode_routed_experts(arr: np.ndarray) -> dict:
    return {
        "data": base64.b85encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
    }


def _extract_prompt_logprobs(
    prompt_logprobs: list[dict[int, Any] | None] | Mapping[int, Any] | None,
) -> list[float | None] | None:
    if prompt_logprobs is None:
        return None
    if isinstance(prompt_logprobs, Mapping):
        prompt_logprobs = [prompt_logprobs]

    extracted: list[float | None] = []
    for token_logprobs in prompt_logprobs:
        if not token_logprobs:
            extracted.append(None)
            continue
        selected = next(iter(token_logprobs.values()))
        logprob = selected.logprob if hasattr(selected, "logprob") else selected.get("logprob")
        extracted.append(float(logprob) if logprob is not None else None)
    return extracted
