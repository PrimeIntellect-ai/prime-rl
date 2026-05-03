"""/v1/generate endpoint — accepts pre-tokenized inputs.

Text-only tokens in, tokens out. The Renderer does all tokenization client-side.
No Jinja rendering, no server-side chat template application.

VLMs do not use this endpoint. The orchestrator routes VLMs to MITO
(/v1/chat/completions) where vLLM handles image preprocessing and chat
templating server-side.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import Request
from pydantic import BaseModel
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.inputs.engine import tokens_input
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


# ── Request / Response schemas ───────────────────────────────────────


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt_token_ids: list[int]

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
    # Prefix-cache invalidation salt. Must match main's
    # /v1/chat/completions/tokens path: the orchestrator sets
    # `extra_body["cache_salt"] = str(ckpt_step)` on every rollout
    # request. vLLM's KV cache hashes include this salt, so when the
    # step changes the cache misses and KV is recomputed with fresh
    # weights. Without this, renderers path silently reuses stale KV
    # from before the latest weight update and its logprobs drift from
    # the trainer's forward pass (mismatch_kl grows 3x over training).
    cache_salt: str | None = None


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
    """Lightweight generate handler — tokens in, tokens out."""

    def __init__(self, engine_client: EngineClient, chat_handler: Any | None = None):
        self.engine_client = engine_client
        self.chat_handler = chat_handler

    async def generate(self, request: GenerateRequest, raw_request: Request) -> GenerateResponse | ErrorResponse | dict:
        # Pre-rendered TokensInput shape (type="token") — avoids vLLM's
        # "raw prompt" deprecation that targets plain lists/strings.
        engine_prompt = tokens_input(request.prompt_token_ids, cache_salt=request.cache_salt)

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

        response = GenerateResponse(
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
        nonfinite = _find_non_finite_generate_value(
            response,
            request=request,
            data_parallel_rank=data_parallel_rank,
            tokenizer_getter=lambda: _get_diagnostic_tokenizer(
                self.chat_handler,
                self.engine_client,
            ),
        )
        if nonfinite is not None:
            return _non_finite_generate_error(nonfinite)
        return response


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


def _find_non_finite_generate_value(
    response: GenerateResponse,
    *,
    request: GenerateRequest,
    data_parallel_rank: int | None,
    tokenizer_getter: Callable[[], Any | None],
) -> dict[str, Any] | None:
    prompt_len = len(response.prompt_token_ids)
    common = {
        "request_id": response.id,
        "model": response.model,
        "prompt_len": prompt_len,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "n": request.n,
        "cache_salt": request.cache_salt,
        "data_parallel_rank": data_parallel_rank,
    }

    for choice in response.choices:
        completion_len = len(choice.token_ids)
        for token_offset, logprob in enumerate(choice.logprobs):
            if math.isfinite(logprob):
                continue
            token_id = (
                choice.token_ids[token_offset]
                if token_offset < completion_len
                else None
            )
            tokenizer = tokenizer_getter()
            return {
                **common,
                "field": "choices.logprobs",
                "choice_index": choice.index,
                "token_offset": token_offset,
                "token_id": token_id,
                "completion_len": completion_len,
                "finish_reason": choice.finish_reason,
                **_token_window_context(
                    "completion",
                    choice.token_ids,
                    token_offset,
                    tokenizer=tokenizer,
                ),
                **_short_sequence_context(
                    "completion",
                    choice.token_ids,
                    tokenizer=tokenizer,
                ),
                **_prompt_tail_context(response.prompt_token_ids, tokenizer=tokenizer),
                "value": repr(logprob),
            }

    if response.prompt_logprobs is not None:
        for token_offset, logprob in enumerate(response.prompt_logprobs):
            if logprob is None or math.isfinite(logprob):
                continue
            token_id = (
                response.prompt_token_ids[token_offset]
                if token_offset < prompt_len
                else None
            )
            tokenizer = tokenizer_getter()
            return {
                **common,
                "field": "prompt_logprobs",
                "token_offset": token_offset,
                "token_id": token_id,
                **_token_window_context(
                    "prompt",
                    response.prompt_token_ids,
                    token_offset,
                    tokenizer=tokenizer,
                ),
                **_prompt_tail_context(response.prompt_token_ids, tokenizer=tokenizer),
                "value": repr(logprob),
            }

    return None


def _get_diagnostic_tokenizer(*objects: Any | None) -> Any | None:
    for obj in objects:
        tokenizer = _find_tokenizer_on_object(obj)
        if tokenizer is not None:
            return tokenizer
    return None


def _find_tokenizer_on_object(obj: Any | None) -> Any | None:
    if obj is None:
        return None
    if _can_decode_tokens(obj):
        return obj

    for attr in ("tokenizer", "_tokenizer"):
        tokenizer = getattr(obj, attr, None)
        if _can_decode_tokens(tokenizer):
            return tokenizer

    renderer = getattr(obj, "renderer", None)
    if renderer is not None:
        tokenizer = _find_tokenizer_on_object(renderer)
        if tokenizer is not None:
            return tokenizer

    get_tokenizer = getattr(obj, "get_tokenizer", None)
    if callable(get_tokenizer):
        try:
            tokenizer = get_tokenizer()
        except Exception:
            return None
        if _can_decode_tokens(tokenizer):
            return tokenizer

    return None


def _can_decode_tokens(tokenizer: Any | None) -> bool:
    return tokenizer is not None and callable(getattr(tokenizer, "decode", None))


def _token_window_context(
    prefix: str,
    token_ids: list[int],
    token_offset: int,
    *,
    tokenizer: Any | None,
    radius: int = 16,
) -> dict[str, Any]:
    start = max(0, token_offset - radius)
    end = min(len(token_ids), token_offset + radius + 1)
    window_ids = token_ids[start:end]
    return {
        f"{prefix}_token_window_start": start,
        f"{prefix}_token_window_end": end,
        f"{prefix}_token_window_token_ids": window_ids,
        **_decode_token_context(
            f"{prefix}_token_window",
            window_ids,
            tokenizer=tokenizer,
        ),
    }


def _short_sequence_context(
    prefix: str,
    token_ids: list[int],
    *,
    tokenizer: Any | None,
    max_tokens: int = 128,
) -> dict[str, Any]:
    if len(token_ids) > max_tokens:
        return {}
    return {
        f"{prefix}_token_ids": token_ids,
        **_decode_token_context(prefix, token_ids, tokenizer=tokenizer),
    }


def _prompt_tail_context(
    prompt_token_ids: list[int],
    *,
    tokenizer: Any | None,
    max_tokens: int = 64,
) -> dict[str, Any]:
    tail_ids = prompt_token_ids[-max_tokens:]
    return {
        "prompt_tail_token_start": len(prompt_token_ids) - len(tail_ids),
        "prompt_tail_token_ids": tail_ids,
        **_decode_token_context("prompt_tail", tail_ids, tokenizer=tokenizer),
    }


def _decode_token_context(
    prefix: str,
    token_ids: list[int],
    *,
    tokenizer: Any | None,
) -> dict[str, Any]:
    if not _can_decode_tokens(tokenizer):
        return {}

    context: dict[str, Any] = {}
    try:
        context[f"{prefix}_text"] = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )
    except Exception as exc:
        context[f"{prefix}_decode_error"] = repr(exc)

    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_ids_to_tokens):
        try:
            context[f"{prefix}_tokens"] = list(convert_ids_to_tokens(token_ids))
        except Exception as exc:
            context[f"{prefix}_tokens_error"] = repr(exc)
    return context


def _non_finite_generate_error(context: dict[str, Any]) -> ErrorResponse:
    message = "Non-finite /v1/generate response value before JSON serialization: " + json.dumps(context, sort_keys=True)
    logger.error(message)
    return ErrorResponse(
        error=ErrorInfo(
            message=message,
            type="BadRequestError",
            param=context["field"],
            code=400,
        )
    )
