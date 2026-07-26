"""Small Prime extensions to vLLM's canonical token-in/token-out handler."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from pydantic import ConfigDict
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata, UsageInfo
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.compat import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    ServingTokens,
    serving_renderer,
)
from prime_rl.inference.vllm.routed_experts import compact_vllm_routed_experts


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    model_config = ConfigDict(extra="allow")

    routed_experts: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    created: int | None = None
    choices: list[PrimeRlGenerateResponseChoice]
    usage: UsageInfo | None = None


def _prime_response(
    response: GenerateResponse,
    *,
    usage: UsageInfo | None,
    model_name: str,
    routed_experts_start: int,
) -> PrimeRlGenerateResponse:
    """Replace only routed-expert choices while retaining all response fields."""
    payload = response.model_dump(exclude={"choices"})
    payload.setdefault("model", model_name)
    payload.setdefault("created", int(time.time()))
    if payload.get("usage") is None:
        payload["usage"] = usage
    payload["choices"] = [
        {
            **choice.model_dump(exclude={"routed_experts"}),
            "routed_experts": compact_vllm_routed_experts(
                choice.routed_experts,
                start=routed_experts_start,
            ),
        }
        for choice in response.choices
    ]
    return PrimeRlGenerateResponse.model_validate(payload)


class PrimeRlServingTokens(ServingTokens):
    """Add KV handoff and Prime's compact routed-expert response encoding."""

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        if request.kv_transfer_params is None:
            return await super().serve_tokens(request, raw_request)

        forwarded = request.model_copy(deep=True)
        extra_args = dict(forwarded.sampling_params.extra_args or {})
        extra_args["kv_transfer_params"] = forwarded.kv_transfer_params
        forwarded.sampling_params.extra_args = extra_args
        return await super().serve_tokens(forwarded, raw_request)

    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        response = await super().serve_tokens_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            request_metadata,
        )
        if not isinstance(response, GenerateResponse):
            return response
        return _prime_response(
            response,
            usage=request_metadata.final_usage_info,
            model_name=model_name,
            routed_experts_start=request.sampling_params.routed_experts_prompt_start or 0,
        )


def build_prime_serving_tokens(upstream: ServingTokens) -> PrimeRlServingTokens:
    """Construct the Prime subclass through vLLM's public initializer."""
    return PrimeRlServingTokens(
        upstream.engine_client,
        upstream.models,
        serving_renderer(upstream),
        request_logger=upstream.request_logger,
        return_tokens_as_token_ids=upstream.return_tokens_as_token_ids,
        force_no_detokenize=upstream.force_no_detokenize,
        enable_prompt_tokens_details=True,
        enable_log_outputs=upstream.enable_log_outputs,
    )
