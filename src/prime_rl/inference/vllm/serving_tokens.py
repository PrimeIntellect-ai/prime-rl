"""Small Prime extensions to vLLM's canonical token-in/token-out handler."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.routed_experts import compact_vllm_routed_experts


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    routed_experts: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


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
        if not isinstance(response, GenerateResponse) or not any(
            choice.routed_experts is not None for choice in response.choices
        ):
            return response
        start = request.sampling_params.routed_experts_prompt_start or 0
        return PrimeRlGenerateResponse(
            **response.model_dump(exclude={"choices"}),
            choices=[
                PrimeRlGenerateResponseChoice(
                    **choice.model_dump(exclude={"routed_experts"}),
                    routed_experts=compact_vllm_routed_experts(choice.routed_experts, start=start),
                )
                for choice in response.choices
            ],
        )
