from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChatCompletionChunkChoice,
)
from pydantic import BaseModel, Field
from verifiers.clients.openai_chat_completions_token_client import OpenAIChatCompletionsTokenClient
from verifiers.types import Messages, TrajectoryStep
from verifiers.types import State as VfState
from verifiers.utils.message_utils import concat_messages, from_raw_message
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from vllm.logger import init_logger

router = APIRouter()

logger = init_logger("vllm.entrypoints.openai.rollout_gateway")

RolloutStatus = Literal["active", "cancelled", "completed"]


class RegisterRolloutRequest(BaseModel):
    model: str
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    max_turns: int = -1
    max_seq_len: int | None = None


class TokensResponse(BaseModel):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool
    is_truncated: bool
    routed_experts: list[list[list[int]]] | None = None


class TrajectoryStepResponse(BaseModel):
    prompt: Messages
    completion: Messages
    tokens: TokensResponse | None = None
    reward: float | None = None
    advantage: float | None = None
    is_truncated: bool = False
    trajectory_id: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)


class TrajectoryResponse(BaseModel):
    rollout_id: str
    status: RolloutStatus
    num_turns: int
    model: str
    prompt: Messages
    completion: Messages
    is_truncated: bool
    trajectory: list[TrajectoryStepResponse]


class RegisterRolloutResponse(BaseModel):
    rollout_id: str
    status: RolloutStatus
    model: str
    max_turns: int
    max_seq_len: int | None


class RolloutStatusResponse(BaseModel):
    rollout_id: str
    status: RolloutStatus


@dataclass
class RolloutConfig:
    model: str
    sampling_params: dict[str, Any] = field(default_factory=dict)
    max_turns: int = -1
    max_seq_len: int | None = None


@dataclass
class RolloutState:
    config: RolloutConfig
    localhost_client: AsyncOpenAI
    vf_client: OpenAIChatCompletionsTokenClient
    vf_state: VfState
    dp_rank: int = 0
    turn_count: int = 0
    status: RolloutStatus = "active"
    is_truncated: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RolloutRegistry:
    def __init__(self, port: int, dp_size: int = 1):
        self._port = port
        self._dp_size = dp_size
        self._rollouts: dict[str, RolloutState] = {}
        self._lock = asyncio.Lock()

    def _least_loaded_dp_rank(self) -> int:
        rank_counts = Counter(r.dp_rank for r in self._rollouts.values())
        return min(range(self._dp_size), key=lambda r: rank_counts.get(r, 0))

    async def register(self, rollout_id: str, config: RolloutConfig) -> RolloutState:
        async with self._lock:
            if rollout_id in self._rollouts:
                raise ValueError(f"Rollout already registered: {rollout_id}")

            dp_rank = self._least_loaded_dp_rank()
            headers = {"X-data-parallel-rank": str(dp_rank)} if self._dp_size > 1 else {}
            localhost_client = AsyncOpenAI(
                base_url=f"http://localhost:{self._port}/v1",
                api_key="EMPTY",
                max_retries=0,
                default_headers=headers,
            )
            vf_client = OpenAIChatCompletionsTokenClient(localhost_client)
            vf_state = VfState(
                model=config.model,
                oai_tools=[],
                trajectory=[],
                trajectory_id="",
                prompt=[],
                completion=[],
                is_truncated=False,
                _cached_suffix_ids=None,
            )
            rollout_state = RolloutState(
                config=config,
                localhost_client=localhost_client,
                vf_client=vf_client,
                vf_state=vf_state,
                dp_rank=dp_rank,
            )
            self._rollouts[rollout_id] = rollout_state
            logger.info(f"rollout={rollout_id} registered dp_rank={dp_rank}/{self._dp_size}")
            return rollout_state

    def _get(self, rollout_id: str) -> RolloutState:
        try:
            return self._rollouts[rollout_id]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}") from None

    async def get(self, rollout_id: str) -> RolloutState:
        async with self._lock:
            return self._get(rollout_id)

    async def cancel(self, rollout_id: str) -> RolloutState:
        async with self._lock:
            rollout = self._get(rollout_id)
            rollout.status = "cancelled"
            return rollout

    async def unregister(self, rollout_id: str) -> RolloutState:
        async with self._lock:
            rollout = self._get(rollout_id)
            del self._rollouts[rollout_id]
        await rollout.localhost_client.close()
        return rollout


def _get_rollout_registry(request: Request) -> RolloutRegistry:
    registry = getattr(request.app.state, "rollout_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Rollout gateway is disabled. Start vLLM with api_server_count=1 to enable /v1/rollouts endpoints."
            ),
        )
    return registry


def _render_completion(vf_state: VfState) -> Messages:
    trajectory = vf_state["trajectory"]
    if not trajectory:
        return []

    last_prompt = trajectory[-1]["prompt"]
    last_completion = trajectory[-1]["completion"]
    full_conversation = concat_messages([last_prompt, last_completion])

    final_env_response = vf_state.get("final_env_response")
    if final_env_response:
        full_conversation = concat_messages([full_conversation, final_env_response])

    rollout_prompt = vf_state["prompt"]
    return full_conversation[len(rollout_prompt) :]


def _validate_rollout_status(rollout: RolloutState, rollout_id: str) -> None:
    if rollout.status == "cancelled":
        raise HTTPException(status_code=409, detail=f"Rollout cancelled: {rollout_id}")
    if rollout.status == "completed":
        raise HTTPException(status_code=409, detail=f"Rollout completed: {rollout_id}")


async def _call_chat_with_messages(
    rollout: RolloutState,
    raw_messages: list[ChatCompletionMessageParam],
    tools: list[dict[str, Any]] | None,
    sampling_args: dict[str, Any],
) -> ChatCompletion:
    request_body: dict[str, Any] = {
        "model": rollout.config.model,
        "messages": raw_messages,
        **sampling_args,
    }
    if tools:
        request_body["tools"] = tools

    return await rollout.localhost_client.post(
        "/chat/completions",
        body=request_body,
        cast_to=ChatCompletion,
    )


async def _fake_stream(response: ChatCompletion):
    """
    Fake two chunks stream response for the rollout gateway.
    """
    choice = response.choices[0]
    message = choice.message

    delta_tool_calls = None
    if message.tool_calls:
        delta_tool_calls = [
            ChoiceDeltaToolCall(
                index=i,
                id=tool_call.id,
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for i, tool_call in enumerate(message.tool_calls)
        ]

    content_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    content=message.content,
                    tool_calls=delta_tool_calls,
                ),
                finish_reason=None,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    yield f"data: {content_chunk.model_dump_json()}\n\n"

    finish_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=choice.finish_reason,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    yield f"data: {finish_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/rollouts/{rollout_id}/register", response_model=RegisterRolloutResponse)
async def register_rollout(
    rollout_id: str,
    body: RegisterRolloutRequest,
    request: Request,
):
    registry = _get_rollout_registry(request)
    sampling_params = dict(body.sampling_params)
    extra_body = sampling_params.pop("extra_body", {})
    sampling_params = {**sampling_params, **extra_body, "logprobs": True, "return_token_ids": True}
    rollout_config = RolloutConfig(
        model=body.model,
        sampling_params=sampling_params,
        max_turns=body.max_turns,
        max_seq_len=body.max_seq_len,
    )
    try:
        rollout = await registry.register(rollout_id, rollout_config)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return RegisterRolloutResponse(
        rollout_id=rollout_id,
        status=rollout.status,
        model=rollout.config.model,
        max_turns=rollout.config.max_turns,
        max_seq_len=rollout.config.max_seq_len,
    )


@router.post("/v1/rollouts/{rollout_id}/chat/completions")
async def chat_completions(
    rollout_id: str,
    body: dict[str, Any],
    request: Request,
):
    registry = _get_rollout_registry(request)
    rollout = await registry.get(rollout_id)

    stream = body.get("stream", False)
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise HTTPException(status_code=400, detail="Chat request must include a `messages` array.")
    messages: Messages = [from_raw_message(m) for m in raw_messages]
    tools = body.get("tools")

    async with rollout.lock:
        _validate_rollout_status(rollout, rollout_id)
        turn_index = rollout.turn_count

        logger.info(
            f"rollout={rollout_id} turn={turn_index} request messages={len(messages)} tools={len(tools or [])} stream={stream} status={rollout.status}"
        )

        if tools is not None:
            rollout.vf_state["oai_tools"] = tools

        sampling_args = rollout.config.sampling_params

        try:
            request_mode = "chat_completions"
            if rollout.turn_count == 0:
                rollout.vf_state["prompt"] = messages
            response = await _call_chat_with_messages(
                rollout=rollout,
                raw_messages=raw_messages,
                tools=tools,
                sampling_args=sampling_args,
            )
        except BadRequestError as exc:
            detail = str(exc)
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    detail = message if isinstance(message, str) else str(error)
                else:
                    message = body.get("message")
                    detail = message if isinstance(message, str) else str(body)
            logger.warning(
                f"rollout={rollout_id} turn={turn_index} upstream_bad_request={detail!r}",
            )
            raise HTTPException(status_code=400, detail=detail) from exc

        vf_response = await rollout.vf_client.from_native_response(response)
        completion_messages = await parse_response_message(vf_response)
        response_is_truncated = vf_response.message.is_truncated or False
        tokens = await parse_response_tokens(vf_response, max_seq_len=rollout.config.max_seq_len)
        token_is_truncated = tokens is not None and tokens["is_truncated"]
        step_is_truncated = response_is_truncated or token_is_truncated

        trajectory_step = TrajectoryStep(
            prompt=messages,
            completion=completion_messages,
            response=vf_response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=step_is_truncated,
            trajectory_id=rollout.vf_state["trajectory_id"],
            extras={},
        )
        rollout.vf_state["trajectory"].append(trajectory_step)

        rollout.turn_count += 1
        rollout.is_truncated = rollout.is_truncated or step_is_truncated
        rollout.vf_state["is_truncated"] = rollout.is_truncated
        rollout.vf_state["completion"] = _render_completion(rollout.vf_state)

        if rollout.config.max_turns > 0 and rollout.turn_count >= rollout.config.max_turns:
            rollout.status = "completed"

        finish_reason = response.choices[0].finish_reason if response.choices else None
        assistant_content = response.choices[0].message.content if response.choices else None
        preview = None
        if assistant_content:
            preview = assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
        prompt_token_count = len(tokens["prompt_ids"]) if tokens is not None else None
        completion_token_count = len(tokens["completion_ids"]) if tokens is not None else None
        logger.info(
            f"rollout={rollout_id} turn={turn_index} response mode={request_mode} "
            f"finish={finish_reason} truncated={step_is_truncated} "
            f"prompt_tokens={prompt_token_count} completion_tokens={completion_token_count} "
            f"steps={len(rollout.vf_state['trajectory'])} preview={preview!r}"
        )

        if request.app.state.log_rollout_gateway_turns:
            raw_tool_calls = response.choices[0].message.tool_calls if response.choices else []
            tool_calls = [tc.model_dump(mode="json") for tc in (raw_tool_calls or [])]
            prompt_tool_responses = [
                message for message in messages if isinstance(message, dict) and message.get("role") == "tool"
            ]
            logger.info(f"rollout={rollout_id} turn={turn_index} full_completion={assistant_content}")
            logger.info(f"rollout={rollout_id} turn={turn_index} full_tool_calls={tool_calls}")
            if prompt_tool_responses:
                logger.info(f"rollout={rollout_id} turn={turn_index} prompt_tool_responses={prompt_tool_responses}")

    if stream:
        return StreamingResponse(
            _fake_stream(response),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return JSONResponse(content=response.model_dump())


@router.get("/v1/rollouts/{rollout_id}/trajectory", response_model=TrajectoryResponse)
async def get_trajectory(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = await registry.get(rollout_id)

    async with rollout.lock:
        vf_state = rollout.vf_state
        completion = vf_state.get("completion") or _render_completion(vf_state)
        trajectory = [
            TrajectoryStepResponse.model_validate(step, from_attributes=True) for step in vf_state["trajectory"]
        ]
        logger.info(
            f"rollout={rollout_id} trajectory_fetch status={rollout.status} turns={rollout.turn_count} steps={len(trajectory)} truncated={rollout.is_truncated}"
        )

        return TrajectoryResponse(
            rollout_id=rollout_id,
            status=rollout.status,
            num_turns=rollout.turn_count,
            model=rollout.config.model,
            prompt=vf_state.get("prompt", []),
            completion=completion,
            is_truncated=rollout.is_truncated,
            trajectory=trajectory,
        )


@router.post("/v1/rollouts/{rollout_id}/cancel", response_model=RolloutStatusResponse)
async def cancel_rollout(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = await registry.cancel(rollout_id)
    return RolloutStatusResponse(rollout_id=rollout_id, status=rollout.status)


@router.post("/v1/rollouts/{rollout_id}/unregister", response_model=RolloutStatusResponse)
async def unregister_rollout(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = await registry.unregister(rollout_id)
    return RolloutStatusResponse(rollout_id=rollout_id, status=rollout.status)
