from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
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
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from vllm.logger import init_logger

router = APIRouter()

RolloutStatus = Literal["active", "cancelled", "completed"]
logger = init_logger("vllm.entrypoints.openai.rollout_gateway")


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
    prompt: list[dict[str, Any]]
    completion: list[dict[str, Any]]
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
    prompt: list[dict[str, Any]]
    completion: list[dict[str, Any]]
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
    turn_count: int = 0
    status: RolloutStatus = "active"
    is_truncated: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RolloutRegistry:
    def __init__(self, port: int):
        self._port = port
        self._rollouts: dict[str, RolloutState] = {}
        self._lock = asyncio.Lock()

    async def register(self, rollout_id: str, config: RolloutConfig) -> RolloutState:
        async with self._lock:
            if rollout_id in self._rollouts:
                raise ValueError(f"Rollout already registered: {rollout_id}")

            localhost_client = AsyncOpenAI(
                base_url=f"http://localhost:{self._port}/v1",
                api_key="EMPTY",
                max_retries=0,
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
            )
            self._rollouts[rollout_id] = rollout_state
            return rollout_state

    async def get(self, rollout_id: str) -> RolloutState | None:
        async with self._lock:
            return self._rollouts.get(rollout_id)

    async def cancel(self, rollout_id: str) -> RolloutState | None:
        async with self._lock:
            rollout = self._rollouts.get(rollout_id)
            if rollout is None:
                return None
            rollout.status = "cancelled"
            return rollout

    async def unregister(self, rollout_id: str) -> RolloutState | None:
        async with self._lock:
            rollout = self._rollouts.pop(rollout_id, None)
        if rollout is not None:
            await rollout.localhost_client.close()
        return rollout


def _normalize_sampling_args(sampling_params: dict[str, Any]) -> dict[str, Any]:
    args = dict(sampling_params)
    if "max_tokens" in args:
        if args["max_tokens"] is None:
            args.pop("max_tokens")
        elif "max_completion_tokens" not in args:
            args["max_completion_tokens"] = args.pop("max_tokens")
        else:
            args.pop("max_tokens")
    if "max_completion_tokens" in args and args["max_completion_tokens"] is None:
        args.pop("max_completion_tokens")
    return {k: v for k, v in args.items() if v is not None}


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


def _assert_rollout(
    rollout: RolloutState | None,
    rollout_id: str,
) -> None:
    if rollout is None:
        raise HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")


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


def _validate_rollout_accepting_requests(rollout: RolloutState, rollout_id: str) -> None:
    if rollout.status == "cancelled":
        raise HTTPException(status_code=409, detail=f"Rollout cancelled: {rollout_id}")
    if rollout.status == "completed":
        raise HTTPException(status_code=409, detail=f"Rollout completed: {rollout_id}")
    max_turns = rollout.config.max_turns
    if max_turns > 0 and rollout.turn_count >= max_turns:
        rollout.status = "completed"
        raise HTTPException(
            status_code=409,
            detail=(
                f"Rollout reached max turns ({max_turns}): {rollout_id}. Register a new rollout for additional turns."
            ),
        )


@lru_cache(maxsize=1)
def _get_full_turn_log_config() -> tuple[str | None, set[int] | None]:
    rollout_filter = os.getenv("PRIME_ROLLOUT_LOG_ROLLOUT_ID")
    rollout_filter = rollout_filter.strip() if rollout_filter else None
    if rollout_filter == "":
        rollout_filter = None

    turns_raw = os.getenv("PRIME_ROLLOUT_LOG_FULL_TURNS", "").strip()
    if not turns_raw:
        return rollout_filter, None
    if turns_raw.lower() in {"all", "*"}:
        return rollout_filter, set()

    turns: set[int] = set()
    for part in turns_raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            turns.add(int(token))
        except ValueError:
            logger.warning(
                "Ignoring invalid PRIME_ROLLOUT_LOG_FULL_TURNS token: %r",
                token,
            )
    return rollout_filter, turns if turns else None


def _should_log_full_turn(rollout_id: str, turn_index: int) -> bool:
    rollout_filter, turns = _get_full_turn_log_config()
    if turns is None:
        return False
    if rollout_filter is not None and rollout_filter != rollout_id:
        return False
    # empty set is sentinel for "all turns"
    return not turns or turn_index in turns


def _serialize_tool_calls(tool_calls: list[ChatCompletionToolParam] | None) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if hasattr(tool_call, "model_dump"):
            serialized.append(cast(dict[str, Any], tool_call.model_dump(mode="json")))
        elif isinstance(tool_call, dict):
            serialized.append(tool_call)
    return serialized


async def _call_chat_with_messages(
    rollout: RolloutState,
    messages: Messages,
    tools: list[dict[str, Any]] | None,
    sampling_args: dict[str, Any],
) -> ChatCompletion:
    extra_body = sampling_args.pop("extra_body", {})
    request_body: dict[str, Any] = {
        "model": rollout.config.model,
        "messages": messages,
        **sampling_args,
        **extra_body,
    }
    if tools:
        request_body["tools"] = tools

    return await rollout.localhost_client.post(
        "/chat/completions",
        body=request_body,
        cast_to=ChatCompletion,
    )


async def _call_chat_with_tokens(
    rollout: RolloutState,
    messages: Messages,
    tools: list[dict[str, Any]] | None,
    prompt_ids: list[int],
    sampling_args: dict[str, Any],
) -> ChatCompletion:
    extra_body = sampling_args.pop("extra_body", {})
    request_body: dict[str, Any] = {
        "model": rollout.config.model,
        "messages": messages,
        "tokens": prompt_ids,
        **sampling_args,
        **extra_body,
    }
    if tools:
        request_body["tools"] = tools

    return await rollout.localhost_client.post(
        "/chat/completions/tokens",
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
    rollout_config = RolloutConfig(
        model=body.model,
        sampling_params=dict(body.sampling_params),
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
    _assert_rollout(rollout, rollout_id)

    stream = body.get("stream", False)
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Chat request must include a `messages` array.")
    tools = body.get("tools")

    async with rollout.lock:
        _validate_rollout_accepting_requests(rollout, rollout_id)
        turn_index = rollout.turn_count

        logger.info(
            "rollout=%s turn=%d request messages=%d tools=%d stream=%s status=%s",
            rollout_id,
            turn_index,
            len(messages),
            len(tools or []),
            stream,
            rollout.status,
        )

        if tools is not None:
            rollout.vf_state["oai_tools"] = tools

        sampling_args = _normalize_sampling_args(rollout.config.sampling_params)
        sampling_args["logprobs"] = True
        extra_body = sampling_args.setdefault("extra_body", {})
        extra_body["return_token_ids"] = True

        try:
            if rollout.turn_count == 0:
                request_mode = "chat_completions"
                rollout.vf_state["prompt"] = messages
                response = await _call_chat_with_messages(
                    rollout=rollout,
                    messages=messages,
                    tools=tools,
                    sampling_args=sampling_args,
                )
            else:
                # Agents like OpenCode may compact (summarize) old messages or prune
                # tool outputs mid-conversation, breaking the prefix assumption that
                # the token path relies on.  Fall back to regular chat completions
                # when that happens.
                prev_step = rollout.vf_state["trajectory"][-1]
                prev_context = concat_messages([prev_step["prompt"], prev_step["completion"]])
                is_prefix_extension = (
                    len(messages) > len(prev_context) and messages[: len(prev_context)] == prev_context
                )
                if is_prefix_extension:
                    request_mode = "chat_completions_tokens"
                    prompt_ids = await rollout.vf_client.get_prompt_ids(
                        state=rollout.vf_state,
                        prompt_messages=messages,
                        oai_tools=rollout.vf_state.get("oai_tools"),
                    )
                    logger.info(f"rollout={rollout_id} turn={turn_index} prompt_ids_len={len(prompt_ids)}")
                    response = await _call_chat_with_tokens(
                        rollout=rollout,
                        messages=messages,
                        tools=tools,
                        prompt_ids=prompt_ids,
                        sampling_args=sampling_args,
                    )
                else:
                    request_mode = "chat_completions"
                    response = await _call_chat_with_messages(
                        rollout=rollout,
                        messages=messages,
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

        if _should_log_full_turn(rollout_id, turn_index):
            tool_calls = _serialize_tool_calls(response.choices[0].message.tool_calls) if response.choices else []
            prompt_tool_responses = [
                message for message in messages if isinstance(message, dict) and message.get("role") == "tool"
            ]
            logger.info(
                "rollout=%s turn=%d full_completion=%s",
                rollout_id,
                turn_index,
                assistant_content,
            )
            logger.info(
                "rollout=%s turn=%d full_tool_calls=%s",
                rollout_id,
                turn_index,
                tool_calls,
            )
            if prompt_tool_responses:
                logger.info(
                    "rollout=%s turn=%d prompt_tool_responses=%s",
                    rollout_id,
                    turn_index,
                    prompt_tool_responses,
                )

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
    _assert_rollout(rollout, rollout_id)

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
    _assert_rollout(rollout, rollout_id)
    return RolloutStatusResponse(rollout_id=rollout_id, status=rollout.status)


@router.post("/v1/rollouts/{rollout_id}/unregister", response_model=RolloutStatusResponse)
async def unregister_rollout(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = await registry.unregister(rollout_id)
    _assert_rollout(rollout, rollout_id)
    return RolloutStatusResponse(rollout_id=rollout_id, status=rollout.status)
