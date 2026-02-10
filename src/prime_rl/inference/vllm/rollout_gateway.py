from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
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
from verifiers.types import Messages, TrajectoryStep, TrajectoryStepTokens
from verifiers.types import State as VfState
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import (
    parse_is_truncated,
    parse_response_messages,
    parse_response_tokens,
)
from verifiers.utils.token_utils import (
    get_prompt_ids,
    prepare_sampling_args_for_token_prompts,
)
from vllm.entrypoints.openai.api_server import router

RolloutStatus = Literal["active", "cancelled", "completed"]


class RegisterRolloutRequest(BaseModel):
    model: str
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    max_turns: int = -1
    max_seq_len: int | None = None


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
            vf_state = _init_vf_state(config.model)
            rollout_state = RolloutState(
                config=config,
                localhost_client=localhost_client,
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


def _init_vf_state(model: str) -> VfState:
    state = VfState()
    state["model"] = model
    state["oai_tools"] = []
    state["trajectory"] = []
    state["trajectory_id"] = ""
    state["prompt"] = []
    state["completion"] = []
    state["is_truncated"] = False
    state["_cached_suffix_ids"] = None
    return state


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
                "Rollout gateway is disabled. Start vLLM with api_server_count=1 "
                "to enable /v1/rollouts endpoints."
            ),
        )
    return cast(RolloutRegistry, registry)


def _require_rollout_state(
    rollout: RolloutState | None,
    rollout_id: str,
) -> RolloutState:
    if rollout is None:
        raise HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")
    return rollout


def _render_completion(vf_state: VfState) -> Messages:
    trajectory = vf_state.get("trajectory", [])
    if not trajectory:
        return []

    last_prompt = trajectory[-1]["prompt"]
    last_completion = trajectory[-1]["completion"]
    full_conversation = concat_messages([last_prompt, last_completion])

    final_env_response = vf_state.get("final_env_response")
    if final_env_response:
        full_conversation = concat_messages([full_conversation, final_env_response])

    rollout_prompt = cast(Messages, vf_state.get("prompt", []))
    return cast(Messages, full_conversation[len(cast(list[dict[str, Any]], rollout_prompt)) :])


def _serialize_tokens(tokens: TrajectoryStepTokens | None) -> dict[str, Any] | None:
    if tokens is None:
        return None
    return {
        "prompt_ids": list(tokens["prompt_ids"]),
        "prompt_mask": list(tokens["prompt_mask"]),
        "completion_ids": list(tokens["completion_ids"]),
        "completion_mask": list(tokens["completion_mask"]),
        "completion_logprobs": list(tokens["completion_logprobs"]),
        "overlong_prompt": bool(tokens["overlong_prompt"]),
        "is_truncated": bool(tokens["is_truncated"]),
    }


def _serialize_trajectory_step(step: TrajectoryStep) -> dict[str, Any]:
    return {
        "prompt": step["prompt"],
        "completion": step["completion"],
        "tokens": _serialize_tokens(step["tokens"]),
        "reward": step["reward"],
        "advantage": step["advantage"],
        "is_truncated": bool(step["is_truncated"]),
        "trajectory_id": step["trajectory_id"],
        "extras": step["extras"],
    }


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
                f"Rollout reached max turns ({max_turns}): {rollout_id}. "
                "Register a new rollout for additional turns."
            ),
        )


async def _call_chat_with_messages(
    rollout: RolloutState,
    messages: Messages,
    tools: list[dict[str, Any]] | None,
    sampling_args: dict[str, Any],
) -> ChatCompletion:
    request_body: dict[str, Any] = {
        "model": rollout.config.model,
        "messages": messages,
        **sampling_args,
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


async def _synthesize_stream(response: ChatCompletion):
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


@router.post("/v1/rollouts/{rollout_id}/register")
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

    return {
        "rollout_id": rollout_id,
        "status": rollout.status,
        "model": rollout.config.model,
        "max_turns": rollout.config.max_turns,
        "max_seq_len": rollout.config.max_seq_len,
    }


@router.post("/v1/rollouts/{rollout_id}/chat/completions")
async def chat_completions(
    rollout_id: str,
    body: dict[str, Any],
    request: Request,
):
    registry = _get_rollout_registry(request)
    rollout = _require_rollout_state(await registry.get(rollout_id), rollout_id)

    stream = bool(body.get("stream", False))
    messages = cast(Messages, body.get("messages"))
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Chat request must include a `messages` array.")
    tools = cast(list[dict[str, Any]] | None, body.get("tools"))

    async with rollout.lock:
        _validate_rollout_accepting_requests(rollout, rollout_id)

        if tools is not None:
            rollout.vf_state["oai_tools"] = tools

        sampling_args = _normalize_sampling_args(rollout.config.sampling_params)
        sampling_args = prepare_sampling_args_for_token_prompts(sampling_args)

        if rollout.turn_count == 0:
            rollout.vf_state["prompt"] = messages
            response = await _call_chat_with_messages(
                rollout=rollout,
                messages=messages,
                tools=tools,
                sampling_args=dict(sampling_args),
            )
        else:
            prompt_ids = await get_prompt_ids(
                state=rollout.vf_state,
                prompt_messages=messages,
                client=rollout.localhost_client,
            )
            response = await _call_chat_with_tokens(
                rollout=rollout,
                messages=messages,
                tools=tools,
                prompt_ids=prompt_ids,
                sampling_args=dict(sampling_args),
            )

        completion_messages = await parse_response_messages(response, "chat")
        response_is_truncated = await parse_is_truncated(response, "chat")
        tokens = await parse_response_tokens(
            response=response,
            message_type="chat",
            max_seq_len=rollout.config.max_seq_len,
        )
        token_is_truncated = tokens is not None and bool(tokens.get("is_truncated"))
        step_is_truncated = response_is_truncated or token_is_truncated

        trajectory_step = TrajectoryStep(
            prompt=messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=step_is_truncated,
            trajectory_id=cast(str, rollout.vf_state["trajectory_id"]),
            extras={},
        )
        rollout.vf_state["trajectory"].append(trajectory_step)

        rollout.turn_count += 1
        rollout.is_truncated = rollout.is_truncated or step_is_truncated
        rollout.vf_state["is_truncated"] = rollout.is_truncated
        rollout.vf_state["completion"] = _render_completion(rollout.vf_state)

        if rollout.config.max_turns > 0 and rollout.turn_count >= rollout.config.max_turns:
            rollout.status = "completed"

    if stream:
        return StreamingResponse(
            _synthesize_stream(response),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return JSONResponse(content=response.model_dump())


@router.get("/v1/rollouts/{rollout_id}/trajectory")
async def get_trajectory(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = _require_rollout_state(await registry.get(rollout_id), rollout_id)

    async with rollout.lock:
        vf_state = rollout.vf_state
        trajectory_payload = [
            _serialize_trajectory_step(cast(TrajectoryStep, step))
            for step in vf_state.get("trajectory", [])
        ]
        completion = vf_state.get("completion") or _render_completion(vf_state)

        return {
            "rollout_id": rollout_id,
            "status": rollout.status,
            "num_turns": rollout.turn_count,
            "model": rollout.config.model,
            "prompt": vf_state.get("prompt", []),
            "completion": completion,
            "is_truncated": rollout.is_truncated,
            "trajectory": trajectory_payload,
        }


@router.post("/v1/rollouts/{rollout_id}/cancel")
async def cancel_rollout(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = _require_rollout_state(await registry.cancel(rollout_id), rollout_id)
    return {
        "rollout_id": rollout_id,
        "status": rollout.status,
    }


@router.post("/v1/rollouts/{rollout_id}/unregister")
async def unregister_rollout(rollout_id: str, request: Request):
    registry = _get_rollout_registry(request)
    rollout = _require_rollout_state(await registry.unregister(rollout_id), rollout_id)
    return {
        "rollout_id": rollout_id,
        "status": rollout.status,
    }
