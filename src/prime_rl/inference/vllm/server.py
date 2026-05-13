import asyncio
import json
import os
import re
import time
import traceback
from argparse import Namespace
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

import uvloop
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
from starlette.types import Message, Receive, Scope, Send
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.logger import get_logger

MODEL_TOOL_CALL_PARSER: dict[str, str] = {
    # GLM-4.5
    "zai-org/GLM-4.5": "glm45",
    "zai-org/GLM-4.5-FP8": "glm45",
    "zai-org/GLM-4.5-Base": "glm45",
    "zai-org/GLM-4.5-Air": "glm45",
    "zai-org/GLM-4.5-Air-FP8": "glm45",
    "zai-org/GLM-4.5-Air-Base": "glm45",
    "zai-org/GLM-4.5V": "glm45",
    "zai-org/GLM-4.5V-FP8": "glm45",
    # GLM-4.7
    "zai-org/GLM-4.7": "glm47",
    "zai-org/GLM-4.7-FP8": "glm47",
    "zai-org/GLM-4.7-Flash": "glm47",
    # GLM-5
    "zai-org/GLM-5": "glm47",
    "zai-org/GLM-5-FP8": "glm47",
    # GLM-5.1
    "zai-org/GLM-5.1": "glm47",
    "zai-org/GLM-5.1-FP8": "glm47",
    # MiniMax M2
    "MiniMaxAI/MiniMax-M2": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.1": "minimax_m2",
    "MiniMaxAI/MiniMax-M2.5": "minimax_m2",
    # INTELLECT-3
    "PrimeIntellect/INTELLECT-3": "hermes",
    "PrimeIntellect/INTELLECT-3-FP8": "hermes",
    "PrimeIntellect/INTELLECT-3.1": "hermes",
    # Qwen3 dense
    "Qwen/Qwen3-0.6B": "hermes",
    "Qwen/Qwen3-0.6B-Base": "hermes",
    "Qwen/Qwen3-0.6B-FP8": "hermes",
    "Qwen/Qwen3-1.7B": "hermes",
    "Qwen/Qwen3-1.7B-Base": "hermes",
    "Qwen/Qwen3-1.7B-FP8": "hermes",
    "Qwen/Qwen3-4B": "hermes",
    "Qwen/Qwen3-4B-Base": "hermes",
    "Qwen/Qwen3-4B-FP8": "hermes",
    "Qwen/Qwen3-8B": "hermes",
    "Qwen/Qwen3-8B-Base": "hermes",
    "Qwen/Qwen3-8B-FP8": "hermes",
    "Qwen/Qwen3-14B": "hermes",
    "Qwen/Qwen3-14B-Base": "hermes",
    "Qwen/Qwen3-14B-FP8": "hermes",
    "Qwen/Qwen3-32B": "hermes",
    "Qwen/Qwen3-32B-FP8": "hermes",
    # Qwen3 MoE
    "Qwen/Qwen3-30B-A3B": "hermes",
    "Qwen/Qwen3-30B-A3B-Base": "hermes",
    "Qwen/Qwen3-30B-A3B-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B": "hermes",
    "Qwen/Qwen3-235B-A22B-FP8": "hermes",
    # Qwen3 2507
    "Qwen/Qwen3-4B-Instruct-2507": "hermes",
    "Qwen/Qwen3-4B-Thinking-2507": "hermes",
    "Qwen/Qwen3-4B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-4B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "hermes",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "hermes",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "hermes",
    "Qwen/Qwen3-235B-A22B-Thinking-2507": "hermes",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "hermes",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": "hermes",
    # Qwen3-Next
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Thinking": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8": "hermes",
    # Qwen3-Coder
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "hermes",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "hermes",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "hermes",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "hermes",
    # Qwen3-Coder-Next
    "Qwen/Qwen3-Coder-Next": "hermes",
    "Qwen/Qwen3-Coder-Next-Base": "hermes",
    "Qwen/Qwen3-Coder-Next-FP8": "hermes",
    # Qwen3.5 dense (uses qwen3_coder tool format, not hermes)
    "Qwen/Qwen3.5-0.8B": "qwen3_coder",
    "Qwen/Qwen3.5-0.8B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-2B": "qwen3_coder",
    "Qwen/Qwen3.5-2B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-4B": "qwen3_coder",
    "Qwen/Qwen3.5-4B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-9B": "qwen3_coder",
    "Qwen/Qwen3.5-9B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-27B": "qwen3_coder",
    "Qwen/Qwen3.5-27B-FP8": "qwen3_coder",
    # Qwen3.5 MoE (uses qwen3_coder tool format, not hermes)
    "Qwen/Qwen3.5-35B-A3B": "qwen3_coder",
    "Qwen/Qwen3.5-35B-A3B-Base": "qwen3_coder",
    "Qwen/Qwen3.5-35B-A3B-FP8": "qwen3_coder",
    "Qwen/Qwen3.5-122B-A10B": "qwen3_coder",
    "Qwen/Qwen3.5-122B-A10B-FP8": "qwen3_coder",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_coder",
    "Qwen/Qwen3.5-397B-A17B-FP8": "qwen3_coder",
    # NemotronH
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "qwen3_coder",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "qwen3_coder",
}


def resolve_tool_call_parser(model_name: str, tool_call_parser: str | None) -> str | None:
    """Resolve tool_call_parser from model name if set to "auto"."""
    if tool_call_parser == "auto":
        return MODEL_TOOL_CALL_PARSER.get(model_name)
    return tool_call_parser


logger = get_logger()
from prime_rl.inference.patches import (
    monkey_patch_harmony_stop_token_propagation,
    monkey_patch_load_lora_adapter,
    monkey_patch_tokenize_params_validation,
)
from prime_rl.inference.vllm.serving_chat_with_tokens import (
    ChatCompletionRequestWithTokens,
    OpenAIServingChatWithTokens,
)

# NOTE: Fix harmony stop token propagation for GPT-OSS models
# Upstream issue still open: https://github.com/vllm-project/vllm/issues/22519
monkey_patch_harmony_stop_token_propagation()
# NOTE: Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
# May be removable if we pass load_inplace=True (supported since vLLM 0.18, PR #31326)
monkey_patch_load_lora_adapter()
# NOTE: Monkeypatch TokenizeParams to fix overly conservative validation
# Still needed in vLLM 0.20 — upstream rejects prompt_len > max_model_len - max_tokens
monkey_patch_tokenize_params_validation()

logger = init_logger("vllm.entrypoints.openai.api_server")

# Create our own router for custom endpoints
router = APIRouter()

_CURRENT_LORA_CONTEXT: dict[str, Any] = {
    "name": None,
    "path": None,
    "step": None,
    "updated_at": None,
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r", name, value)
        return None


def _infer_lora_step(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"(?:^|/)step_(\d+)(?:/)?$", path)
    if match is None:
        return None
    return int(match.group(1))


def _dump_nonfinite_request(raw_request: Request, body: bytes, exc: BaseException) -> Path:
    dump_dir = Path(os.getenv("PRIME_RL_NONFINITE_DUMP_DIR", "/tmp/prime-rl-nonfinite-requests"))
    dump_dir.mkdir(parents=True, exist_ok=True)

    body_text = body.decode("utf-8", errors="replace")
    try:
        parsed_body: Any = json.loads(body_text)
    except json.JSONDecodeError:
        parsed_body = None

    safe_path = raw_request.url.path.strip("/").replace("/", "_") or "root"
    dump_path = dump_dir / f"{int(time.time() * 1000)}-{os.getpid()}-{safe_path}.json"
    headers = {k: v for k, v in raw_request.headers.items() if k.lower() != "authorization"}
    payload = {
        "method": raw_request.method,
        "url": str(raw_request.url),
        "path": raw_request.url.path,
        "headers": headers,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "body_json": parsed_body,
        "body_text": None if parsed_body is not None else body_text,
    }
    with dump_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return dump_path


def _is_nonfinite_json_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "Out of range float values are not JSON compliant" in str(exc)


def _safe_json_loads(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _request_headers(scope: Scope) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_key, raw_value in scope.get("headers", []):
        key = raw_key.decode("latin1")
        if key.lower() == "authorization":
            continue
        headers[key] = raw_value.decode("latin1", errors="replace")
    return headers


def _request_dump_dir() -> Path:
    dump_dir = Path(os.getenv("PRIME_RL_REQUEST_DUMP_DIR", "/tmp/prime-rl-request-dumps"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def _path_is_capture_target(path: str) -> bool:
    configured = os.getenv(
        "PRIME_RL_REQUEST_DUMP_PATHS",
        "/v1/chat/completions,/v1/chat/completions/tokens,/v1/generate,/inference/v1/generate",
    )
    return any(path == target.strip() for target in configured.split(",") if target.strip())


def _should_dump_request(scope: Scope) -> bool:
    if not _env_flag("PRIME_RL_DUMP_REQUESTS"):
        return False
    if scope.get("method") != "POST":
        return False
    path = str(scope.get("path", ""))
    if not _path_is_capture_target(path):
        return False
    min_step = _env_int("PRIME_RL_REQUEST_DUMP_MIN_LORA_STEP")
    current_step = _CURRENT_LORA_CONTEXT.get("step")
    if min_step is not None and (current_step is None or int(current_step) < min_step):
        return False
    return True


def _dump_request_record(scope: Scope, body: bytes, *, kind: str, exc: BaseException | None = None) -> Path:
    dump_dir = _request_dump_dir()
    body_json = _safe_json_loads(body)
    record = {
        "kind": kind,
        "time": time.time(),
        "pid": os.getpid(),
        "method": scope.get("method"),
        "path": scope.get("path"),
        "query_string": scope.get("query_string", b"").decode("latin1", errors="replace"),
        "headers": _request_headers(scope),
        "lora": dict(_CURRENT_LORA_CONTEXT),
        "body_json": body_json,
        "body_text": None if body_json is not None else body.decode("utf-8", errors="replace"),
    }
    if exc is not None:
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()

    if kind == "request":
        dump_path = dump_dir / f"requests-{os.getpid()}.jsonl"
        with dump_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str))
            f.write("\n")
        return dump_path

    safe_path = str(scope.get("path", "root")).strip("/").replace("/", "_") or "root"
    dump_path = dump_dir / f"{kind}-{int(time.time() * 1000)}-{os.getpid()}-{safe_path}.json"
    with dump_path.open("w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    return dump_path


class RequestDiagnosticsMiddleware:
    """Dump replayable inference request bodies without consuming them."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        messages: list[Message] = []
        body_parts: list[bytes] = []
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        body = b"".join(body_parts)

        async def replay_receive() -> Message:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        if _should_dump_request(scope):
            dump_path = _dump_request_record(scope, body, kind="request")
            logger.debug("Dumped inference request for replay to %s", dump_path)

        try:
            await self.app(scope, replay_receive, send)
        except Exception as exc:
            if _is_nonfinite_json_error(exc) or _env_flag("PRIME_RL_DUMP_ALL_REQUEST_ERRORS"):
                dump_path = _dump_request_record(scope, body, kind="error", exc=exc)
                logger.exception("Dumped inference request after server error to %s", dump_path)
            raise


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def base(request: Request) -> OpenAIServing:
    return request.app.state.openai_serving_tokenization


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}


def chat_with_tokens(request: Request) -> OpenAIServingChatWithTokens | None:
    return request.app.state.openai_serving_chat_with_tokens


@router.post(
    "/v1/chat/completions/tokens",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def _chat_with_tokens(request: ChatCompletionRequestWithTokens, raw_request: Request):
    body = await raw_request.body()
    try:
        handler = chat_with_tokens(raw_request)
        if handler is None:
            return base(raw_request).create_error_response(message="The model does not support Chat Completions API")
        generator = await handler.create_chat_completion_with_tokens(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)

        elif isinstance(generator, ChatCompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")
    except ValueError as exc:
        if not _is_nonfinite_json_error(exc):
            raise
        dump_path = _dump_nonfinite_request(raw_request, body, exc)
        logger.exception("Dumped request that produced a non-finite JSON response to %s", dump_path)
        raise


@router.post("/pause")
async def pause(request: Request):
    await engine_client(request).pause_generation(mode="keep", clear_cache=False)
    return {"status": "paused"}


@router.post("/resume")
async def resume(request: Request):
    await engine_client(request).resume_generation()
    return {"status": "resumed"}


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    await engine_client(request).collective_rpc("update_weights_from_path", args=(data.get("weight_dir"),))
    return {"status": "ok"}


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request):
    """Wrapper around vLLM's /v1/load_lora_adapter."""
    start_time = time.monotonic()
    lora_path = lora_request.lora_path
    lora_step = _infer_lora_step(lora_path)
    _CURRENT_LORA_CONTEXT.update(
        {
            "name": lora_request.lora_name,
            "path": lora_path,
            "step": lora_step,
            "updated_at": time.time(),
        }
    )
    logger.warning(
        "PrimeRL LoRA adapter load start: name=%s path=%s step=%s",
        lora_request.lora_name,
        lora_path,
        lora_step,
    )
    handler = models(raw_request)
    try:
        response = await handler.load_lora_adapter(lora_request)
    except Exception:
        logger.exception(
            "PrimeRL LoRA adapter load failed: name=%s path=%s step=%s elapsed=%.3fs",
            lora_request.lora_name,
            lora_path,
            lora_step,
            time.monotonic() - start_time,
        )
        raise
    if isinstance(response, ErrorResponse):
        logger.warning(
            "PrimeRL LoRA adapter load returned error: name=%s path=%s step=%s elapsed=%.3fs status=%s message=%s",
            lora_request.lora_name,
            lora_path,
            lora_step,
            time.monotonic() - start_time,
            response.error.code,
            response.error.message,
        )
        return JSONResponse(content=response.model_dump(), status_code=response.error.code)
    if _env_flag("PRIME_RL_RESET_PREFIX_CACHE_AFTER_LORA_LOAD"):
        logger.warning(
            "Resetting prefix cache after LoRA load: name=%s path=%s",
            lora_request.lora_name,
            lora_request.lora_path,
        )
        await engine_client(raw_request).reset_prefix_cache(reset_running_requests=False, reset_connector=False)
    logger.warning(
        "PrimeRL LoRA adapter load done: name=%s path=%s step=%s elapsed=%.3fs",
        lora_request.lora_name,
        lora_path,
        lora_step,
        time.monotonic() - start_time,
    )
    return {"status": "ok"}


@router.get("/liveness")
async def liveness(raw_request: Request):
    """Check that the engine event loop can service a no-op worker RPC."""
    try:
        await asyncio.wait_for(
            engine_client(raw_request).collective_rpc("liveness_probe"),
            timeout=raw_request.app.state.liveness_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return JSONResponse({"status": "engine_unresponsive"}, status_code=503)
    return {"status": "ok"}


@router.post("/init_broadcaster")
async def init_broadcaster(request: Request):
    data = await request.json()
    host = data.get("host")
    port = data.get("port")
    timeout = data.get("timeout")
    rank_offset = data.get("rank_offset")
    inference_world_size = data.get("inference_world_size")
    quantize_in_weight_transfer = data.get("quantize_in_weight_transfer", False)
    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, rank_offset, inference_world_size, timeout, quantize_in_weight_transfer),
    )
    return {"status": "ok"}


async def custom_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple,
):
    """
    Modifies init_app_state:
    1. Call the original init_app_state to set up standard state, including
       vLLM 0.20's ``serving_tokens`` for ``/inference/v1/generate``.
    2. Replace ``serving_chat`` with our ``OpenAIServingChatWithTokens`` wrapper
       so the ``/v1/chat/completions/tokens`` (TITO) endpoint can stream
       token IDs alongside the rendered chat completion.
    3. Replace ``serving_tokens`` with ``PrimeRlServingTokens`` so DP-rank
       routing and ``routed_experts`` export survive the migration off the
       legacy ``/v1/generate`` endpoint.
    """
    await init_app_state(engine_client, state, args, supported_tasks)

    state.reset_prefix_cache_after_update = getattr(args, "reset_prefix_cache_after_update", True)
    state.liveness_timeout_seconds = args.liveness_timeout_seconds

    # TITO: server-side chat templating + token IDs.
    if "generate" in supported_tasks and state.openai_serving_chat is not None:
        original_chat = state.openai_serving_chat
        serving_chat = object.__new__(OpenAIServingChatWithTokens)
        serving_chat.__dict__.update(original_chat.__dict__)
        state.openai_serving_chat = serving_chat
        state.openai_serving_chat_with_tokens = serving_chat
    else:
        state.openai_serving_chat_with_tokens = None

    # Swap in our ServingTokens subclass for /inference/v1/generate so the
    # X-data-parallel-rank header and routed_experts response field — both
    # used by prime-RL's renderer / router-replay paths — keep working.
    if "generate" in supported_tasks and state.serving_tokens is not None:
        from prime_rl.inference.vllm.serving_tokens import PrimeRlServingTokens

        upstream = state.serving_tokens
        prime_serving = object.__new__(PrimeRlServingTokens)
        prime_serving.__dict__.update(upstream.__dict__)
        state.serving_tokens = prime_serving


import vllm.entrypoints.openai.api_server
import vllm.v1.utils
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.v1.utils import run_api_server_worker_proc as _original_run_api_server_worker_proc


def custom_build_app(args: Namespace, supported_tasks: tuple, model_config=None):
    """
    Wrap build_app to include our custom router.
    """
    app = _original_build_app(args, supported_tasks, model_config)
    app.include_router(router)
    if _env_flag("PRIME_RL_DUMP_REQUESTS") or _env_flag("PRIME_RL_DUMP_NONFINITE_REQUESTS"):
        app.middleware_stack = RequestDiagnosticsMiddleware(app.middleware_stack or app.build_middleware_stack())
        logger.warning(
            "Enabled inference request diagnostics dumps at %s (min_lora_step=%s paths=%s)",
            _request_dump_dir(),
            os.getenv("PRIME_RL_REQUEST_DUMP_MIN_LORA_STEP"),
            os.getenv("PRIME_RL_REQUEST_DUMP_PATHS"),
        )
    if _env_flag("PRIME_RL_DUMP_NONFINITE_REQUESTS"):
        dump_dir = os.getenv("PRIME_RL_NONFINITE_DUMP_DIR", "/tmp/prime-rl-nonfinite-requests")
        logger.warning("Enabled /v1/chat/completions/tokens non-finite request dumps at %s", dump_dir)
    return app


def custom_run_api_server_worker_proc(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """
    Re-import our module in child processes so monkey patches (custom routes,
    custom init_app_state) are applied in multi-API-server mode.
    """
    import prime_rl.inference.vllm.server  # noqa: F401

    _original_run_api_server_worker_proc(listen_address, sock, args, client_config, **uvicorn_kwargs)


vllm.entrypoints.openai.api_server.init_app_state = custom_init_app_state
vllm.entrypoints.openai.api_server.build_app = custom_build_app
vllm.v1.utils.run_api_server_worker_proc = custom_run_api_server_worker_proc


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference we do some config translation (i.e. pass populated namespace
# to `parse_args`) and additional arg validation
def server(config: InferenceConfig, vllm_extra: dict[str, Any] | None = None):
    import os

    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.openai.api_server import run_server

    # Signal worker processes to disable LoRA on MoE layers when LoRA targets don't include experts
    if config.lora_target_modules and not any("expert" in m for m in config.lora_target_modules):
        os.environ["PRIME_NO_MOE_LORA"] = "1"

    namespace = config.to_vllm()
    if vllm_extra:
        for key, value in vllm_extra.items():
            setattr(namespace, key, value)

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=namespace)
    assert args is not None
    validate_parsed_serve_args(args)

    args.tool_call_parser = resolve_tool_call_parser(args.model, args.tool_call_parser)
    args.enable_auto_tool_choice = args.tool_call_parser is not None
    if args.tool_call_parser is not None:
        logger.info(f"Using tool_call_parser='{args.tool_call_parser}' for model '{args.model}'")

    # Set the worker extension class based on the broadcast backend
    args.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
