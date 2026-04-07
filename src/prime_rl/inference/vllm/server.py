from argparse import Namespace
from http import HTTPStatus
from typing import Any

import uvloop
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
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
# Still needed in vLLM 0.19 — upstream rejects prompt_len > max_model_len - max_tokens
monkey_patch_tokenize_params_validation()

logger = init_logger("vllm.entrypoints.openai.api_server")

# Create our own router for custom endpoints
router = APIRouter()


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
    await engine_client(request).reset_prefix_cache()
    return {"status": "ok"}


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request):
    """Load a LoRA adapter and reset the prefix cache.

    Wrapper around vLLM's /v1/load_lora_adapter that also resets the prefix cache
    to invalidate KV states computed with old weights.
    """
    handler = models(raw_request)
    response = await handler.load_lora_adapter(lora_request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.error.code)
    # Reset prefix cache to invalidate KV states computed with old weights
    await engine_client(raw_request).reset_prefix_cache()
    return {"status": "ok"}


@router.post("/init_broadcaster")
async def init_broadcaster(request: Request):
    data = await request.json()
    host = data.get("host")
    port = data.get("port")
    timeout = data.get("timeout")
    rank_offset = data.get("rank_offset")
    inference_world_size = data.get("inference_world_size")
    gpus_per_server = data.get("gpus_per_server")
    quantize_in_weight_transfer = data.get("quantize_in_weight_transfer", False)
    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, rank_offset, inference_world_size, gpus_per_server, timeout, quantize_in_weight_transfer),
    )
    return {"status": "ok"}


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
    handler = chat_with_tokens(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Chat Completions API")
    try:
        generator = await handler.create_chat_completion_with_tokens(request, raw_request)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


async def custom_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple,
):
    """
    Modifies init_app_state:
    1. Call the original init_app_state to set up standard state.
    2. Replace the serving_chat with our OpenAIServingChatWithTokens wrapper.
    """
    await init_app_state(engine_client, state, args, supported_tasks)

    if "generate" in supported_tasks and state.openai_serving_chat is not None:
        original_chat = state.openai_serving_chat
        serving_chat = object.__new__(OpenAIServingChatWithTokens)
        serving_chat.__dict__.update(original_chat.__dict__)
        state.openai_serving_chat = serving_chat
        state.openai_serving_chat_with_tokens = serving_chat
    else:
        state.openai_serving_chat_with_tokens = None


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
    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.openai.api_server import run_server

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
