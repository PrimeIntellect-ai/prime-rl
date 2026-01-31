from argparse import Namespace
from http import HTTPStatus

import uvloop
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.cli.serve import run_api_server_worker_proc
from vllm.entrypoints.logger import RequestLogger
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

from prime_rl.inference.config import InferenceConfig
from prime_rl.inference.patches import (
    monkey_patch_load_lora_adapter,
    monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode,
)
from prime_rl.inference.vllm.serving_chat_with_tokens import (
    ChatCompletionRequestWithTokens,
    OpenAIServingChatWithTokens,
)

# NOTE: Monkeypatch PrometheusStatLogger to avoid NotImplementedError for LoRA in DP mode
monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode()
# NOTE: Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
monkey_patch_load_lora_adapter()

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


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    await engine_client(request).collective_rpc("update_weights", args=(data.get("weight_dir"),))
    # Reset prefix cache to invalidate KV states computed with old weights
    await engine_client(request).reset_prefix_cache()
    return {"status": "ok"}


@router.post("/reload_weights")
async def reload_weights(request: Request):
    await engine_client(request).collective_rpc("reload_weights")
    # Reset prefix cache to invalidate KV states computed with old weights
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
    # Support both legacy and new field names
    server_rank = data.get("server_rank")
    num_inference_server = data.get("num_inference_server")
    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, server_rank, num_inference_server, timeout),
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
    1. Set up the custom OpenAIServingChatWithTokens state.
    2. Monkey-patch to allow updating lora adapters in-place.
    """
    # Setup the regular app state first (in-place)
    await init_app_state(engine_client, state, args, supported_tasks)

    # NOTE: Initialize the custom OpenAIServingChatWithTokens state here
    # TODO: Here, we repeat some calls done in init_app_state to be able to
    # correctly set up the OpenAIServingChatWithTokens state, which is a bit
    # brittle, and could probably be made nicer
    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    resolved_chat_template = load_chat_template(args.chat_template)

    state.openai_serving_chat_with_tokens = (
        OpenAIServingChatWithTokens(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )


def custom_run_api_server_worker_proc(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """
    Modifies run_api_server_worker_proc:
    1. Re-import our module to ensure monkey patches are applied in child processes
    """
    # NOTE: This hack ensures that monkey patches are applied in child processes
    # to make our custom routes work in multi-API-server settings.
    import prime_rl.inference.vllm.server  # noqa: F401

    run_api_server_worker_proc(listen_address, sock, args, client_config, **uvicorn_kwargs)


import vllm.entrypoints.cli.serve
import vllm.entrypoints.openai.api_server
from vllm.entrypoints.openai.api_server import build_app as _original_build_app


def custom_build_app(args: Namespace, supported_tasks: tuple):
    """
    Wrap build_app to include our custom router.
    """
    app = _original_build_app(args, supported_tasks)
    app.include_router(router)
    return app


# Also monkey patch run_api_server_worker_proc for multi-api-server mode
# This is needed because worker processes spawned by run_multi_api_server
# re-import modules and would otherwise use the original run_server_worker
vllm.entrypoints.openai.api_server.init_app_state = custom_init_app_state
vllm.entrypoints.openai.api_server.build_app = custom_build_app
vllm.entrypoints.cli.serve.run_api_server_worker_proc = custom_run_api_server_worker_proc


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference we do some config translation (i.e. pass populated namespace
# to `parse_args`) and additional arg validation
def server(config: InferenceConfig, vllm_args: list[str]):
    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.openai.api_server import run_server

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    assert args is not None
    validate_parsed_serve_args(args)

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
