from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.utils import load_aware_call, with_cancellation

from prime_rl.inference.patches import monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode
from prime_rl.inference.vllm.serving_chat_with_tokens import (
    ChatCompletionRequestWithTokens,
    OpenAIServingChatWithTokens,
)

# Monkeypatch PrometheusStatLogger to avoid NotImplementedError for LoRA in DP mode
monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode()

# ruff: noqa
import vllm.entrypoints.openai.api_server

import uvloop
import vllm.envs as envs
from fastapi import Depends, HTTPException, Request
from vllm.config import LogprobsMode, VllmConfig
from starlette.datastructures import State
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    base,
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    load_log_config,
    maybe_register_tokenizer_info_endpoint,
    validate_json_request,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from prime_rl.inference.config import InferenceConfig

logger = init_logger("vllm.entrypoints.openai.api_server")


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we extend the engine args with our custom worker extension
@asynccontextmanager
async def custom_build_async_engine_client(
    args: Namespace,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = args.worker_extension_cls
    engine_args.logprobs_mode = LogprobsMode.PROCESSED_LOGPROBS

    async with build_async_engine_client_from_engine_args(
        engine_args, disable_frontend_multiprocessing=args.disable_frontend_multiprocessing, client_config=client_config
    ) as engine:
        yield engine


async def custom_init_app_state(engine_client: EngineClient, vllm_config: VllmConfig, state: State, args: Namespace):
    await init_app_state(engine_client, vllm_config, state, args)

    # Repeat from init_app_state to have
    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    model_config = vllm_config.model_config

    if envs.VLLM_USE_V1:
        supported_tasks = await engine_client.get_supported_tasks()  # type: ignore
    else:
        supported_tasks = model_config.supported_tasks

    resolved_chat_template = load_chat_template(args.chat_template)

    # Also serve OAI chat completion tokens
    state.openai_serving_chat_with_tokens = (
        OpenAIServingChatWithTokens(
            engine_client,
            model_config,
            state.openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we inject custom routes and build async engine client differently
async def custom_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with custom_build_async_engine_client(args, client_config) as engine_client:
        maybe_register_tokenizer_info_endpoint(args)
        app = build_app(args)

        ### CUSTOM ENDPOINTS ###
        @app.post("/update_weights")
        async def _update_weights(request: Request):
            data = await request.json()
            await engine_client.collective_rpc("update_weights", args=(data.get("weight_dir"),))
            return {"status": "ok"}

        @app.post("/reload_weights")
        async def _reload_weights(request: Request):
            await engine_client.collective_rpc("reload_weights")
            return {"status": "ok"}

        @app.post("/init_broadcaster")
        async def _init_broadcaster(request: Request):
            data = await request.json()
            host = data.get("host")
            port = data.get("port")
            timeout = data.get("timeout")
            # Support both legacy and new field names
            server_rank = data.get("server_rank")
            num_inference_server = data.get("num_inference_server")
            await engine_client.collective_rpc(
                "init_broadcaster",
                args=(host, port, server_rank, num_inference_server, timeout),
            )
            return {"status": "ok"}

        def chat_with_tokens(request: Request) -> Optional[OpenAIServingChatWithTokens]:
            return request.app.state.openai_serving_chat_with_tokens

        @app.post(
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
                return base(raw_request).create_error_response(
                    message="The model does not support Chat Completions API"
                )
            try:
                generator = await handler.create_chat_completion_with_tokens(request, raw_request)
            except Exception as e:
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)

            elif isinstance(generator, ChatCompletionResponse):
                return JSONResponse(content=generator.model_dump())

            return StreamingResponse(content=generator, media_type="text/event-stream")

        vllm_config = await engine_client.get_vllm_config()
        await custom_init_app_state(engine_client, vllm_config, app.state, args)

        # This hack allows us to update lora adapters in-place by skipping the check for already loaded adapters.
        async def do_nothing(*args, **kwargs):
            return None

        app.state.openai_serving_models._check_load_lora_adapter_request = do_nothing

        logger.info("Starting vLLM API server %d on %s", server_index, listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


def custom_run_api_server_worker_proc(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Entrypoint for individual API server worker processes."""
    # Import our module to ensure monkey patches are applied in child processes
    # This is critical because child processes start fresh and re-import modules
    import prime_rl.inference.vllm.server  # noqa: F401

    from vllm.utils import set_process_title, decorate_logs

    # Set process title and add process-specific prefix to stdout and stderr.
    server_index = client_config.get("client_index", 0) if client_config else 0
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    uvloop.run(custom_run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs))


import vllm.entrypoints.openai.api_server
import vllm.entrypoints.cli.serve

# Also monkey patch run_api_server_worker_proc for multi-api-server mode
# This is needed because worker processes spawned by run_multi_api_server
# re-import modules and would otherwise use the original run_server_worker
vllm.entrypoints.openai.api_server.run_server_worker = custom_run_server_worker
vllm.entrypoints.cli.serve.run_api_server_worker_proc = custom_run_api_server_worker_proc


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference we do some config translation (i.e. pass populated namespace
# to `parse_args`) and additional arg validation
def server(config: InferenceConfig, vllm_args: list[str]):
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    assert args is not None
    validate_parsed_serve_args(args)

    # Set the worker extension class based on the broadcast backend
    args.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]

    # Raise error if logprobs_mode is not set to processed_logprobs
    if args.logprobs_mode != "processed_logprobs":
        raise ValueError("logprobs_mode must be 'processed_logprobs' to be compatible with the orchestrator.")

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
