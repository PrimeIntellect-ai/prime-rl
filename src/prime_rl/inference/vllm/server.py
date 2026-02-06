from argparse import Namespace
from http import HTTPStatus

from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.cli.serve import run_api_server_worker_proc
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.openai.protocol import ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.utils import load_aware_call, with_cancellation

from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.utils import load_aware_call, with_cancellation

from prime_rl.inference.patches import (
    monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode,
    monkey_patch_load_lora_adapter,
)
from prime_rl.inference.vllm.serving_chat_with_tokens import (
    ChatCompletionRequestWithTokens,
    OpenAIServingChatWithTokens,
)

# NOTE: Monkeypatch PrometheusStatLogger to avoid NotImplementedError for LoRA in DP mode
monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode()
# NOTE: Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
monkey_patch_load_lora_adapter()

# ruff: noqa
import vllm.entrypoints.openai.api_server

import uvloop
import vllm.envs as envs
from fastapi import Request
from fastapi import Depends, HTTPException, Request
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import (
    router,
    engine_client,
    base,
    init_app_state,
    models,
)
from vllm.entrypoints.openai.protocol import LoadLoRAAdapterRequest
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.inference.config import InferenceConfig, ReplicaBootstrapConfig

logger = init_logger("vllm.entrypoints.openai.api_server")

# Module-level config for replica bootstrap (set by server() before startup)
_replica_bootstrap_config: ReplicaBootstrapConfig | None = None

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


@router.post("/broadcast_weights_to_peer")
async def broadcast_weights_to_peer(request: Request):
    """
    Broadcast model weights to a peer pod via NCCL.

    This endpoint is called by newly scaled pods that started with --load-format dummy.
    They discover this pod via K8s DNS, then request weights via this endpoint.
    The weights are transferred via NCCL (uses InfiniBand if available, falls back to TCP).

    Request body:
        nccl_port: Port for NCCL communication (default: 29600)
    """
    data = await request.json()
    nccl_port = data.get("nccl_port", 29600)
    await engine_client(request).collective_rpc(
        "broadcast_to_peer",
        args=(nccl_port,),
    )
    return {"status": "ok"}


@router.post("/fetch_weights_from_peer")
async def fetch_weights_from_peer(request: Request):
    """
    Fetch model weights from a healthy peer via NCCL.

    This endpoint is called after starting with --load-format dummy to quickly
    receive weights from an existing pod instead of loading from disk.

    Can use either K8s DNS discovery OR direct peer IP:
    - K8s mode: provide headless_service and namespace
    - Direct mode: provide peer_ip

    Request body:
        headless_service: K8s headless service name for peer discovery (optional)
        namespace: K8s namespace (optional)
        peer_ip: Direct peer IP address to fetch from (optional)
        http_port: HTTP port for health checks (default: 8000)
        nccl_port: Port for NCCL communication (default: 29600)

    Returns:
        {"status": "ok", "fetched": true/false}
    """
    data = await request.json()
    headless_service = data.get("headless_service")
    namespace = data.get("namespace")
    peer_ip = data.get("peer_ip")
    http_port = data.get("http_port", 8000)
    nccl_port = data.get("nccl_port", 29600)

    # Must have either peer_ip or headless_service (namespace auto-detected if not provided)
    if not peer_ip and not headless_service:
        return JSONResponse(
            content={"error": "Must provide either peer_ip or headless_service"},
            status_code=400,
        )

    results = await engine_client(request).collective_rpc(
        "fetch_weights_from_peer",
        kwargs={
            "headless_service": headless_service,
            "namespace": namespace,
            "peer_ip": peer_ip,
            "http_port": http_port,
            "nccl_port": nccl_port,
        },
    )
    # All workers should return the same result
    success = results[0] if results else False
    return {"status": "ok", "fetched": success}


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


async def custom_init_app_state(engine_client: EngineClient, state: State, args: Namespace):
    """
    Modifies init_app_state:
    1. Set up the custom OpenAIServingChatWithTokens state.
    2. Monkey-patch to allow updating lora adapters in-place.
    3. Auto-fetch weights from peer if configured.
    """
    # Setup the regular app state first (in-place)
    await init_app_state(engine_client, state, args)

    # Replica bootstrap: fetch weights from existing peer via NCCL
    global _replica_bootstrap_config
    if _replica_bootstrap_config and _replica_bootstrap_config.enabled:
        logger.info("Replica bootstrap enabled, fetching weights from peer...")
        try:
            results = await engine_client.collective_rpc(
                "fetch_weights_from_peer",
                kwargs={
                    "headless_service": _replica_bootstrap_config.headless_service,
                    "namespace": _replica_bootstrap_config.namespace,
                    "peer_ip": _replica_bootstrap_config.peer_ip,
                    "http_port": _replica_bootstrap_config.http_port,
                    "nccl_port": _replica_bootstrap_config.nccl_port,
                },
            )
            success = results[0] if results else False
            if success:
                logger.info("Replica bootstrap complete!")
            else:
                logger.warning("No peer available for bootstrap, model has dummy weights")
        except Exception as e:
            logger.error(f"Replica bootstrap failed: {e}")

    # NOTE: Initialize the custom OpenAIServingChatWithTokens state here
    # TODO: Here, we repeat some calls done in init_app_state to be able to
    # correctly set up the OpenAIServingChatWithTokens state, which is a bit
    # brittle, and could probably be made nicer
    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    supported_tasks = await engine_client.get_supported_tasks()
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


import vllm.entrypoints.openai.api_server
import vllm.entrypoints.cli.serve

# Also monkey patch run_api_server_worker_proc for multi-api-server mode
# This is needed because worker processes spawned by run_multi_api_server
# re-import modules and would otherwise use the original run_server_worker
vllm.entrypoints.openai.api_server.init_app_state = custom_init_app_state
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

    # Store replica bootstrap config for use in init_app_state
    global _replica_bootstrap_config
    _replica_bootstrap_config = config.replica_bootstrap

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
