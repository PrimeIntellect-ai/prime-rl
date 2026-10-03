import asyncio
from argparse import Namespace

import uvloop
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig

logger = init_logger("vllm.entrypoints.launchers.api_server.entry")

# Create our own router for custom endpoints
router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
}


@router.post("/pause")
async def pause(request: Request):
    logger.debug("Received /pause request (mode=keep, clear_cache=False)")
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
    session_id = data.get("session_id", "default")
    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, rank_offset, inference_world_size, timeout, session_id),
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
    2. Replace ``serving_tokens`` with ``PrimeRlServingTokens`` so DP-rank
       routing and ``routed_experts`` export survive the migration off the
       legacy ``/v1/generate`` endpoint.
    """
    await init_app_state(engine_client, state, args, supported_tasks)

    state.liveness_timeout_seconds = args.liveness_timeout_seconds

    # Swap in our ServingTokens subclass for /inference/v1/generate so the
    # X-data-parallel-rank header and routed_experts response field — both
    # used by prime-RL's renderer / router-replay paths — keep working.
    if "generate" in supported_tasks and state.serving_tokens is not None:
        from prime_rl.inference.vllm.serving_tokens import PrimeRlServingTokens

        upstream = state.serving_tokens
        prime_serving = object.__new__(PrimeRlServingTokens)
        prime_serving.__dict__.update(upstream.__dict__)
        state.serving_tokens = prime_serving


import vllm.entrypoints.launchers.api_server.entry
import vllm.v1.utils
from vllm.entrypoints.launchers.app import build_app as _original_build_app
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


vllm.entrypoints.launchers.api_server.entry.init_app_state = custom_init_app_state
vllm.entrypoints.launchers.api_server.entry.build_app = custom_build_app
vllm.v1.utils.run_api_server_worker_proc = custom_run_api_server_worker_proc


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference we do some config translation (i.e. pass populated namespace
# to `parse_args`) and additional arg validation
def server(config: InferenceConfig):
    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.launchers.api_server.entry import run_server

    namespace = config.to_namespace()

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=namespace)
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
