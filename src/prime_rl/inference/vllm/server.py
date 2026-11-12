import asyncio
import gc
from argparse import Namespace
from typing import Literal

import uvloop
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.common.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.specdec import check_draft_swap, local_draft_dir

logger = init_logger("vllm.entrypoints.launchers.api_server.entry")

# Create our own router for custom endpoints
router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


API_SERVER_GC_THRESHOLDS = (50_000, 20, 100)

WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
    "modelexpress": "prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker",
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
    client = engine_client(request)
    await client.collective_rpc("update_weights_from_path", args=(data.get("weight_dir"),))
    # Label the tokens sampled from now on (``weight_versions`` spans on /inference/v1/generate).
    await client.update_weight_version(str(data["weight_version"]))
    return {"status": "ok"}


@router.post("/update_weights_from_modelexpress")
async def update_weights_from_modelexpress(request: Request):
    data = await request.json()
    client = engine_client(request)
    await client.collective_rpc("update_weights_from_modelexpress", args=(data["version_uid"],))
    await client.update_weight_version(str(data["weight_version"]))
    return {"status": "ok"}


@router.post("/update_draft_weights")
async def update_draft_weights(request: Request):
    """Load a draft checkpoint from disk on every engine. Cached draft KV stays; the target verifies every token."""
    weight_dir = (await request.json())["weight_dir"]
    client = engine_client(request)
    speculative_config = client.vllm_config.speculative_config
    if speculative_config is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This server runs without speculative decoding")
    try:
        draft = local_draft_dir(weight_dir)
        check_draft_swap(local_draft_dir(speculative_config.model, speculative_config.revision), draft)
    except (OSError, ValueError) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    await client.collective_rpc("update_draft_weights", args=(str(draft),))
    return {"status": "ok"}


@router.post("/speculative_decoding")
async def speculative_decoding(request: Request):
    """Set the draft tokens per step on every engine: 0 turns drafting off, null restores the launch count.
    Workers re-record their CUDA graphs for the new count. Pause the engines first so data-parallel ranks
    switch together."""
    num = (await request.json())["num_speculative_tokens"]
    client = engine_client(request)
    speculative_config = client.vllm_config.speculative_config
    if speculative_config is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This server runs without speculative decoding")
    launch = speculative_config.num_speculative_tokens
    num = launch if num is None else num
    if not 0 <= num <= launch:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"num_speculative_tokens must be in [0, {launch}]")
    await client.collective_rpc("set_num_speculative_tokens", args=(num,))
    await client.engine_core.call_utility_all_async("set_scheduled_speculative_tokens", num)
    return {"status": "ok"}


@router.get("/weight_fingerprint/{target}")
async def weight_fingerprint(request: Request, target: Literal["model", "draft"]):
    # vLLM's public collective RPC discards all but the first DP engine's result.
    results = await engine_client(request).engine_core.call_utility_all_async(
        "collective_rpc", "get_weight_fingerprint", None, (), {"target": target}
    )
    return {"results": [worker for engine in results for worker in engine]}


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request, weight_version: str):
    """Wrapper around vLLM's /v1/load_lora_adapter.

    prime-rl reloads a fixed-name adapter with fresh weights every step (the path
    changes per policy version; the name is constant — the model name every client
    sends. vLLM rejects an adapter named like a served base model, so with LoRA on
    the base model is served as ``<model>-base``; see ``InferenceConfig.to_namespace``.)
    vLLM's native loader rejects a same-name reload unless ``load_inplace=True``, so
    we force it here — that makes the worker re-read the new weights during
    ``add_lora``, reusing the existing ``lora_int_id``. vLLM resets the stored
    request's flag to ``False`` before caching it (vllm-project/vllm#41482), so it
    does not force a disk reload on every later generation request.
    """
    handler = models(raw_request)
    lora_request.load_inplace = True
    response = await handler.load_lora_adapter(lora_request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.error.code)
    await engine_client(raw_request).update_weight_version(weight_version)
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
    args = (host, port, rank_offset, inference_world_size, timeout, session_id)
    if "staging_buffer_bytes" in data:
        args += (data["staging_buffer_bytes"], data["staging_buffers_count"])
    await engine_client(request).collective_rpc("init_broadcaster", args=args)
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

    # Under RL load the API server holds millions of live objects, and CPython's
    # default thresholds rerun full collections over them every few seconds.
    # Move everything alive after startup out of GC and collect less often.
    gc.collect()
    gc.freeze()
    gc.set_threshold(*API_SERVER_GC_THRESHOLDS)


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
    import os

    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.launchers.api_server.entry import run_server

    # Signal worker processes to disable LoRA on MoE layers when LoRA targets don't include experts
    lora_target_modules = config.vllm.lora_target_modules
    if lora_target_modules and not any("expert" in m for m in lora_target_modules):
        os.environ["PRIME_NO_MOE_LORA"] = "1"

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
