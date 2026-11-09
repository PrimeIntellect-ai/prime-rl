"""prime-rl's additions to vLLM's OpenAI API server: admin routes and the ``/inference/v1/generate`` handler.

``install_api_server_hooks`` runs from the vLLM general plugin, which vLLM loads in every API server process
before it builds the app, so a stock ``vllm serve`` picks these up.
"""

import asyncio
import gc
import os
from argparse import Namespace

import vllm.entrypoints.launchers.api_server.entry as api_server_entry
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.logger import init_logger

LIVENESS_TIMEOUT_ENV = "PRIME_RL_LIVENESS_TIMEOUT_S"

logger = init_logger("vllm.entrypoints.launchers.api_server.entry")

# Create our own router for custom endpoints
router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


API_SERVER_GC_THRESHOLDS = (50_000, 20, 100)


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


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request, weight_version: str):
    """Wrapper around vLLM's /v1/load_lora_adapter.

    prime-rl reloads a fixed-name adapter with fresh weights every step (the path
    changes per policy version; the name is constant — the model name every client
    sends. vLLM rejects an adapter named like a served base model, so with LoRA on
    the base model is served as ``<model>-base``; see ``InferenceConfig.to_namespace``.)
    vLLM's native loader rejects a same-name reload unless ``load_inplace=True``, so
    we force it here — that makes the worker re-read the new weights during
    ``add_lora``, reusing the existing ``lora_int_id``.

    We then reset the stored request's flag back to ``False``. ``load_inplace`` is
    a sticky field on the ``LoRARequest`` that ``_maybe_get_adapters`` hands to
    every generation request; left ``True`` it would force a disk reload on each
    scheduler step. The orchestrator awaits this endpoint before dispatching
    rollouts for the new version, so the reset always lands before generation.
    The reset runs regardless of success/error: vLLM only stores the adapter on
    success today, but resetting whatever is stored keeps us correct even if a
    future version were to leave a ``load_inplace=True`` request behind on error.
    """
    handler = models(raw_request)
    lora_request.load_inplace = True
    response = await handler.load_lora_adapter(lora_request)
    stored = handler.lora_requests.get(lora_request.lora_name)
    if stored is not None:
        stored.load_inplace = False
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

    state.liveness_timeout_seconds = float(os.environ.get(LIVENESS_TIMEOUT_ENV, "30"))

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


def custom_build_app(args: Namespace, supported_tasks: tuple, model_config=None):
    """Wrap build_app to include our custom router."""
    app = build_app(args, supported_tasks, model_config)
    app.include_router(router)
    return app


def install_api_server_hooks() -> None:
    api_server_entry.init_app_state = custom_init_app_state
    api_server_entry.build_app = custom_build_app
