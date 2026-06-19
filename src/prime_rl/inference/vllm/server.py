import asyncio
from argparse import Namespace
from typing import Any

import uvloop
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.logger import get_logger

logger = get_logger()
from prime_rl.inference.patches import (
    monkey_patch_dp_coordinator_startup_timeout,
    monkey_patch_harmony_stop_token_propagation,
    monkey_patch_load_lora_adapter,
    monkey_patch_nano_v3_reasoning_parser,
    monkey_patch_strip_routed_experts_from_chat,
    monkey_patch_tokenize_params_validation,
    monkey_patch_vllm_padded_input_scrub,
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
# NOTE: Register Nano V3 reasoning parser so configs can use
# `reasoning_parser = "nano_v3"` without a vLLM plugin file.
monkey_patch_nano_v3_reasoning_parser()
# NOTE: Optional mitigation for vLLM padded decode inputs until the native fix
# is available in our pinned runtime.
monkey_patch_vllm_padded_input_scrub()
# NOTE: routed_experts are consumed only via the serialized /generate path (router
# replay). The chat-completions path encodes them as a base64 np.save string the PD
# router cannot merge, which fails eval rollouts (they use chat completions). Strip
# routed_experts from chat responses since the server-wide enable flag has no
# per-request toggle.
monkey_patch_strip_routed_experts_from_chat()
# NOTE: vLLM hard-codes a 30s DP coordinator startup timeout, which the rank-0
# API server blows through when all engine-core ranks on the node are loading
# weights concurrently (multi-node disaggregated deployments).
monkey_patch_dp_coordinator_startup_timeout()

logger = init_logger("vllm.entrypoints.openai.api_server")

# Create our own router for custom endpoints
router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
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


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request):
    """Wrapper around vLLM's /v1/load_lora_adapter."""
    handler = models(raw_request)
    response = await handler.load_lora_adapter(lora_request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.error.code)
    return {"status": "ok"}


def _lora_update_tasks(request: Request) -> dict[int, asyncio.Task]:
    tasks = getattr(request.app.state, "lora_update_tasks", None)
    if tasks is None:
        tasks = {}
        request.app.state.lora_update_tasks = tasks
    return tasks


def _lora_update_adapters(request: Request) -> dict[int, list[dict]]:
    adapters = getattr(request.app.state, "lora_update_adapters", None)
    if adapters is None:
        adapters = {}
        request.app.state.lora_update_adapters = adapters
    return adapters


def _registered_lora_update_steps(request: Request) -> set[int]:
    steps = getattr(request.app.state, "registered_lora_update_steps", None)
    if steps is None:
        steps = set()
        request.app.state.registered_lora_update_steps = steps
    return steps


def _normalize_lora_update_adapter(adapter: dict) -> dict:
    adapter = dict(adapter)
    if "lora_name" not in adapter and "name" in adapter:
        adapter["lora_name"] = adapter["name"]
    if "lora_name" not in adapter:
        raise ValueError("LoRA update adapter is missing 'lora_name'")
    adapter.setdefault("lora_int_id", 1)
    return adapter


def _register_lora_update_success(request: Request, step: int) -> None:
    registered_steps = _registered_lora_update_steps(request)
    if step in registered_steps:
        return

    adapters = _lora_update_adapters(request).get(step)
    if not adapters:
        raise RuntimeError(f"No LoRA adapter metadata stored for step {step}")
    adapter = adapters[0]
    lora_name = adapter["lora_name"]
    # Sentinel path: the adapter was committed in-memory by the workers and must stay resident in
    # their adapter caches. If vLLM ever evicts it and tries to reload from this path it fails
    # loudly -- correct, since there is nothing on disk to reload.
    lora_path = adapter.get("lora_path") or f"/__prime_rl_nccl_lora__/{step}/{lora_name}"
    models(request).lora_requests[lora_name] = LoRARequest(
        lora_name=lora_name,
        lora_int_id=int(adapter["lora_int_id"]),
        lora_path=lora_path,
    )
    registered_steps.add(step)


@router.post("/update_lora")
async def update_lora(request: Request):
    data = await request.json()
    step = data["step"]
    try:
        adapters = [_normalize_lora_update_adapter(adapter) for adapter in data.get("adapters", [])]
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    if len(adapters) != 1:
        return JSONResponse(
            {"status": "error", "message": "NCCL LoRA currently supports exactly one adapter per update"},
            status_code=400,
        )

    header_expectation = {
        "step": step,
        "adapters": [
            {
                "lora_name": adapters[0]["lora_name"],
                "lora_int_id": adapters[0]["lora_int_id"],
            }
        ],
    }
    await engine_client(request).collective_rpc("arm_lora_receive", args=(step, header_expectation))
    task = asyncio.create_task(engine_client(request).collective_rpc("wait_lora_receive", args=(step,)))
    _lora_update_tasks(request)[step] = task
    _lora_update_adapters(request)[step] = adapters
    return JSONResponse({"status": "receiving", "step": step}, status_code=202)


@router.get("/update_lora/status")
async def update_lora_status(request: Request, step: int):
    task = _lora_update_tasks(request).get(step)
    if task is None:
        return JSONResponse({"status": "unknown", "step": step}, status_code=404)
    if not task.done():
        return {"status": "receiving", "step": step}
    try:
        await task
    except Exception as exc:
        return JSONResponse({"status": "error", "step": step, "message": repr(exc)}, status_code=500)
    _register_lora_update_success(request, step)
    return {"status": "ok", "step": step}


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
    2. Replace ``serving_tokens`` with ``PrimeRlServingTokens`` so DP-rank
       routing and ``routed_experts`` export survive the migration off the
       legacy ``/v1/generate`` endpoint.
    """
    await init_app_state(engine_client, state, args, supported_tasks)

    state.reset_prefix_cache_after_update = getattr(args, "reset_prefix_cache_after_update", True)
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
