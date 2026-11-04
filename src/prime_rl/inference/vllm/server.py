import asyncio
import gc
import os
import tempfile
import time
import uuid
from argparse import Namespace
from pathlib import Path

import uvloop
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig

logger = init_logger("vllm.entrypoints.launchers.api_server.entry")

# Create our own router for custom endpoints
router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


API_SERVER_GC_THRESHOLDS = (50_000, 20, 100)

# After the engine core dies, the API server's graceful shutdown can wait forever on requests the
# dead engine never answers (e.g. a weight update). Exit hard so a supervisor can restart it.
ENGINE_DEATH_EXIT_GRACE_S = 30.0


async def exit_when_engine_dies(client: EngineClient) -> None:
    while not client.errored:
        await asyncio.sleep(1.0)
    logger.error(f"The engine core died; exiting in {ENGINE_DEATH_EXIT_GRACE_S:.0f}s unless shutdown completes")
    await asyncio.sleep(ENGINE_DEATH_EXIT_GRACE_S)
    os._exit(1)


# Every API server process of one server reports the same instance, readiness and model state: the
# instance ID is inherited through the environment, and readiness, the policy served, the monitor's
# lease and the marker of an incomplete update are files, since an update, a probe or a resume
# reaches only the API process that took the request.
INSTANCE_ID_ENV = "PRIME_RL_INFERENCE_INSTANCE_ID"


def server_file(instance_id: str, kind: str) -> Path:
    return Path("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()) / f"prime-rl-{instance_id}.{kind}"


def write_server_file(path: Path, value: int | float | str) -> None:
    """Replace the file whole: another API server process may read it at any moment, and an empty
    file would read as no policy required."""
    partial = path.with_name(f"{path.name}.{os.getpid()}")
    partial.write_text(str(value))
    os.replace(partial, path)


def serve_policy(request: Request, version: int | str) -> None:
    """Record the policy this server now holds."""
    write_server_file(request.app.state.policy_file, version)


def read_policy(path: Path) -> int:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return -1


def lease_lapsed(state, now: float) -> bool:
    try:
        return now > float(state.lease_file.read_text())
    except (OSError, ValueError):
        return False


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
    async with request.app.state.weight_update_lock:
        if request.app.state.invalid_file.exists():
            return JSONResponse({"error": "A complete weight update is required before resuming"}, status_code=409)
        await engine_client(request).resume_generation()
    return {"status": "resumed"}


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    client = engine_client(request)
    async with request.app.state.weight_update_lock:
        request.app.state.invalid_file.touch()
        await client.collective_rpc(
            "update_weights_from_path",
            args=(data.get("weight_dir"),),
            kwargs={"abort_path": data["abort_path"]} if data.get("abort_path") else {},
        )
        # Label the tokens sampled from now on (``weight_versions`` spans on /inference/v1/generate).
        await client.update_weight_version(str(data["weight_version"]))
        request.app.state.invalid_file.unlink(missing_ok=True)
        serve_policy(request, data["weight_version"])
    return {"status": "ok"}


@router.post("/update_weights_from_modelexpress")
async def update_weights_from_modelexpress(request: Request):
    data = await request.json()
    client = engine_client(request)
    await client.collective_rpc("update_weights_from_modelexpress", args=(data["version_uid"],))
    await client.update_weight_version(str(data["weight_version"]))
    serve_policy(request, data["weight_version"])
    return {"status": "ok"}


@router.post("/reset_weight_transfer")
async def reset_weight_transfer(request: Request):
    """Drop the NIXL transfer plan with a replaced trainer (its staging buffers are gone)."""
    data = await request.json()
    async with request.app.state.weight_update_lock:
        await engine_client(request).collective_rpc(
            "reset_weight_transfer",
            args=(data["stale_trainer_instances"],),
        )
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
    serve_policy(raw_request, weight_version)
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, policy: int | None = None, lease: float | None = None):
    """Whether this server serves a policy the orchestrator delivered (503 while it awaits one, or once
    its engine core died). A router that health-checks this route skips a replacement server until its
    first weight update; the orchestrator reads the instance ID to tell a replacement from a server
    that came back.

    The orchestrator's inference monitor passes the policy inference serves and a lease. A server
    behind that policy, updated without it while the monitor could not reach it, awaits the next
    update. One whose lease lapsed takes no requests until the monitor reaches it again. Either way it
    stops serving a stale policy by itself, without a call from the orchestrator that could fail."""
    state = request.app.state
    now = time.time()
    if lease is not None:
        write_server_file(state.lease_file, now + lease)
    # Readiness compares two versions that only grow, the policy held and the newest one required, so
    # it holds whatever order a probe and a concurrent update on another API process write them in.
    if policy is not None and read_policy(state.required_file) < policy:
        write_server_file(state.required_file, policy)
    awaiting = read_policy(state.policy_file) < read_policy(state.required_file) or lease_lapsed(state, now)
    engine_dead = engine_client(request).errored
    return JSONResponse(
        {"instance_id": state.instance_id, "awaiting_policy": awaiting, "engine_dead": engine_dead},
        status_code=503 if awaiting or engine_dead else 200,
    )


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
    async with request.app.state.weight_update_lock:
        await engine_client(request).collective_rpc(
            "init_broadcaster",
            args=args,
            kwargs={key: data[key] for key in ("recoverable", "abort_path") if key in data},
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
    state.weight_update_lock = asyncio.Lock()
    state.instance_id = os.environ.setdefault(INSTANCE_ID_ENV, uuid.uuid4().hex)
    state.invalid_file = server_file(state.instance_id, "invalid")
    state.required_file = server_file(state.instance_id, "required")
    state.policy_file = server_file(state.instance_id, "policy")
    state.lease_file = server_file(state.instance_id, "lease")
    state.engine_watchdog = asyncio.create_task(exit_when_engine_dies(engine_client), name="engine-watchdog")

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
    import atexit

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

    instance_id = os.environ[INSTANCE_ID_ENV] = uuid.uuid4().hex
    if config.server.await_policy:
        # Take no requests before the first weight update.
        write_server_file(server_file(instance_id, "required"), 0)
    for kind in ("required", "policy", "lease", "invalid"):
        atexit.register(server_file(instance_id, kind).unlink, missing_ok=True)

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
