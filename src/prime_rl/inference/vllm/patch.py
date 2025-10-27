import os
from typing import Any, Callable

from fastapi import Request

from vllm.config import LogprobsMode

# Force V0 engine which supports worker_extension_cls
# V1 engine in vLLM 0.10+ doesn't support worker extensions
os.environ.setdefault("VLLM_USE_V1", "0")


def apply_patches() -> None:
    """Apply monkey patches to vLLM API server for custom endpoints and worker config."""
    import vllm.entrypoints.openai.api_server as api_mod

    # Idempotent: only patch once
    if getattr(api_mod, "_prime_rl_patched", False):
        return

    _orig_build_app: Callable = api_mod.build_app

    async def _update_weights(request: Request):
        engine = api_mod.engine_client(request)
        data = await request.json()
        await engine.collective_rpc("update_weights", args=((data or {}).get("weight_dir"),))
        return {"status": "ok"}

    async def _reload_weights(request: Request):
        engine = api_mod.engine_client(request)
        await engine.collective_rpc("reload_weights")
        return {"status": "ok"}

    def _patched_build_app(args):
        app = _orig_build_app(args)
        app.add_api_route("/update_weights", _update_weights, methods=["POST"])
        app.add_api_route("/reload_weights", _reload_weights, methods=["POST"])
        return app

    api_mod.build_app = _patched_build_app

    _orig_build_engine_client = api_mod.build_async_engine_client_from_engine_args

    def _patched_build_engine_client(engine_args, **kw: Any):
        engine_args.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"
        engine_args.logprobs_mode = LogprobsMode.PROCESSED_LOGPROBS
        return _orig_build_engine_client(engine_args, **kw)

    api_mod.build_async_engine_client_from_engine_args = _patched_build_engine_client
    api_mod._prime_rl_patched = True
