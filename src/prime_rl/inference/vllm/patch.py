from typing import Callable

from fastapi import Request

from vllm.config import LogprobsMode


def apply_patches() -> None:
    """Apply monkey patches to vLLM API server for custom endpoints and worker config."""
    import vllm.entrypoints.openai.api_server as api_mod
    from vllm.engine.arg_utils import AsyncEngineArgs

    # Idempotent: only patch once
    if getattr(api_mod, "_prime_rl_patched", False):
        return

    # Patch AsyncEngineArgs.create_engine_config to inject worker_extension_cls
    _orig_create_engine_config = AsyncEngineArgs.create_engine_config

    def _patched_create_engine_config(self, *args, **kwargs):
        # Set worker_extension_cls before creating config
        self.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"
        self.logprobs_mode = LogprobsMode.PROCESSED_LOGPROBS
        return _orig_create_engine_config(self, *args, **kwargs)

    AsyncEngineArgs.create_engine_config = _patched_create_engine_config

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
    api_mod._prime_rl_patched = True
