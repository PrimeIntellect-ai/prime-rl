import sglang.srt.entrypoints.http_server as shttp
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from sglang.srt.entrypoints.http_server import app, launch_server
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.server_args import prepare_server_args

from prime_rl.inference.config import InferenceConfig


def server(config: InferenceConfig, sglang_args: list[str]):
    server_args = prepare_server_args(sglang_args)

    def _remove_route(path: str):
        for r in list(app.router.routes):
            if isinstance(r, APIRoute) and r.path == path and "POST" in r.methods:
                app.router.routes.remove(r)
                break

    async def _update(path: str, request: Request):
        obj = UpdateWeightFromDiskReqInput(model_path=path)
        success, message, _ = await shttp._global_state.tokenizer_manager.update_weights_from_disk(obj, request)
        if not success:
            raise HTTPException(400, message)
        return {"status": "ok"}

    @app.post("/update_weights")
    async def _update_weights(request: Request):
        data = await request.json()
        model_path = data.get("model_path")
        if not model_path:
            raise HTTPException(400, "model_path missing")
        return await _update(model_path, request)

    @app.post("/reload_weights")
    async def _reload_weights(request: Request):
        return await _update(server_args.model_path, request)

    async def _update_from_tensor(request: Request):
        data = await request.json()
        obj = UpdateWeightsFromTensorReqInput(**data)
        success, message = await shttp._global_state.tokenizer_manager.update_weights_from_tensor(obj, request)
        if not success:
            raise HTTPException(400, message)
        return {"status": "ok"}

    _remove_route("/update_weights_from_tensor")
    app.post("/update_weights_from_tensor")(_update_from_tensor)

    _remove_route("/flush_cache")

    @app.post("/flush_cache")
    async def _flush_cache(request: Request):
        """Expose a manual cache flush for orchestrator weight updates.

        The tokenizer manager keeps KV cache state across reloads, so we trigger
        an explicit flush after swapping checkpoints to avoid stale generations.
        """

        result = await shttp._global_state.tokenizer_manager.flush_cache()
        if getattr(result, "success", False):
            return {"status": "ok"}
        raise HTTPException(400, "Cache flush failed")

    launch_server(server_args)


def startup(config: InferenceConfig) -> None:
    """Boot the SGLang HTTP server with the routes we expose to the orchestrator."""

    server(config, sglang_args=config.get_unknown_args())
