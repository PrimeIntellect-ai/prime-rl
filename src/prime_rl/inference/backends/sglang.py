from pathlib import Path
from typing import Any

import httpx
import sglang.srt.entrypoints.http_server as shttp
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from sglang.srt.entrypoints.http_server import app, launch_server
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.server_args import prepare_server_args

from prime_rl.inference.backends.base import BaseBackend
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
        result = await shttp._global_state.tokenizer_manager.flush_cache()
        if getattr(result, "success", False):
            return {"status": "ok"}
        raise HTTPException(400, "Cache flush failed")

    launch_server(server_args)


class SGLangBackend(BaseBackend):
    def __init__(self) -> None:
        self.config: InferenceConfig | None = None

    def startup(self, config: InferenceConfig) -> None:
        self.config = config
        server(config, sglang_args=config.get_unknown_args())

    async def _post(self, route: str, body: dict[str, Any]) -> None:
        if self.config is None:
            raise RuntimeError("backend not started")
        host = self.config.server.host or "localhost"
        port = self.config.server.port
        url = f"http://{host}:{port}/{route}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body)
            if resp.status_code == 404:
                return
            resp.raise_for_status()

    async def update_weights(self, path: str) -> None:
        await self._post("update_weights", {"model_path": Path(path).as_posix()})

    async def reload_weights(self) -> None:
        await self._post("reload_weights", {})

    async def flush_cache(self) -> None:
        await self._post("flush_cache", {})
