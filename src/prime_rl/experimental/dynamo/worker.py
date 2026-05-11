"""Dynamo vLLM worker entrypoint with prime-rl admin routes."""

from __future__ import annotations

from typing import Any, Callable

from prime_rl.utils.logger import get_logger

logger = get_logger()


async def _pause_generation(handler, body: dict[str, Any]) -> dict[str, str]:
    mode = body.get("mode", "keep")
    clear_cache = bool(body.get("clear_cache", False))
    await handler.engine_client.pause_generation(mode=mode, clear_cache=clear_cache)
    return {"status": "ok"}


async def _resume_generation(handler, _body: dict[str, Any]) -> dict[str, str]:
    await handler.engine_client.resume_generation()
    return {"status": "ok"}


async def _update_weights(handler, body: dict[str, Any]) -> dict[str, str]:
    weight_dir = body.get("weight_dir")
    await handler.engine_client.pause_generation(mode="keep", clear_cache=False)
    try:
        await handler.engine_client.collective_rpc("update_weights_from_path", args=(weight_dir,))
        reset_prefix_cache = bool(body.get("reset_prefix_cache", True))
        if reset_prefix_cache and hasattr(handler.engine_client, "reset_prefix_cache"):
            await handler.engine_client.reset_prefix_cache()
    finally:
        await handler.engine_client.resume_generation()
    return {"status": "ok"}


async def _init_broadcaster(handler, body: dict[str, Any]) -> dict[str, str]:
    await handler.engine_client.collective_rpc(
        "init_broadcaster",
        args=(
            body.get("host"),
            body.get("port"),
            body.get("rank_offset"),
            body.get("inference_world_size"),
            body.get("timeout"),
            body.get("quantize_in_weight_transfer", False),
        ),
    )
    return {"status": "ok"}


async def _liveness_probe(handler, _body: dict[str, Any]) -> dict[str, str]:
    await handler.engine_client.collective_rpc("liveness_probe")
    return {"status": "ok"}


def _bind(handler, callback: Callable[[Any, dict[str, Any]], Any]):
    async def route(body: dict[str, Any] | None = None):
        return await callback(handler, body or {})

    return route


def patch_dynamo_vllm_worker() -> None:
    """Register prime-rl admin routes on Dynamo's worker system server."""
    from dynamo.vllm.handlers import BaseWorkerHandler

    if getattr(BaseWorkerHandler, "_prime_rl_admin_routes_patched", False):
        return

    original_init = BaseWorkerHandler.__init__

    def patched_init(self, runtime, *args, **kwargs) -> None:
        original_init(self, runtime, *args, **kwargs)
        if getattr(self, "_prime_rl_admin_routes_registered", False):
            return
        runtime.register_engine_route("pause", _bind(self, _pause_generation))
        runtime.register_engine_route("resume", _bind(self, _resume_generation))
        runtime.register_engine_route("update_weights", _bind(self, _update_weights))
        runtime.register_engine_route("init_broadcaster", _bind(self, _init_broadcaster))
        runtime.register_engine_route("liveness", _bind(self, _liveness_probe))
        self._prime_rl_admin_routes_registered = True
        logger.info(
            "Registered prime-rl Dynamo admin routes: "
            "/engine/pause, /engine/resume, /engine/update_weights, /engine/init_broadcaster, "
            "/engine/liveness"
        )

    BaseWorkerHandler.__init__ = patched_init
    BaseWorkerHandler._prime_rl_admin_routes_patched = True


def main() -> None:
    patch_dynamo_vllm_worker()

    from dynamo.vllm.main import main as dynamo_vllm_main

    dynamo_vllm_main()


if __name__ == "__main__":
    main()
