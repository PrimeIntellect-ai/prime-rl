import asyncio
import time
from pathlib import Path

import httpx

from prime_rl.utils.pathing import get_step_path


class InferenceAdmin:
    """Single-endpoint admin client for health, model check, and weight update.

    Implements the VersionObserver contract: on a new step, resolves
    broadcast_dir / step_N itself rather than having callers pass the path.
    """

    def __init__(self, base_url: str, api_key: str | None, broadcast_dir: Path):
        base_url = base_url.rstrip("/").removesuffix("/v1")
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        self.base_url = base_url
        self.broadcast_dir = broadcast_dir
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )

    async def wait_healthy(self, timeout: float = 1800.0, interval: float = 1.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                r = await self.client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(interval)
        raise TimeoutError(f"Inference server {self.base_url} not healthy after {timeout}s")

    async def check_model(self, model_name: str) -> None:
        r = await self.client.get("/v1/models")
        r.raise_for_status()
        models = r.json().get("data", [])
        if not any(m["id"] == model_name for m in models):
            raise ValueError(f"Model '{model_name}' not found on {self.base_url}")

    async def on_new_version(self, step: int) -> None:
        # vLLM's update_weights_from_path passes the string straight to HF's
        # DefaultModelLoader, which first validates as a repo ID. A relative
        # path with multiple slashes trips that check before the local-path
        # fallback. Resolve to absolute here.
        path = get_step_path(self.broadcast_dir, step).resolve().as_posix()
        (await self.client.post("/pause", params={"mode": "keep", "clear_cache": "false"})).raise_for_status()
        try:
            (await self.client.post("/update_weights", json={"weight_dir": path})).raise_for_status()
        finally:
            (await self.client.post("/resume")).raise_for_status()
