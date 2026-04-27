import asyncio
import time
from pathlib import Path
from typing import Literal

import httpx

from prime_rl.utils.client import NCCL_READY_MARKER
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_step_path

Mode = Literal["filesystem", "nccl"]


class InferenceUnhealthy(RuntimeError):
    """Raised by InferenceAdmin.watch_health when the inference server has
    been unreachable / unhealthy for long enough that we'd rather crash the
    orchestrator than keep blasting failing requests at a dead server."""


class InferenceAdmin:
    """Single-endpoint admin client for health, model check, and weight update.

    Implements the VersionObserver contract: on a new step, resolves
    broadcast_dir / step_N itself rather than having callers pass the path.

    In NCCL mode, touches the `NCCL_READY` marker before /update_weights so the
    trainer's NCCL sender (which polls for it) starts the collective. The
    weight_dir is still forwarded — vLLM dispatches internally between
    filesystem load and NCCL receive based on whether the broadcaster is
    initialized.
    """

    # Health-watcher tunables. ~60s of unhealth → abort. Aggressive enough to
    # surface dead/hung vLLM quickly, lenient enough to ride out a transient
    # reload (vLLM occasionally takes >30s to come back from a pause).
    HEALTH_INTERVAL = 10.0
    HEALTH_MAX_FAILURES = 6
    HEALTH_PROBE_TIMEOUT = 5.0

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        broadcast_dir: Path,
        mode: Mode = "filesystem",
        lora_name: str | None = None,
    ):
        base_url = base_url.rstrip("/").removesuffix("/v1")
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        self.base_url = base_url
        self.broadcast_dir = broadcast_dir
        self.mode = mode
        # When set, on_new_version routes through /load_lora_adapter (hot-swap
        # adapter weights by name) instead of the full-model /update_weights
        # path. NCCL broadcast doesn't support LoRA — caller validates upstream.
        self.lora_name = lora_name
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )
        self.logger = get_logger()

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

    async def watch_health(self) -> None:
        """Periodic /health probe. Raises InferenceUnhealthy after enough
        consecutive failures to crash the orchestrator deterministically;
        otherwise rollouts pile up retrying against a dead/hung server."""
        consecutive = 0
        while True:
            await asyncio.sleep(self.HEALTH_INTERVAL)
            ok = False
            reason = ""
            try:
                r = await self.client.get("/health", timeout=self.HEALTH_PROBE_TIMEOUT)
                ok = r.status_code == 200
                if not ok:
                    reason = f"status {r.status_code}"
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                reason = repr(e)
            if ok:
                consecutive = 0
                continue
            consecutive += 1
            self.logger.warning(
                f"Inference health probe failed ({reason}); consecutive={consecutive}/{self.HEALTH_MAX_FAILURES}"
            )
            if consecutive >= self.HEALTH_MAX_FAILURES:
                raise InferenceUnhealthy(
                    f"Inference server {self.base_url} unhealthy for "
                    f"~{int(consecutive * self.HEALTH_INTERVAL)}s — aborting orchestrator"
                )

    async def check_model(self, model_name: str) -> None:
        r = await self.client.get("/v1/models")
        r.raise_for_status()
        models = r.json().get("data", [])
        if not any(m["id"] == model_name for m in models):
            raise ValueError(f"Model '{model_name}' not found on {self.base_url}")

    async def init_nccl_broadcaster(
        self,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        r = await self.client.post(
            "/init_broadcaster",
            json={
                "host": host,
                "port": port,
                "rank_offset": 0,
                "inference_world_size": inference_world_size,
                "timeout": timeout,
                "quantize_in_weight_transfer": quantize_in_weight_transfer,
            },
        )
        r.raise_for_status()

    async def on_new_version(self, step: int) -> None:
        # vLLM's update_weights_from_path passes the string straight to HF's
        # DefaultModelLoader, which first validates as a repo ID. A relative
        # path with multiple slashes trips that check before the local-path
        # fallback. Resolve to absolute here.
        step_dir = get_step_path(self.broadcast_dir, step)
        path = step_dir.resolve().as_posix()

        if self.lora_name is not None:
            # LoRA hot-swap: no pause/resume — vLLM adds the adapter to its
            # registry and serves it under the requested name on subsequent
            # /v1/chat/completions calls. Re-loading the same name updates the
            # adapter path in place.
            r = await self.client.post(
                "/load_lora_adapter",
                json={"lora_name": self.lora_name, "lora_path": path},
            )
            r.raise_for_status()
            return

        (await self.client.post("/pause", params={"mode": "keep", "clear_cache": "false"})).raise_for_status()
        try:
            if self.mode == "nccl":
                # Trainer's NCCL sender is polling for this marker before it
                # joins the collective. Touch it before /update_weights so the
                # send + receive line up.
                step_dir.mkdir(parents=True, exist_ok=True)
                (step_dir / NCCL_READY_MARKER).touch()
            (await self.client.post("/update_weights", json={"weight_dir": path})).raise_for_status()
        finally:
            (await self.client.post("/resume")).raise_for_status()
