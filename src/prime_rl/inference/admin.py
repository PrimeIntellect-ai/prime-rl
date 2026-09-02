from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from httpx import AsyncClient


class AdminPlane(Protocol):
    """Control plane for a policy inference deployment."""

    clients: list[AsyncClient]

    async def wait_for_ready(self, model_name: str) -> None: ...

    async def initialize_nccl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None: ...

    async def initialize_nixl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        session_id: str,
    ) -> None: ...

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None: ...

    async def load_lora_adapter(self, lora_name: str, lora_path: Path) -> None: ...

    async def aclose(self) -> None: ...
