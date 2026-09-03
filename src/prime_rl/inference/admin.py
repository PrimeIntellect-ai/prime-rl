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

    async def update_nccl_weights(
        self,
        weight_dir: Path,
        *,
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None: ...

    async def aclose(self) -> None: ...
