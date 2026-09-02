from typing import Protocol

from httpx import AsyncClient


class AdminPlane(Protocol):
    """Control plane for a policy inference deployment."""

    clients: list[AsyncClient]

    async def wait_for_ready(self, model_name: str) -> None: ...

    async def aclose(self) -> None: ...
