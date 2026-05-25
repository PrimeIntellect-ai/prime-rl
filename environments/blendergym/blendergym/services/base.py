"""GPU service scaffolding: BaseGPURouter + BaseService.

BaseGPURouter provides a unified lease() async context manager for GPU
resource acquisition/release. BaseService provides common FastAPI app
setup, CLI args, health endpoint, and lifecycle hooks.
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .health import clear_sentinels, report_crash, report_ready

logger = logging.getLogger(__name__)


class BaseGPURouter(ABC):
    """GPU routing base class with lease-based resource management."""

    def __init__(self, gpu_pool: list[int]):
        self.gpu_pool = gpu_pool

    @abstractmethod
    @asynccontextmanager
    async def lease(self) -> AsyncGenerator[int, None]:
        """Acquire a GPU, yield its id, auto-release on exit."""
        yield 0  # pragma: no cover


class BaseService(ABC):
    """Common FastAPI service scaffolding for GPU-backed services."""

    def __init__(self, name: str, gpu_pool: list[int], service_id: str | None = None):
        self.name = name
        self.gpu_pool = gpu_pool
        self.service_id = service_id or name.lower().replace(" ", "_")
        self.app = FastAPI(title=name)
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_event_handler("startup", self._startup)
        self.app.add_event_handler("shutdown", self._shutdown)

    @abstractmethod
    async def on_startup(self) -> None: ...

    @abstractmethod
    async def on_shutdown(self) -> None: ...

    @abstractmethod
    async def health_detail(self) -> dict: ...

    async def health(self):
        detail = await self.health_detail()
        all_ok = detail.pop("_ok", True)
        return {
            "service": self.name,
            "status": "ok" if all_ok else "degraded",
            **detail,
        }

    async def _startup(self):
        clear_sentinels(self.service_id)
        logger.info("%s starting on GPUs %s", self.name, self.gpu_pool)
        try:
            await self.on_startup()
        except Exception:
            tb = traceback.format_exc()
            logger.critical("%s startup FAILED:\n%s", self.name, tb)
            report_crash(self.service_id, tb)
            raise
        report_ready(self.service_id, pid=os.getpid())
        logger.info("%s ready", self.name)

    async def _shutdown(self):
        logger.info("%s shutting down", self.name)
        await self.on_shutdown()

    @classmethod
    def common_cli_args(
        cls,
        parser: argparse.ArgumentParser,
        *,
        default_port: int = 8420,
    ):
        parser.add_argument("--port", type=int, default=default_port)
        parser.add_argument(
            "--gpu-pool",
            type=str,
            default="0,1,2,3,4,5",
            help="Comma-separated GPU ids",
        )
        parser.add_argument("--log-level", default="INFO")

    @staticmethod
    def parse_gpu_pool(s: str) -> list[int]:
        return [int(x) for x in s.split(",")]

    def run(self, port: int, log_config: str | dict | None = None):
        uvicorn.run(self.app, host="0.0.0.0", port=port, log_config=log_config)
