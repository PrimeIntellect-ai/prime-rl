import contextlib
import threading
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI

from prime_rl.configs.shared import ControlConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller


class ControlServer:
    """Serves the control API from a background thread. A no-op without a ``[control]`` config."""

    def __init__(self, controller: Controller, config: ControlConfig | None) -> None:
        self.server = _UvicornServer(_uvicorn_config(create_app(controller), config)) if config is not None else None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.server is None:
            return
        self.thread = threading.Thread(target=self.server.run, name="control", daemon=True)
        self.thread.start()

    def wait(self) -> None:
        """Block while the API serves, which is until the process is stopped."""
        if self.thread is not None:
            self.thread.join()


async def serve(app: FastAPI, config: ControlConfig) -> None:
    """Serve ``app`` on the running event loop, so its handlers share state with an asyncio
    component without crossing threads."""
    await _UvicornServer(_uvicorn_config(app, config)).serve()


class _UvicornServer(uvicorn.Server):
    """Leaves signal handling to the host process."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _uvicorn_config(app: FastAPI, config: ControlConfig) -> uvicorn.Config:
    return uvicorn.Config(app, host=config.host, port=config.port, log_level="warning")
