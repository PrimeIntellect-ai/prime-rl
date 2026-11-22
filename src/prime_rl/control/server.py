import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from prime_rl.configs.shared import ADMIN_TOKEN_HEADER, ControlConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller
from prime_rl.utils.admin_token import admin_token_matches, digest_var, expected_admin_digest
from prime_rl.utils.logger import get_logger


class ControlServer:
    """Serves the control API from a background thread. A no-op without a ``[control]`` config."""

    def __init__(self, controller: Controller, config: ControlConfig | None, ckpt_dir: Path | None = None) -> None:
        self.server = (
            _UvicornServer(_uvicorn_config(create_app(controller, ckpt_dir), config)) if config is not None else None
        )
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


def require_admin_token(app: FastAPI, token_var: str) -> None:
    """Refuse every request to ``app`` that lacks the run's admin token, when the process
    has the token or its digest; otherwise leave ``app`` open and warn."""
    expected = expected_admin_digest(token_var)
    if expected is None:
        get_logger().warning(
            f"${token_var} and ${digest_var(token_var)} are not set: the control API accepts unauthenticated requests"
        )
        return

    @app.middleware("http")
    async def check_admin_token(request: Request, call_next):
        if not admin_token_matches(request.headers.get(ADMIN_TOKEN_HEADER), expected):
            return JSONResponse({"detail": "Missing or invalid admin token"}, status_code=status.HTTP_401_UNAUTHORIZED)
        return await call_next(request)


def _uvicorn_config(app: FastAPI, config: ControlConfig) -> uvicorn.Config:
    require_admin_token(app, config.admin_token_var)
    return uvicorn.Config(app, host=config.host, port=config.port, log_level="warning")
