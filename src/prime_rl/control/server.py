import contextlib
import threading
from collections.abc import Iterator

import uvicorn

from prime_rl.configs.shared import ControlConfig
from prime_rl.control.app import create_app
from prime_rl.control.controller import Controller


class ControlServer:
    """Serves the control API from a background thread. A no-op without a ``[control]`` config."""

    def __init__(self, controller: Controller, config: ControlConfig | None) -> None:
        self.server = None
        if config is not None:
            uvicorn_config = uvicorn.Config(
                create_app(controller),
                host=config.host,
                port=config.port,
                log_level="warning",
            )
            self.server = _UvicornServer(uvicorn_config)
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


class _UvicornServer(uvicorn.Server):
    """Leaves signal handling to the host process."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield
