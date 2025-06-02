import asyncio
import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiohttp
import psutil
import pynvml
from pydantic import model_validator
from pydantic_config import BaseConfig

import zeroband.utils.envs as envs
from zeroband.utils.logger import get_logger

# Module logger
logger = get_logger("INFER")


class MonitorConfig(BaseConfig):
    # Whether to log to this monitor
    enable: bool = False


class FileMonitorConfig(MonitorConfig):
    # The file path to log to
    path: str | None = None


class SocketMonitorConfig(MonitorConfig):
    # The socket path to log to
    path: str | None = None


class APIMonitorConfig(MonitorConfig):
    # The API URL to log to
    url: str | None = None

    # The API auth token to use
    auth_token: str | None = None


class MultiMonitorConfig(BaseConfig):
    # All possible monitors (currently only supports one instance per type)
    file: FileMonitorConfig = FileMonitorConfig()
    socket: SocketMonitorConfig = SocketMonitorConfig()
    api: APIMonitorConfig = APIMonitorConfig()

    # Interval in seconds to log system metrics (if 0, no system metrics are logged)
    system_log_frequency: int = 0

    @model_validator(mode="after")
    def assert_valid_frequency(self):
        assert self.system_log_frequency >= 0, "Frequency must be at least 0"
        return self


class Monitor(ABC):
    """Base class for logging metrics to a single monitoring type (e.g. file, socket, API)."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.lock = threading.Lock()
        self.metadata = {
            "task_id": envs.PRIME_TASK_ID,
        }
        self.has_metadata = any(self.metadata.values())
        if not self.has_metadata:
            logger.warning("No run metadata found. This is fine for local runs, but unexpected when contributing to a public run.")
        logger.debug(f"Initializing {self.__class__.__name__} ({str(self.config).replace(' ', ', ')})")

    def _serialize_metrics(self, metrics: dict[str, Any]) -> str:
        if self.has_metadata:
            metrics.update(self.metadata)
        return json.dumps(metrics)

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None: ...


class FileMonitor(Monitor):
    """Logs to a file. Used for debugging."""

    def __init__(self, config: FileMonitorConfig):
        super().__init__(config)
        self.file_path = self.config.path
        assert self.file_path is not None, "File path must be set for FileOutput. Set it as --monitor.file.path."
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with open(self.file_path, "a") as f:
                    f.write(self._serialize_metrics(metrics) + "\n")
                logger.debug(f"Logged successfully to {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to log metrics to {self.file_path}: {e}")


class SocketMonitor(Monitor):
    """Logs to a Unix socket. Previously called `PrimeMetrics`."""

    def __init__(self, config: SocketMonitorConfig):
        super().__init__(config)
        self.socket_path = self.config.path or envs.PRIME_SOCKET_PATH

        # Assert that the socket path is set
        assert self.socket_path is not None, (
            "Socket path must be set for SocketOutput. Set it as --monitor.socket.path or PRIME_SOCKET_PATH."
        )

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(self.socket_path)
                    sock.sendall(self._serialize_metrics(metrics).encode())
                logger.debug(f"Logged successfully to {self.socket_path}")
            except Exception as e:
                logger.error(f"Failed to log metrics to {self.socket_path}: {e}")


class APIMonitor(Monitor):
    """Logs to an API via HTTP. Previously called `HttpMonitor`."""

    def __init__(self, config: APIMonitorConfig):
        super().__init__(config)
        self.url = self.config.url or envs.PRIME_API_URL
        self.auth_token = self.config.auth_token or envs.PRIME_API_AUTH_TOKEN

        # Assert that the URL and auth token are set
        assert self.url is not None, "URL must be set for APIOutput. Set it as --monitor.api.url or PRIME_API_URL."
        assert self.auth_token is not None, (
            "Auth token must be set for APIOutput. Set it as --monitor.api.auth_token or PRIME_API_AUTH_TOKEN."
        )

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to the server"""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.auth_token}"}
        payload = {"metrics": self._serialize_metrics(metrics)}

        async def _post_metrics():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, json=payload, headers=headers) as response:
                        if response is not None:
                            response.raise_for_status()
                    logger.debug(f"Logged successfully to server {self.url}")
            except Exception as e:
                logger.error(f"Failed to log metrics to {self.url}: {e}")

        asyncio.run(_post_metrics())


class MultiMonitor:
    """
    Log progress, performance, and system metrics to multiple (configurable) outputs.
    """

    def __init__(self, config: MultiMonitorConfig):
        # Initialize outputs
        self.outputs = []
        if config.file.enable:
            self.outputs.append(FileMonitor(config.file))
        if config.socket.enable:
            self.outputs.append(SocketMonitor(config.socket))
        if config.api.enable:
            self.outputs.append(APIMonitor(config.api))

        self.disabled = len(self.outputs) == 0
        logger.info(f"Initialized Monitor{' (disabled)' if self.disabled else ''}")

        # Start metrics collection thread, if system_log_frequency is greater than 0
        if config.system_log_frequency > 0:
            logger.info(f"Starting thread to log system metrics every {config.system_log_frequency}s")
            self._system_log_frequency = config.system_log_frequency
            self._has_gpu = self._set_has_gpu()
            self._thread = None
            self._stop_event = threading.Event()
            self._start_metrics_thread()

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to all outputs."""
        if self.disabled:
            return
        logger.info(f"Logging metrics: {metrics}")
        for output in self.outputs:
            output.log(metrics)

    def _set_has_gpu(self) -> bool:
        """Determines if a GPU is available at runtime"""
        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(0)  # Check if at least one GPU exists
            return True
        except pynvml.NVMLError:
            return False

    def _start_metrics_thread(self):
        """Starts the system metrics logging thread"""
        assert self._thread is None, "Metrics thread already started"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._log_system_metrics, daemon=True)
        self._thread.start()

    def _stop_metrics_thread(self):
        """Stops the system metrics logging thread"""
        assert self._thread is not None, "Metrics thread not started"
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _log_system_metrics(self):
        """Loop that periodically logs system metrics."""
        assert self._thread is not None, "Metrics thread not started"
        while not self._stop_event.is_set():
            metrics = {
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent,
                "system/memory_usage": psutil.virtual_memory().used,
                "system/memory_total": psutil.virtual_memory().total,
            }

            if self._has_gpu:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    metrics.update(
                        {
                            f"system/gpu_{i}_memory_used": info.used,
                            f"system/gpu_{i}_memory_total": info.total,
                            f"system/gpu_{i}_utilization": gpu_util.gpu,
                        }
                    )

            self.log(metrics)
            time.sleep(self._system_log_frequency)

    def __del__(self):
        # Need to check hasattr because __del__ sometime delete attributes before
        if hasattr(self, "_thread") and self._thread is not None:
            self._stop_metrics_thread()


def setup_monitor(config: MultiMonitorConfig) -> MultiMonitor:
    """Sets up a monitor to log metrics to multiple specified outputs."""
    return MultiMonitor(config)
