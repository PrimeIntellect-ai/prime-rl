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

from zeroband.utils.config import APIMonitorConfig, BaseConfig, FileMonitorConfig, MultiMonitorConfig, SocketMonitorConfig
from zeroband.utils.logger import get_logger


class Monitor(ABC):
    """Base class for logging metrics to a single monitoring type (e.g. file, socket, API)."""

    def __init__(self, config: BaseConfig, task_id: str | None = None):
        self.config = config
        self.lock = threading.Lock()
        self.metadata = {"task_id": task_id}
        self.has_metadata = any(self.metadata.values())
        self.logger = get_logger()
        if not self.has_metadata:
            self.logger.warning("No run metadata found. This is fine for local runs, but unexpected when contributing to a public run.")
        self.logger.debug(f"Initializing {self.__class__.__name__} ({str(self.config).replace(' ', ', ')})")

    def _serialize_metrics(self, metrics: dict[str, Any]) -> str:
        if self.has_metadata:
            metrics.update(self.metadata)
        return json.dumps(metrics)

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None: ...


class FileMonitor(Monitor):
    """Logs to a file. Used for debugging."""

    def __init__(self, config: FileMonitorConfig, task_id: str | None = None):
        super().__init__(config, task_id)
        self.file_path = self.config.path
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with open(self.file_path.as_posix(), "a") as f:
                    f.write(self._serialize_metrics(metrics) + "\n")
                self.logger.debug(f"Logged successfully to {self.file_path}")
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {self.file_path}: {e}")


class SocketMonitor(Monitor):
    """Logs to a Unix socket. Previously called `PrimeMetrics`."""

    def __init__(self, config: SocketMonitorConfig, task_id: str | None = None):
        super().__init__(config, task_id)
        self.socket_path = self.config.path

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(self.socket_path.as_posix())
                    sock.sendall(self._serialize_metrics(metrics).encode())
                self.logger.debug(f"Logged successfully to {self.socket_path}")
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {self.socket_path}: {e}")


class APIMonitor(Monitor):
    """Logs to an API via HTTP. Previously called `HttpMonitor`."""

    def __init__(self, config: APIMonitorConfig, task_id: str | None = None):
        super().__init__(config, task_id)
        self.url = self.config.url
        self.auth_token = self.config.auth_token

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
                    self.logger.debug(f"Logged successfully to server {self.url}")
            except Exception as e:
                self.logger.error(f"Failed to log metrics to {self.url}: {e}")

        asyncio.run(_post_metrics())


class MultiMonitor:
    """
    Log progress, performance, and system metrics to multiple (configurable) outputs.
    """

    def __init__(self, config: MultiMonitorConfig, task_id: str | None = None):
        self.logger = get_logger()
        # Initialize outputs
        self.outputs = []
        if config.file is not None:
            self.outputs.append(FileMonitor(config.file, task_id))
        if config.socket is not None:
            self.outputs.append(SocketMonitor(config.socket, task_id))
        if config.api is not None:
            self.outputs.append(APIMonitor(config.api, task_id))

        self.disabled = len(self.outputs) == 0

        # Start metrics collection thread, if system_log_frequency is greater than 0
        if config.system_log_frequency > 0:
            self.logger.info(f"Starting thread to log system metrics every {config.system_log_frequency}s")
            self._system_log_frequency = config.system_log_frequency
            self._has_gpu = self._set_has_gpu()
            self._thread = None
            self._stop_event = threading.Event()
            self._start_metrics_thread()

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to all outputs."""
        if self.disabled:
            return
        self.logger.info(f"Logging metrics: {metrics}")
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


def setup_monitor(config: MultiMonitorConfig, task_id: str | None = None) -> MultiMonitor:
    """Sets up a monitor to log metrics to multiple specified outputs."""
    get_logger().info(f"Initializing monitor ({config})")
    return MultiMonitor(config, task_id)
