import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import psutil
import pynvml
from pydantic import model_validator
from pydantic_config import BaseConfig

import zeroband.inference.envs as envs
from zeroband.utils.logger import get_logger

# Module logger
logger = get_logger("INFER")


class MetricsConfig(BaseConfig):
    # Type of metrics to log
    type: Literal["file", "socket", "null"] = "null"

    # Interval in seconds to log system metrics (if 0, no system metrics are logged)
    system_log_frequency: int = 0

    @model_validator(mode="after")
    def assert_valid_frequency(self):
        assert self.system_log_frequency >= 0, "Frequency must be at least 0"
        return self


class Metrics(ABC):
    """
    Abstract class for logging metrics. Periodically collects and logs system
    metrics, including CPU, memory and GPU usage via a background thread, if
    log_system_frequency is greater than 0. All subclasses must implement the
    `log` method which decides where to log a metrics dictionary.
    """

    def __init__(self, metrics_config: MetricsConfig):
        # Start metrics collection thread, if log_system_frequency is greater than 0
        if metrics_config.system_log_frequency > 0:
            self._system_log_frequency = metrics_config.system_log_frequency
            self._has_gpu = self._set_has_gpu()
            self._thread = None
            self._stop_event = threading.Event()
            self._start_metrics_thread()

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None:
        """Logs a dictionary of metrics. Subclasses must implement this method."""
        pass

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
        # need to check hasattr because __del__ sometime delete attributes before
        if hasattr(self, "_thread") and self._thread is not None:
            self._stop_metrics_thread()


class NullMetrics(Metrics):
    """Null class. Does not do anything. Initialized if metrics are disabled."""

    def log(self, _) -> None:
        pass


class FileMetrics(Metrics):
    """Logs to a file. Used for debugging."""

    def __init__(self, metrics_config: MetricsConfig):
        super().__init__(metrics_config)
        self.file_path = Path("outputs/metrics.txt")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_lock = threading.Lock()
        logger.info(f"Initialized FileMetrics (file_path={self.file_path})")

    def log(self, metrics: dict[str, Any]) -> None:
        with self.file_lock:
            logger.debug(f"Logging metrics: {metrics}")
            try:
                with open(self.file_path, "a") as f:
                    msg_buffer = []
                    for key, value in metrics.items():
                        msg_buffer.append(json.dumps({"label": key, "value": value}) + "\n")
                    f.write("".join(msg_buffer))
                logger.debug(f"Logged successfully to {self.file_path}: {metrics}")
            except Exception as e:
                logger.error(f"Failed to log metrics to file: {e}")


class SocketMetrics(Metrics):
    """Logs to a Unix socket. Used in SYNTHETIC-2 to send unverified metrics to protocol worker. Previously called `PrimeMetrics`."""

    DEFAULT_SOCKET_PATH = "/var/run/com.prime.miner/metrics.sock"

    def __init__(self, metrics_config: MetricsConfig):
        """Initializes the SocketMetrics instance."""
        super().__init__(metrics_config)

        # Get socket path from environment variable or use system default
        assert envs.PRIME_TASK_ID is not None, "PRIME_TASK_ID must be set when using SocketMetrics"
        self.task_id = envs.PRIME_TASK_ID
        self.socket_path = envs.PRIME_TASK_BRIDGE_SOCKET or self.DEFAULT_SOCKET_PATH

        logger.info(f"Initialized SocketMetrics (socket_path={self.socket_path})")

    def log(self, metrics: dict[str, Any]) -> None:
        logger.debug(f"Logging metrics to socket: {metrics}")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(self.socket_path)

                msg_buffer = []
                for key, value in metrics.items():
                    msg_buffer.append(json.dumps({"label": key, "value": value, "task_id": self.task_id}))
                sock.sendall(("\n".join(msg_buffer)).encode())
            logger.debug(f"Logged successfully to {self.socket_path}: {metrics}")
        except Exception as e:
            logger.error(f"Logging failed with error: {e}")


def setup_metrics(metrics_config: MetricsConfig) -> Metrics:
    match metrics_config.type:
        case "file":
            return FileMetrics(metrics_config)
        case "socket":
            return SocketMetrics(metrics_config)
        case "null":
            return NullMetrics(metrics_config)
        case _:
            raise ValueError(f"Invalid metrics type: {metrics_config.type}")
