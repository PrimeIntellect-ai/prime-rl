import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import psutil
import pynvml
from pydantic import field_validator, model_validator
from pydantic_config import BaseConfig

import zeroband.inference.envs as envs
from zeroband.utils.logger import get_logger

# Module logger
logger = get_logger("INFER")


class OutputConfig(BaseConfig):
    # Whether to log to this output
    enable: bool = False

    # The task ID the metrics belong to
    task_id: str | None = None

    @field_validator("task_id", mode="before")
    @classmethod
    def get_task_id_from_env(cls, v):
        if v is None:
            return envs.PRIME_TASK_ID
        return v


class FileOutputConfig(OutputConfig):
    # The file path to log to
    path: str = "outputs/metrics.txt"


class SocketOutputConfig(OutputConfig):
    # The socket path to log to
    path: str = "/var/run/com.prime.miner/metrics.sock"


class MonitorConfig(BaseConfig):
    # List of possible outputs to log to
    file: FileOutputConfig = FileOutputConfig()
    socket: SocketOutputConfig = SocketOutputConfig()

    # Interval in seconds to log system metrics (if 0, no system metrics are logged)
    system_log_frequency: int = 0

    @model_validator(mode="after")
    def assert_valid_frequency(self):
        assert self.system_log_frequency >= 0, "Frequency must be at least 0"
        return self


class Output(ABC):
    """Base class for logging metrics to a single output."""

    def __init__(self, config: OutputConfig):
        self.config = config
        self.lock = threading.Lock()
        if not self.config.task_id:
            logger.warning("Task ID it not set. Set it in the config or as PRIME_TASK_ID.")
        logger.info(f"Initialized {self.__class__.__name__} ({str(self.config).replace(' ', ', ')})")

    def _add_task_id(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.config.task_id:
            metrics["task_id"] = self.config.task_id
        return metrics

    @abstractmethod
    def log(self, metrics: dict[str, Any]) -> None:
        pass


class FileOutput(Output):
    """Logs to a file. Used for debugging."""

    def __init__(self, config: FileOutputConfig):
        super().__init__(config)
        self.file_path = Path(config.path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with open(self.file_path, "a") as f:
                    f.write(json.dumps(self._add_task_id(metrics)) + "\n")
                logger.debug(f"Logged successfully to {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to log metrics to {self.file_path}: {e}")


class SocketOutput(Output):
    """Logs to a Unix socket. Previously called `PrimeMetrics`."""

    def __init__(self, config: SocketOutputConfig):
        super().__init__(config)
        self.socket_path = config.path

    def log(self, metrics: dict[str, Any]) -> None:
        with self.lock:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(self.socket_path)
                    msg_buffer = []
                    for key, value in metrics.items():
                        msg_buffer.append(json.dumps(self._add_task_id({"label": key, "value": value})))
                    sock.sendall(("\n".join(msg_buffer)).encode())
                logger.debug(f"Logged successfully to {self.socket_path}")
            except Exception as e:
                logger.error(f"Failed to log metrics to {self.socket_path}: {e}")


class Monitor:
    """
    Log progress, performance, and system metrics to multiple (configurable) outputs.
    """

    def __init__(self, config: MonitorConfig):
        logger.info(f"Initialized monitor ({str(config).replace(' ', ',')})")

        # Initialize outputs
        self.outputs = []
        if config.file.enable:
            self.outputs.append(FileOutput(config.file))
        if config.socket.enable:
            self.outputs.append(SocketOutput(config.socket))

        # Start metrics collection thread, if system_log_frequency is greater than 0
        if config.system_log_frequency > 0:
            logger.debug(f"Starting system metrics logging thread with frequency {config.system_log_frequency}s")
            self._system_log_frequency = config.system_log_frequency
            self._has_gpu = self._set_has_gpu()
            self._thread = None
            self._stop_event = threading.Event()
            self._start_metrics_thread()

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to all outputs."""
        logger.debug(f"Logging metrics: {metrics}")
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


def setup_monitor(config: MonitorConfig) -> Monitor:
    """Sets up a monitor to log metrics to the specified outputs."""
    return Monitor(config)
