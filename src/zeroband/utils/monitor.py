import asyncio
import json
import os
import random
import socket
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import aiohttp
import pandas as pd
import psutil
import pynvml
import wandb
from transformers.tokenization_utils import PreTrainedTokenizer

from zeroband.utils.config import (
    APIMonitorConfig,
    BaseConfig,
    FileMonitorConfig,
    MultiMonitorConfig,
    SocketMonitorConfig,
    WandbMonitorConfig,
)
from zeroband.utils.logger import get_logger
from zeroband.utils.pydantic_config import BaseSettings


class Monitor(ABC):
    """Base class for logging metrics to a single monitoring type (e.g. file, socket, API)."""

    def __init__(self, config: BaseConfig, task_id: str | None = None):
        self.config = config
        self.lock = threading.Lock()
        self.metadata = {"task_id": task_id}
        self.has_metadata = any(self.metadata.values())
        self.logger = get_logger()
        if not self.has_metadata:
            self.logger.warning(
                "No run metadata found. This is fine for local runs, but unexpected when contributing to a public run."
            )
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

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
        self.url = config.url
        self.auth_token = config.auth_token

    def log(self, metrics: dict[str, Any]) -> None:
        """Logs metrics to the server"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
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


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbMonitorConfig,
        tokenizer: PreTrainedTokenizer | None = None,
        task_id: str | None = None,
        run_config: BaseSettings | None = None,
    ):
        super().__init__(config, task_id)
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("DP_RANK", "0")))
        self.config = config
        self.is_master = rank == 0
        if not self.is_master:
            self.logger.warning(f"Skipping WandbMonitor initialization from non-master rank ({rank})")
            return
        self.wandb = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            dir=config.dir,
            config=run_config.model_dump() if run_config else None,
            mode="offline" if config.offline else None,
        )
        
        # Optionally, initialize sample logging attributes
        if config.log_samples:
            assert tokenizer is not None, "Tokenizer is required for sample logging"
            self.tokenizer = tokenizer
            self.last_log_step = -1
            self.samples = []

    def log(self, metrics: dict[str, Any]) -> None:
        if not self.is_master:
            return
        wandb.log(metrics, step=metrics.get("step", None))
    
    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        step: int,
    ) -> None:
        """Log prompt/response samples to W&B table.
        
        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences  
            rewards: List of rewards for each sample
            task_rewards: Optional list of task-specific rewards
            step: Current training step
        """
        if not self.config.log_samples or step % self.config.log_samples.interval != 0:
            # Do not log samples if not enabled or not log interval step
            return
        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_step < step, "Step must be greater than last logged step"
            
        self.logger.debug(f"Logging {self.config.log_samples.num_samples} samples to W&B table at step {step}")
        start_time = time.time()
        batch_size = len(input_tokens)
        num_samples = min(self.config.log_samples.num_samples, batch_size)
            
        # Randomly select and log samples
        indices = random.sample(range(batch_size), num_samples)
        for idx in indices:
            sample = {
                "step": step,
                "sample_idx": idx,
                "num_input_tokens": len(input_tokens[idx]),
                "num_output_tokens": len(output_tokens[idx]),
                "input_tokens": input_tokens[idx],
                "output_tokens": output_tokens[idx],
                "prompt": self.tokenizer.decode(input_tokens[idx]),
                "completion": self.tokenizer.decode(output_tokens[idx]),
                "reward": float(rewards[idx]),
                "advantage": float(advantages[idx]),
            }
            self.samples.append(sample)
            
        # Log to W&B table at configured intervals
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        wandb.log({"completions": table}, step=step)
        self.last_log_step = step
        self.logger.debug(f"Logged {len(indices)} samples to W&B table in {time.time() - start_time:.2f}s")


MonitorType = Literal["file", "socket", "api", "wandb"]


class MultiMonitor:
    """
    Log progress, performance, and system metrics to multiple (configurable) outputs.
    """

    def __init__(
        self,
        config: MultiMonitorConfig,
        task_id: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.logger = get_logger()
        # Initialize outputs
        self.outputs: dict[MonitorType, Monitor] = {}
        self.wandb = None
        if config.file is not None:
            self.outputs["file"] = FileMonitor(config.file, task_id)
        if config.socket is not None:
            self.outputs["socket"] = SocketMonitor(config.socket, task_id)
        if config.api is not None:
            self.outputs["api"] = APIMonitor(config.api, task_id)
        if config.wandb is not None:
            self.wandb = WandbMonitor(config.wandb, tokenizer, task_id, run_config=run_config)
            self.outputs["wandb"] = self.wandb

        self.disabled = len(self.outputs) == 0

        # Start metrics collection thread, if system_log_frequency is greater than 0
        if config.system_log_frequency > 0:
            self.logger.info(f"Starting thread to log system metrics every {config.system_log_frequency}s")
            self._system_log_frequency = config.system_log_frequency
            self._has_gpu = self._set_has_gpu()
            self._thread = None
            self._stop_event = threading.Event()
            self._start_metrics_thread()

    def log(
        self,
        metrics: dict[str, Any],
        wandb_prefix: str | None = None,
        exclude: list[MonitorType] = [],
    ) -> None:
        """Logs metrics to all outputs."""
        if self.disabled:
            return
        self.logger.debug(f"Logging metrics: {metrics}")
        for output_type, output in self.outputs.items():
            if output_type not in exclude:
                if output_type == "wandb" and wandb_prefix is not None:
                    step = metrics.pop("step", None)
                    metrics = {
                        **{f"{wandb_prefix}/{k}": v for k, v in metrics.items()},
                        "step": step,
                    }
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

            self.log(metrics, exclude=["wandb"])
            time.sleep(self._system_log_frequency)

    def __del__(self):
        # Need to check hasattr because __del__ sometime delete attributes before
        if hasattr(self, "_thread") and self._thread is not None:
            self._stop_metrics_thread()


_MONITOR: MultiMonitor | None = None


def get_monitor() -> MultiMonitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("Monitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    config: MultiMonitorConfig,
    task_id: str | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config: BaseSettings | None = None,
) -> MultiMonitor:
    """Sets up a monitor to log metrics to multiple specified outputs."""
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("Monitor already initialized. Please call `setup_monitor` only once.")
    _MONITOR = MultiMonitor(config, task_id, tokenizer, run_config)
    return _MONITOR
