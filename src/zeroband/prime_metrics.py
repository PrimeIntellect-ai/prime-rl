import socket
import platform
import json
import os
from typing import Any
import psutil
import threading
import time
import pynvml


class PrimeMetric:
    """
    A class to log metrics to Prime Miner via Unix socket.

    Periodically collects and logs system metrics including CPU, memory and GPU usage.

    Args:
        disable (bool): If True, disables metric logging. Defaults to False.
        period (int): Collection interval in seconds. Defaults to 5.

    Usage:
        metrics = PrimeMetric()
        metrics.log_prime({"custom_metric": value})
    """

    def __init__(self, disable: bool = False, period: int = 5):
        self.disable = disable
        self.period = period
        self._thread = None
        self.has_gpu = False

        if self.disable:
            return
        self._stop_event = threading.Event()
        self._start_metrics_thread()

        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(0)  # Check if at least one GPU exists
            self.has_gpu = True
        except pynvml.NVMLError:
            pass

    ## public

    def log_prime(self, metric: dict[str, Any]):
        if self.disable:
            return
        if not (self._send_message_prime(metric)):
            print(f"Prime logging failed: {metric}")

    ## private

    @classmethod
    def _get_default_socket_path(cls) -> str:
        """Returns the default socket path based on the operating system."""
        default = "/tmp/com.prime.miner/metrics.sock" if platform.system() == "Darwin" else "/var/run/com.prime.miner/metrics.sock"
        return os.getenv("PRIME_TASK_BRIDGE_SOCKET", default=default)

    def _send_message_prime(self, metric: dict, socket_path: str = None) -> bool:
        """Sends a message to the specified socket path or uses the default if none is provided."""
        socket_path = socket_path or os.getenv("PRIME_TASK_BRIDGE_SOCKET", self._get_default_socket_path())
        # print("Sending message to socket: ", socket_path)

        task_id = os.getenv("PRIME_TASK_ID", None)
        if task_id is None:
            print("No task ID found, skipping logging to Prime")
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(socket_path)

                msg_buffer = []
                for key, value in metric.items():
                    msg_buffer.append(json.dumps({"label": key, "value": value, "task_id": task_id}))
                sock.sendall(("\n".join(msg_buffer)).encode())
            return True
        except Exception as e:
            print(f"Error sending message to Prime: {e}")
            print(f"Socket path: {socket_path}")
            return False

    ### background system metrics

    def _start_metrics_thread(self):
        """Starts the metrics collection thread"""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_metrics)
        self._thread.daemon = True
        self._thread.start()

    def _stop_metrics_thread(self):
        """Stops the metrics collection thread"""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _collect_metrics(self):
        while not self._stop_event.is_set():
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_usage": psutil.virtual_memory().used,
                "memory_total": psutil.virtual_memory().total,
            }

            if self.has_gpu:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    metrics.update(
                        {
                            f"gpu_{i}_memory_used": info.used,
                            f"gpu_{i}_memory_total": info.total,
                            f"gpu_{i}_utilization": gpu_util.gpu,
                        }
                    )

            self.log_prime(metrics)
            time.sleep(self.period)

    def __del__(self):
        if hasattr(self, "_thread") and self._thread is not None:
            # need to check hasattr because __del__ sometine delete attributes befores
            self._stop_metrics_thread()
