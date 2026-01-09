"""Prometheus metrics server for trainer observability.

Exposes training metrics at /metrics in Prometheus format.
Runs in a background thread to avoid blocking the training loop.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from prime_rl.utils.config import MetricsServerConfig

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsServer:
    """Prometheus metrics server for trainer observability.

    Uses an isolated CollectorRegistry to avoid global state pollution.
    Disabled by default - enable by setting `metrics_server` in trainer config.
    """

    def __init__(self, config: "MetricsServerConfig"):
        self.config = config
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started = False

        if PROMETHEUS_AVAILABLE:
            self._registry = CollectorRegistry()
            self._step = Gauge("trainer_step", "Current training step", registry=self._registry)
            self._loss = Gauge("trainer_loss", "Current training loss", registry=self._registry)
            self._throughput = Gauge(
                "trainer_throughput_tokens_per_sec", "Training throughput in tokens/sec", registry=self._registry
            )
            self._last_step_ts = Gauge(
                "trainer_last_step_timestamp_seconds", "Unix timestamp of last step", registry=self._registry
            )
            self._grad_norm = Gauge("trainer_grad_norm", "Gradient norm", registry=self._registry)
            self._peak_mem = Gauge("trainer_peak_memory_gib", "Peak GPU memory in GiB", registry=self._registry)
            self._lr = Gauge("trainer_learning_rate", "Current learning rate", registry=self._registry)
            self._mfu = Gauge("trainer_mfu_percent", "Model FLOPS utilization %", registry=self._registry)
            self._entropy = Gauge("trainer_entropy", "Mean entropy", registry=self._registry)
            self._mismatch_kl = Gauge(
                "trainer_mismatch_kl", "KL divergence between trainer and inference model", registry=self._registry
            )
        else:
            self._registry = None

    def _make_handler(self):
        """Create handler class with access to our registry."""
        registry = self._registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    if registry is not None:
                        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                        self.end_headers()
                        self.wfile.write(generate_latest(registry))
                    else:
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"# prometheus_client not installed\n")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        return Handler

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self._started:
            logger.warning("Metrics server already started")
            return

        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not installed. Install with: uv sync --extra metrics")

        self._server = HTTPServer((self.config.host, self.config.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._started = True
        logger.info(f"Metrics server started at http://{self.config.host}:{self.config.port}/metrics")

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None
            self._started = False
            logger.info("Metrics server stopped")

    def update(
        self,
        step: int,
        loss: float,
        throughput: float,
        grad_norm: float,
        peak_memory_gib: float,
        learning_rate: float,
        mfu: float = 0.0,
        entropy: float = 0.0,
        mismatch_kl: float = 0.0,
    ) -> None:
        """Update metrics after a training step."""
        if not PROMETHEUS_AVAILABLE:
            return

        self._step.set(step)
        self._loss.set(loss)
        self._throughput.set(throughput)
        self._grad_norm.set(grad_norm)
        self._peak_mem.set(peak_memory_gib)
        self._lr.set(learning_rate)
        self._mfu.set(mfu)
        self._entropy.set(entropy)
        self._mismatch_kl.set(mismatch_kl)
        self._last_step_ts.set(time.time())
