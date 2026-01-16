"""Prometheus metrics server for trainer observability.

Exposes training metrics at /metrics in Prometheus format.
Also exposes /health endpoint for Kubernetes liveness probes.
Runs in a background thread to avoid blocking the training loop.
"""

import threading
import time
from dataclasses import dataclass
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


@dataclass
class RunStats:
    """Statistics for a single run/LoRA adapter."""

    run_id: str
    step: int
    total_tokens: int
    learning_rate: float
    ready: bool


class HealthServer:
    """Lightweight HTTP server exposing /health for Kubernetes liveness probes.

    Can be subclassed to add additional endpoints (e.g., MetricsServer).
    """

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        """Create the request handler class. Override to add endpoints."""

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"ok\n")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        return Handler

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._started:
            logger.warning(f"{self.__class__.__name__} already started")
            return

        self._server = HTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._started = True
        logger.info(f"Health server started at http://{self.host}:{self.port}/health")

    def stop(self) -> None:
        """Stop the server and release the port."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._server = None
            self._thread = None
            self._started = False
            logger.info(f"{self.__class__.__name__} stopped")


class MetricsServer(HealthServer):
    """Prometheus metrics server extending HealthServer with /metrics endpoint.

    Uses an isolated CollectorRegistry to avoid global state pollution.
    Disabled by default - enable by setting `metrics_server` in trainer config.
    """

    def __init__(self, config: "MetricsServerConfig"):
        super().__init__(config.port, config.host)
        self.config = config

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
            # Aggregate run metrics
            self._runs_discovered = Gauge(
                "trainer_runs_discovered", "Number of run folders discovered", registry=self._registry
            )
            self._runs_active = Gauge(
                "trainer_runs_active", "Number of runs with assigned slots", registry=self._registry
            )
            self._runs_ready = Gauge(
                "trainer_runs_ready", "Number of runs ready for gradient updates", registry=self._registry
            )
            self._runs_max = Gauge("trainer_runs_max", "Maximum run capacity", registry=self._registry)
            # Per-run metrics with labels
            self._run_step = Gauge("trainer_run_step", "Training step for run", ["run"], registry=self._registry)
            self._run_tokens = Gauge(
                "trainer_run_tokens", "Total tokens processed by run", ["run"], registry=self._registry
            )
            self._run_learning_rate = Gauge(
                "trainer_run_learning_rate", "Current learning rate for run", ["run"], registry=self._registry
            )
            self._run_ready = Gauge(
                "trainer_run_ready",
                "Whether run is ready for updates (1=ready, 0=not ready)",
                ["run"],
                registry=self._registry,
            )
            # Track known run labels for cleanup
            self._known_runs: set[str] = set()
        else:
            self._registry = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        """Create handler with /metrics and /health endpoints."""
        registry = self._registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self._handle_metrics()
                elif self.path == "/health":
                    self._handle_health()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _handle_metrics(self):
                self.send_response(200)
                if registry is not None:
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(generate_latest(registry))
                else:
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"# prometheus_client not installed\n")

            def _handle_health(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok\n")

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

        self._server = HTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._started = True
        logger.info(f"Metrics server started at http://{self.host}:{self.port}/metrics")
        logger.info(f"Health endpoint available at http://{self.host}:{self.port}/health")

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

    def update_runs(
        self,
        runs_discovered: int,
        runs_max: int,
        run_stats: list[RunStats],
    ) -> None:
        """Update run/LoRA metrics.

        Args:
            runs_discovered: Number of run_* folders found in output directory
            runs_max: Maximum run capacity
            run_stats: List of per-run statistics
        """
        if not PROMETHEUS_AVAILABLE:
            return

        # Update aggregate metrics
        self._runs_discovered.set(runs_discovered)
        self._runs_active.set(len(run_stats))
        self._runs_ready.set(sum(1 for r in run_stats if r.ready))
        self._runs_max.set(runs_max)

        # Track current runs for cleanup
        current_runs = {r.run_id for r in run_stats}

        # Remove metrics for runs that no longer exist
        removed_runs = self._known_runs - current_runs
        for run_id in removed_runs:
            self._run_step.remove(run_id)
            self._run_tokens.remove(run_id)
            self._run_learning_rate.remove(run_id)
            self._run_ready.remove(run_id)

        # Update per-run metrics
        for run in run_stats:
            self._run_step.labels(run=run.run_id).set(run.step)
            self._run_tokens.labels(run=run.run_id).set(run.total_tokens)
            self._run_learning_rate.labels(run=run.run_id).set(run.learning_rate)
            self._run_ready.labels(run=run.run_id).set(1 if run.ready else 0)

        self._known_runs = current_runs
