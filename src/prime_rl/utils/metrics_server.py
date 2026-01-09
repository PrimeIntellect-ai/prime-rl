"""Prometheus metrics server for trainer observability.

Exposes training metrics at /metrics in Prometheus format.
Also exposes /health endpoint for Kubernetes liveness probes.
Runs in a background thread to avoid blocking the training loop.
"""

import json
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
        self._start_time: float = 0.0
        self._last_heartbeat: float = 0.0

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
            self._run_step = Gauge(
                "trainer_run_step", "Training step for run", ["run"], registry=self._registry
            )
            self._run_tokens = Gauge(
                "trainer_run_tokens", "Total tokens processed by run", ["run"], registry=self._registry
            )
            self._run_learning_rate = Gauge(
                "trainer_run_learning_rate", "Current learning rate for run", ["run"], registry=self._registry
            )
            self._run_ready = Gauge(
                "trainer_run_ready", "Whether run is ready for updates (1=ready, 0=not ready)", ["run"], registry=self._registry
            )
            # Track known run labels for cleanup
            self._known_runs: set[str] = set()
        else:
            self._registry = None

    def _make_handler(self):
        """Create handler class with access to our registry and health state."""
        registry = self._registry
        server = self

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
                """Liveness probe: checks if main thread is responsive."""
                health_config = server.config.health

                # No health config = always healthy
                if health_config is None:
                    self._respond(200, {"status": "healthy"})
                    return

                now = time.time()
                startup_age = now - server._start_time

                # During startup grace period, always healthy
                if startup_age < health_config.startup_grace_seconds:
                    self._respond(200, {
                        "status": "healthy",
                        "reason": "startup_grace",
                        "startup_age_seconds": round(startup_age, 1),
                    })
                    return

                # No heartbeat yet after grace period = unhealthy
                if server._last_heartbeat == 0:
                    self._respond(503, {
                        "status": "unhealthy",
                        "reason": "no_heartbeat",
                    })
                    return

                # Check heartbeat age
                heartbeat_age = now - server._last_heartbeat
                if heartbeat_age > health_config.timeout_seconds:
                    self._respond(503, {
                        "status": "unhealthy",
                        "reason": "heartbeat_timeout",
                        "last_heartbeat_seconds": round(heartbeat_age, 1),
                    })
                else:
                    self._respond(200, {
                        "status": "healthy",
                        "last_heartbeat_seconds": round(heartbeat_age, 1),
                    })

            def _respond(self, code: int, body: dict):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

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

        self._start_time = time.time()
        self._server = HTTPServer((self.config.host, self.config.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._started = True
        logger.info(f"Metrics server started at http://{self.config.host}:{self.config.port}/metrics")
        logger.info(f"Health endpoint available at http://{self.config.host}:{self.config.port}/health")

    def stop(self) -> None:
        """Stop the metrics server and release the port."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()  # Close socket to release port
            if self._thread is not None:
                self._thread.join(timeout=5.0)  # Wait for thread to finish
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
        self.heartbeat()

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
        self._last_step_ts.set(self._last_heartbeat)

    def heartbeat(self) -> None:
        """Signal that the main thread is alive.

        Call this from the main training loop every iteration,
        even when idle/waiting for runs. The /health endpoint
        checks this to detect hung threads.
        """
        self._last_heartbeat = time.time()

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
