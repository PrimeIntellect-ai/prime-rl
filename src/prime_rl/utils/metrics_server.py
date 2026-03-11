"""Prometheus metrics server for observability.

Exposes metrics from a registry at /metrics in Prometheus format.
Also exposes /health endpoint for Kubernetes liveness probes.
Runs in a background thread to avoid blocking the main loop.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

if TYPE_CHECKING:
    from prime_rl.configs.shared import MetricsServerConfig


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
    Disabled by default - enable by setting `metrics_server` in config.
    """

    def __init__(self, config: "MetricsServerConfig", registry: CollectorRegistry | None = None):
        super().__init__(config.port, config.host)
        self.config = config
        self._registry = registry or CollectorRegistry()

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

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
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(generate_latest(registry))

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

        self._server = HTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._started = True
        logger.info(f"Metrics server started at http://{self.host}:{self.port}/metrics")
        logger.info(f"Health endpoint available at http://{self.host}:{self.port}/health")
