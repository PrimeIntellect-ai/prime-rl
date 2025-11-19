"""Environment worker process manager for local development."""

import asyncio
import atexit
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import msgpack
import zmq
from loguru import logger

from prime_rl.orchestrator.utils import msgpack_encoder


class EnvironmentManager:
    """Manages a single environment worker process."""

    def __init__(
        self,
        env_id: str,
        instance_name: str,
        endpoint: str,
        env_args: dict | None = None,
        output_dir: Path | None = None,
        log_level: str = "info",
    ):
        """
        Initialize environment manager.

        Args:
            env_id: Environment ID (e.g., "math")
            instance_name: Unique instance name (e.g., "math-0")
            endpoint: ZMQ endpoint (e.g., "tcp://localhost:5555" or "ipc:///tmp/math-0.sock")
            env_args: Environment initialization arguments
            output_dir: Output directory for logs (if None, logs won't be written to file)
            log_level: Logging level to use for the environment worker (e.g., "info", "debug")
        """
        self.env_id = env_id
        self.instance_name = instance_name
        self.endpoint = endpoint
        self.env_args = env_args or {}
        self.output_dir = output_dir
        self.log_level = log_level
        self.worker: subprocess.Popen | None = None

    def start_worker(self, startup_timeout: float = 30.0) -> str:
        """
        Start environment worker subprocess - this is primarily used for local development.

        Args:
            startup_timeout: Maximum time to wait for worker to start (seconds)

        Returns:
            Endpoint that was used
        """
        # Convert tcp://localhost:PORT to tcp://*:PORT for binding
        # Use urlparse for robust hostname replacement
        parsed = urlparse(self.endpoint)
        if parsed.scheme == "tcp":
            # Replace hostname with * for binding (listen on all interfaces)
            bind_address = urlunparse(parsed._replace(netloc=f"*:{parsed.port}"))
        else:
            # For IPC or other schemes, use as-is
            bind_address = self.endpoint

        # Set up log file if output_dir is provided
        log_file = None
        if self.output_dir:
            log_dir = self.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "env.log"
            logger.debug(f"Writing {self.instance_name} worker logs to {log_file}")

        # Build command
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "prime_rl.environment.zmq_server",
            "--bind",
            bind_address,
            "--env-id",
            self.env_id,
            "--instance-name",
            self.instance_name,
            "--log-level",
            self.log_level,
        ]

        if log_file:
            cmd.extend(["--log-file", str(log_file)])

        if self.env_args:
            cmd.extend(["--env-args", json.dumps(self.env_args)])

        logger.info(f"Starting {self.instance_name} worker: {' '.join(cmd)}")

        # Start subprocess - loguru will handle file logging, so we don't redirect stdout/stderr
        self.worker = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for startup with health check polling
        logger.info(f"Waiting for {self.instance_name} to be ready (timeout: {startup_timeout}s)...")
        self._wait_for_health_check(startup_timeout, log_file)

        logger.success(f"✓ {self.instance_name} worker started on {self.endpoint}")

        # Register cleanup on exit
        atexit.register(self.cleanup)

        return self.endpoint

    def _wait_for_health_check(self, timeout: float, log_file: Path | None = None):
        """
        Poll the worker's health endpoint until it responds or timeout.

        Args:
            timeout: Maximum time to wait in seconds
            log_file: Optional path to log file (included in error messages)

        Raises:
            RuntimeError: If worker fails to start or doesn't respond within timeout
        """
        start_time = time.time()
        poll_interval = 0.1  # Start with 100ms polling

        # Create a temporary ZMQ socket for health check
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 100)  # 100ms receive timeout
        sock.connect(self.endpoint)

        try:
            while time.time() - start_time < timeout:
                # Check if process died
                if self.worker.poll() is not None:
                    stdout, stderr = self.worker.communicate()
                    error_msg = f"Worker {self.instance_name} died during startup"
                    if log_file:
                        error_msg += f". Check logs at {log_file}"
                    error_msg += f"\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                    raise RuntimeError(error_msg)

                try:
                    # Send health check request
                    request_id = uuid.uuid4().bytes
                    health_request = msgpack.packb({"action": "health"}, default=msgpack_encoder, use_bin_type=True)
                    sock.send_multipart([request_id, health_request])

                    # Try to receive response
                    msg = sock.recv_multipart()
                    if len(msg) >= 2:
                        response = msgpack.unpackb(msg[1], raw=False)
                        if response.get("status") == "healthy":
                            logger.debug(f"{self.instance_name} health check succeeded")
                            return  # Success!

                except zmq.Again:
                    # Timeout - server not ready yet
                    pass
                except Exception as e:
                    logger.debug(f"Health check attempt failed: {e}")

                # Exponential backoff up to 1 second
                time.sleep(min(poll_interval, 1.0))
                poll_interval *= 1.5

            # Timeout reached
            error_msg = (
                f"Worker {self.instance_name} failed to respond to health check within {timeout}s. "
                f"It may still be loading dependencies (vllm, transformers, torch)."
            )
            if log_file:
                error_msg += f" Check logs at {log_file}"
            raise RuntimeError(error_msg)

        finally:
            sock.close()
            ctx.term()

    def cleanup(self):
        """Terminate worker process."""
        if self.worker and self.worker.poll() is None:
            logger.info(f"Terminating {self.instance_name} worker...")
            self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Worker {self.instance_name} did not terminate, killing...")
                self.worker.kill()
