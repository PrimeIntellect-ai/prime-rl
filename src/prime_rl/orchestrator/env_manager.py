"""Environment worker process manager for local development."""

import atexit
import json
import subprocess
import time
from pathlib import Path

from loguru import logger


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

    def start_worker(self) -> str:
        """
        Start environment worker subprocess - this is primarily used for local development.

        Returns:
            Endpoint that was used
        """
        # Convert tcp://localhost:PORT to tcp://*:PORT for binding
        bind_address = self.endpoint.replace("tcp://localhost:", "tcp://*:")

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

        # Wait for startup
        time.sleep(1.0)

        # Check if it started successfully
        if self.worker.poll() is not None:
            stdout, stderr = self.worker.communicate()
            error_msg = f"Worker {self.instance_name} failed to start"
            if log_file:
                error_msg += f". Check logs at {log_file}"
            error_msg += f"\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            raise RuntimeError(error_msg)

        logger.success(f"✓ {self.instance_name} worker started on {self.endpoint}")

        # Register cleanup on exit
        atexit.register(self.cleanup)

        return self.endpoint

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
