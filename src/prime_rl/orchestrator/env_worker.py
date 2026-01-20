"""
Environment worker subprocess.

Runs environment rollouts in a separate process to isolate event loop lag.
"""

import asyncio
import queue
import time
import uuid
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import Process, Queue
from pathlib import Path

import httpx
import verifiers as vf
from openai import AsyncOpenAI

from prime_rl.utils.client import setup_clients
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.elastic import discover_server_ips
from prime_rl.utils.logger import get_logger, intercept_verifiers_logging, reset_logger, setup_logger


class WorkerDiedError(Exception):
    """Raised when a worker subprocess dies unexpectedly."""

    pass


@dataclass
class RolloutRequest:
    """Request to generate rollouts for an example."""

    request_id: str
    example_id: int
    rollouts_per_example: int
    model_name: str


@dataclass
class RolloutResponse:
    """Response containing rollout results."""

    request_id: str
    results: list[dict]
    lag_metrics: dict | None = None


async def _check_server(url: str, model_name: str, timeout: float = 5.0) -> tuple[bool, bool]:
    """Check server status. Returns (has_model, is_healthy)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            return model_name in models, len(models) > 0
    except Exception:
        return False, False


async def discover_ready_servers(hostname: str, port: int, model_name: str) -> list[str]:
    """Discover servers via DNS with majority vote logic.

    - If NO servers have the model: return all healthy servers (base model mode)
    - If ANY server has the model: return only those with it (adapter mode)
    """
    loop = asyncio.get_event_loop()
    ips = await loop.run_in_executor(None, discover_server_ips, hostname)
    if not ips:
        return []

    checks = [_check_server(f"http://{ip}:{port}", model_name) for ip in ips]
    results = await asyncio.gather(*checks, return_exceptions=True)

    with_model, healthy = [], []
    for ip, result in zip(ips, results):
        if isinstance(result, Exception):
            continue
        has_model, is_healthy = result
        url = f"http://{ip}:{port}/v1"
        if has_model:
            with_model.append(url)
        if is_healthy:
            healthy.append(url)

    return with_model if with_model else healthy


def extract_result(state: vf.State) -> dict:
    """Extract only the fields needed from vf.State for IPC.

    The extracted dict must contain all fields needed by:
    - Buffer.update(): example_id, task, reward
    - orchestrator metrics: reward, is_truncated, error, timing, metrics, trajectory
    - interleave_rollout/branch_rollout: trajectory[*]["tokens"] with all token fields
    """
    trajectory = []
    for step in state.get("trajectory", []):
        traj_step = {
            "prompt": step.get("prompt"),
            "completion": step.get("completion"),
            "tokens": step.get("tokens"),
        }
        trajectory.append(traj_step)

    return {
        "example_id": state.get("example_id"),
        "task": state.get("task"),
        "reward": state.get("reward"),
        "is_truncated": state.get("is_truncated", False),
        "error": type(state["error"]).__name__ if state.get("error") else None,
        "timing": dict(state.get("timing", {})),
        "metrics": state.get("metrics", {}),
        "prompt": state.get("prompt"),
        "completion": state.get("completion"),
        "trajectory": trajectory,
    }


async def worker_loop(
    request_queue: Queue,
    response_queue: Queue,
    env: vf.Environment,
    clients: list[AsyncOpenAI],
    client_config: ClientConfig,
    max_concurrent: int,
    example_lookup: dict[int, dict],
    sampling_args: dict,
    model_name: str,
    elastic_hostname: str | None = None,
    elastic_port: int = 8000,
    elastic_sync_interval: float = 5.0,
):
    """Main async loop for processing rollout requests."""
    from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor

    logger = get_logger()
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else asyncio.Semaphore(10000)

    lag_monitor = EventLoopLagMonitor(interval=0.1)
    lag_monitor_task = asyncio.create_task(lag_monitor.run())

    # Mutable state for elastic mode
    current_clients = clients
    client_cycle = cycle(current_clients) if current_clients else cycle([None])
    last_refresh = 0.0
    last_urls: set[str] = set()

    pending_tasks: dict[asyncio.Task, str] = {}
    waiting_requests: list[RolloutRequest] = []

    async def refresh_clients():
        """Refresh clients via DNS discovery in elastic mode."""
        nonlocal current_clients, client_cycle, last_refresh, last_urls

        if not elastic_hostname:
            return
        if time.time() - last_refresh < elastic_sync_interval:
            return
        last_refresh = time.time()

        urls = await discover_ready_servers(elastic_hostname, elastic_port, model_name)
        if set(urls) == last_urls:
            return
        last_urls = set(urls)

        for c in current_clients:
            c.close()

        if not urls:
            logger.debug("No ready inference servers found")
            current_clients = []
            client_cycle = cycle([None])
            return

        logger.debug(f"Discovered {len(urls)} ready server(s)")
        current_clients = setup_clients(
            ClientConfig(
                timeout=client_config.timeout,
                base_url=urls,
                api_key_var=client_config.api_key_var,
                headers=client_config.headers,
            )
        )
        client_cycle = cycle(current_clients)

    async def process_request(request: RolloutRequest, client: AsyncOpenAI) -> RolloutResponse:
        example = example_lookup[request.example_id]
        group_inputs = [vf.RolloutInput(**example) for _ in range(request.rollouts_per_example)]
        states = await env.run_group(
            group_inputs=group_inputs,
            client=client,
            model=request.model_name,
            gen_sampling_args=sampling_args,
            gen_sem=semaphore,
            score_sem=semaphore,
        )
        return RolloutResponse(request_id=request.request_id, results=[extract_result(s) for s in states])

    try:
        while True:
            await refresh_clients()

            # Process waiting requests if we now have clients
            if current_clients and waiting_requests:
                for req in waiting_requests:
                    client = next(client_cycle)
                    task = asyncio.create_task(process_request(req, client))
                    pending_tasks[task] = req.request_id
                waiting_requests.clear()

            # Drain request queue
            while True:
                try:
                    request = request_queue.get_nowait()
                except queue.Empty:
                    break
                if request is None:
                    return  # Shutdown

                client = next(client_cycle)
                if client is None:
                    waiting_requests.append(request)
                else:
                    task = asyncio.create_task(process_request(request, client))
                    pending_tasks[task] = request.request_id

            if not pending_tasks:
                await asyncio.sleep(0.01)
                continue

            done, _ = await asyncio.wait(pending_tasks.keys(), timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending_tasks.pop(task)
                response = task.result()
                response.lag_metrics = lag_monitor.get_metrics()
                response_queue.put(response)
    finally:
        lag_monitor_task.cancel()
        for c in current_clients:
            c.close()
        for task in pending_tasks:
            task.cancel()


def worker_main(
    request_queue: Queue,
    response_queue: Queue,
    env_id: str,
    env_args: dict,
    client_config_dict: dict,
    seq_len: int,
    interleaved_rollouts: bool,
    max_concurrent: int,
    example_lookup: dict[int, dict],
    sampling_args: dict,
    log_level: str,
    vf_log_level: str,
    log_file: str | None,
    worker_name: str | None = None,
    model_name: str = "",
    elastic_hostname: str | None = None,
    elastic_port: int = 8000,
    elastic_sync_interval: float = 5.0,
):
    """Main entry point for worker subprocess."""
    if log_file:
        reset_logger()
        setup_logger(log_level, log_file=Path(log_file), append=True, tag=worker_name)
        intercept_verifiers_logging(level=vf_log_level)

    env = vf.load_environment(env_id, **env_args)
    env.set_max_seq_len(seq_len)
    env.set_interleaved_rollouts(interleaved_rollouts)

    client_config = ClientConfig(**client_config_dict)
    clients = [] if elastic_hostname else setup_clients(client_config)

    asyncio.run(
        worker_loop(
            request_queue,
            response_queue,
            env,
            clients,
            client_config,
            max_concurrent,
            example_lookup,
            sampling_args,
            model_name,
            elastic_hostname,
            elastic_port,
            elastic_sync_interval,
        )
    )


class EnvWorker:
    """Manages a worker subprocess for environment rollouts."""

    def __init__(
        self,
        env_id: str,
        env_args: dict,
        client_config: ClientConfig,
        model_name: str,
        seq_len: int,
        interleaved_rollouts: bool,
        max_concurrent: int,
        example_lookup: dict[int, dict],
        sampling_args: dict,
        worker_name: str | None = None,
        log_level: str = "warn",
        vf_log_level: str = "warn",
        log_file: str | None = None,
        max_restarts: int = 5,
        elastic_hostname: str | None = None,
        elastic_port: int = 8000,
        elastic_sync_interval: float = 5.0,
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.client_config = client_config
        self.model_name = model_name
        self.seq_len = seq_len
        self.interleaved_rollouts = interleaved_rollouts
        self.max_concurrent = max_concurrent
        self.example_lookup = example_lookup
        self.sampling_args = sampling_args
        self.worker_name = worker_name or env_id

        self.log_level = log_level
        self.vf_log_level = vf_log_level
        self.log_file = log_file
        self.max_restarts = max_restarts

        # Elastic mode parameters (worker does its own DNS discovery)
        self.elastic_hostname = elastic_hostname
        self.elastic_port = elastic_port
        self.elastic_sync_interval = elastic_sync_interval

        self.request_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.process: Process | None = None

        # Track pending requests for response matching
        self.pending_futures: dict[str, asyncio.Future] = {}

        # Track latest lag metrics from this worker
        self.latest_lag_metrics: dict = {}

        # Track intentional shutdown to avoid false error on clean stop
        self._stopping = False
        # Track if worker died unexpectedly (prevents scheduler from routing to dead worker)
        self._dead = False
        # Track restart count to prevent infinite restart loops
        self._restart_count = 0
        # Track fatal error when max restarts exceeded (orchestrator should crash)
        self._fatal_error: Exception | None = None
        # Track successful responses since last restart (to reset restart count)
        self._responses_since_restart = 0

    def start(self):
        """Start the worker process."""
        self.process = Process(
            target=worker_main,
            args=(
                self.request_queue,
                self.response_queue,
                self.env_id,
                self.env_args,
                self.client_config.model_dump(),
                self.seq_len,
                self.interleaved_rollouts,
                self.max_concurrent,
                self.example_lookup,
                self.sampling_args,
                self.log_level,
                self.vf_log_level,
                self.log_file,
                self.worker_name,
                self.model_name,
                self.elastic_hostname,
                self.elastic_port,
                self.elastic_sync_interval,
            ),
            daemon=True,
        )
        self.process.start()
        self._stopping = False  # Reset after process is alive to avoid race condition
        self._dead = False  # Reset in case of restart

    def stop(self):
        """Stop the worker process."""
        self._stopping = True
        if self.process and self.process.is_alive():
            self.request_queue.put(None)  # Shutdown signal
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()

    def _restart(self):
        """Restart the worker process after unexpected death."""
        # Clean up old process if it exists
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            # Always join to reap zombie process, even if already dead
            self.process.join(timeout=5)
            self.process.close()

        # Clear queues to avoid stale data (drain without blocking)
        while True:
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

        # Start fresh process
        self.start()

    async def submit_request(
        self,
        example_id: int,
        rollouts_per_example: int,
    ) -> tuple[asyncio.Future, str]:
        """Submit a rollout request and return a (future, request_id) tuple."""
        request_id = uuid.uuid4().hex
        request = RolloutRequest(
            request_id=request_id,
            example_id=example_id,
            rollouts_per_example=rollouts_per_example,
            model_name=self.model_name,
        )

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_futures[request_id] = future

        self.request_queue.put(request)
        return future, request_id

    async def collect_responses(self):
        """Background task to collect responses and resolve futures."""
        logger = get_logger()
        while True:
            # Drain queue first to salvage any responses before checking for dead worker
            while True:
                try:
                    response: RolloutResponse = self.response_queue.get_nowait()
                except queue.Empty:
                    break
                # Store latest lag metrics from worker
                if response.lag_metrics:
                    self.latest_lag_metrics = response.lag_metrics
                if response.request_id in self.pending_futures:
                    future = self.pending_futures.pop(response.request_id)
                    # Check if future was cancelled (e.g., by update_policy)
                    if not future.done():
                        future.set_result(response.results)
                    # Track successful responses; reset restart count after stable operation
                    self._responses_since_restart += 1
                    if self._responses_since_restart >= 10 and self._restart_count > 0:
                        logger.debug(
                            f"Worker '{self.worker_name}' stable after {self._responses_since_restart} responses, resetting restart count"
                        )
                        self._restart_count = 0
                        self._responses_since_restart = 0

            # Check if worker process died unexpectedly (but not during intentional shutdown)
            if self.process and not self.process.is_alive() and not self._stopping:
                exit_code = self.process.exitcode
                error = WorkerDiedError(f"Worker '{self.worker_name}' died unexpectedly (exit code: {exit_code})")
                # Mark worker as dead so scheduler won't route new requests here
                self._dead = True
                # Fail remaining pending futures so callers don't hang indefinitely
                for future in self.pending_futures.values():
                    if not future.done():
                        future.set_exception(error)
                self.pending_futures.clear()

                # Check if we've exceeded max restarts (-1 means unlimited)
                self._restart_count += 1
                if self.max_restarts >= 0 and self._restart_count > self.max_restarts:
                    logger.error(
                        f"Worker '{self.worker_name}' died {self._restart_count} times, exceeding max restarts ({self.max_restarts}). Giving up."
                    )
                    # Store fatal error so orchestrator can detect and crash
                    self._fatal_error = error
                    raise error

                # Log warning and restart the worker automatically
                restart_info = (
                    f"{self._restart_count}/{self.max_restarts}" if self.max_restarts >= 0 else f"{self._restart_count}"
                )
                logger.warning(
                    f"Worker '{self.worker_name}' died unexpectedly (exit code: {exit_code}). "
                    f"Restarting worker automatically ({restart_info}). In-flight requests will be rescheduled."
                )
                self._responses_since_restart = 0  # Reset on restart
                self._restart()

            await asyncio.sleep(0.01)

    @property
    def pending_count(self) -> int:
        """Number of pending requests for this worker.

        Returns a large number if the worker is dead to prevent scheduler from selecting it.
        """
        if self._dead:
            return 999999  # Effectively infinite - scheduler will pick other workers
        return len(self.pending_futures)
