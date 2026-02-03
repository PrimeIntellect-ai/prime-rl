"""
Environment worker subprocess.

Runs environment rollouts in a separate process to isolate event loop lag.
"""

import asyncio
import queue
import uuid
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import Process, Queue
from pathlib import Path

import verifiers as vf
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from verifiers.utils.async_utils import maybe_semaphore

from prime_rl.utils.client import setup_clients
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.elastic import ServerDiscovery
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
    model_name: str  # Model name to use for this request (may change for LoRA)
    sampling_args: dict


@dataclass
class RolloutResponse:
    """Response containing rollout results."""

    request_id: str
    results: list[dict]  # Simplified state dicts
    lag_metrics: dict | None = None  # Event loop lag metrics from worker


def extract_result(output: vf.RolloutOutput, temperature: float) -> dict:
    """
    Extract only the fields from vf.RolloutOutput needed to build training
    samples and logging.

    The extracted dict must contain all fields needed by:
    - Buffer.update(): example_id, task, reward
    - orchestrator metrics: reward, is_truncated, error, timing, metrics, trajectory
    - interleave_rollout/branch_rollout: trajectory[*]["tokens"] with all token fields

    For multimodal (Qwen3-VL), tokens dict may also contain:
    - pixel_values: flattened image patches [num_patches, patch_dim]
    - image_grid_thw: grid dimensions [num_images, 3]

    Args:
        state: The vf.State from the environment rollout
        temperature: The temperature used during generation (from sampling args)
    """
    # Get trajectory with tokens (needed for training)
    trajectory = []
    for step in output.get("trajectory", []):
        traj_step = {
            "prompt": step.get("prompt"),
            "completion": step.get("completion"),
            # tokens dict contains: prompt_ids, prompt_mask, completion_ids,
            # completion_mask, completion_logprobs, is_truncated
            # For multimodal: also pixel_values, image_grid_thw
            "tokens": step.get("tokens"),
            "temperature": temperature,  # Store temperature per-turn for per-token temp support
        }
        trajectory.append(traj_step)

    return {
        # Required by buffer
        "example_id": output.get("example_id"),
        "task": output.get("task"),
        "reward": output.get("reward"),
        # Required by orchestrator metrics
        "is_truncated": output.get("is_truncated", False),
        "error": output.get("error"),
        "timing": output.get("timing", {}),
        "metrics": output.get("metrics", {}),
        # Required for training examples
        "prompt": output.get("prompt"),
        "completion": output.get("completion"),
        "trajectory": trajectory,
    }


async def worker_loop(
    request_queue: Queue,
    response_queue: Queue,
    env: vf.Environment,
    clients: list[AsyncOpenAI],
    client_config: ClientConfig,
    max_concurrent_groups: int,
    tasks_per_minute: int,
    example_lookup: dict[int, dict],
    model_name: str,
):
    """Main async loop for processing rollout requests."""
    from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor

    semaphore = await maybe_semaphore(max_concurrent_groups)
    rate_limiter = AsyncLimiter(tasks_per_minute, 60) if tasks_per_minute > 0 else None

    # Start event loop lag monitor for this worker
    lag_monitor = EventLoopLagMonitor(interval=0.1)  # More frequent sampling for workers
    lag_monitor_task = asyncio.create_task(lag_monitor.run())

    # Setup client discovery (elastic mode) or static client cycle
    discovery: ServerDiscovery | None = None
    if client_config.is_elastic:
        discovery = ServerDiscovery.from_config(client_config, model_name)
    static_cycle = cycle(clients) if clients else None

    # Track in-flight tasks
    pending_tasks: dict[asyncio.Task, str] = {}
    waiting_requests: list[RolloutRequest] = []

    def get_next_client():
        """Get next client from discovery or static cycle."""
        if discovery:
            return discovery.get_next_client()
        return next(static_cycle) if static_cycle else None

    def has_clients() -> bool:
        """Check if clients are available."""
        if discovery:
            return discovery.has_clients
        return bool(clients)

    async def process_request(request: RolloutRequest, client: AsyncOpenAI) -> RolloutResponse:
        if rate_limiter:
            await rate_limiter.acquire()
        example = example_lookup[request.example_id]
        group_inputs = [vf.RolloutInput(**example) for _ in range(request.rollouts_per_example)]
        async with semaphore:
            outputs = await env.run_group(
                group_inputs=group_inputs,
                client=client,
                model=request.model_name,
                sampling_args=request.sampling_args,  # Use per-request sampling args for temp scheduling
                state_columns=["trajectory"],
            )
        temperature = request.sampling_args["temperature"]
        return RolloutResponse(request_id=request.request_id, results=[extract_result(o, temperature) for o in outputs])

    try:
        while True:
            if discovery:
                await discovery.refresh()

            # Process waiting requests if we now have clients
            if has_clients() and waiting_requests:
                for req in waiting_requests:
                    task = asyncio.create_task(process_request(req, get_next_client()))
                    pending_tasks[task] = req.request_id
                waiting_requests.clear()

            # Drain request queue
            while True:
                try:
                    request = request_queue.get_nowait()
                except queue.Empty:
                    break
                if request is None:  # Shutdown signal
                    return

                # Update discovery model name if it changed (e.g., switched to LoRA)
                if discovery and request.model_name != discovery.model_name:
                    discovery.model_name = request.model_name
                    discovery._urls = []  # Force refresh on next call

                if has_clients():
                    task = asyncio.create_task(process_request(request, get_next_client()))
                    pending_tasks[task] = request.request_id
                else:
                    waiting_requests.append(request)

            if not pending_tasks:
                # No pending tasks, wait a bit for new requests
                await asyncio.sleep(0.01)
                continue

            # Wait for at least one task to complete
            done, _ = await asyncio.wait(pending_tasks.keys(), timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending_tasks.pop(task)
                response = task.result()
                # Attach lag metrics to response
                response.lag_metrics = lag_monitor.get_metrics()
                response_queue.put(response)
    finally:
        # Cleanup
        lag_monitor_task.cancel()
        if discovery:
            await discovery.stop()
        else:
            for c in clients:
                await c.close()
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
    max_concurrent_groups: int,
    tasks_per_minute: int,
    example_lookup: dict[int, dict],
    log_level: str,
    vf_log_level: str,
    log_file: str | None,
    worker_name: str | None = None,
    model_name: str = "",
    json_logging: bool = False,
):
    """Main entry point for worker subprocess."""
    # Reset logger inherited from parent process, then setup fresh logger for this worker
    if log_file:
        reset_logger()
        setup_logger(log_level, log_file=Path(log_file), append=True, tag=worker_name, json_logging=json_logging)
        intercept_verifiers_logging(level=vf_log_level)

    # Load environment
    env = vf.load_environment(env_id, **env_args)
    env.set_max_seq_len(seq_len)
    env.set_interleaved_rollouts(interleaved_rollouts)

    # Create clients (empty list in elastic mode - workers discover servers dynamically)
    client_config = ClientConfig(**client_config_dict)
    clients = [] if client_config.is_elastic else setup_clients(client_config)

    # Run async loop
    asyncio.run(
        worker_loop(
            request_queue,
            response_queue,
            env,
            clients,
            client_config,
            max_concurrent_groups,
            tasks_per_minute,
            example_lookup,
            model_name,
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
        max_concurrent_groups: int,
        tasks_per_minute: int,
        example_lookup: dict[int, dict],
        worker_name: str | None = None,
        log_level: str = "warn",
        vf_log_level: str = "warn",
        log_file: str | None = None,
        max_restarts: int = 5,
        json_logging: bool = False,
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.client_config = client_config
        self.model_name = model_name
        self.seq_len = seq_len
        self.interleaved_rollouts = interleaved_rollouts
        self.max_concurrent_groups = max_concurrent_groups
        self.tasks_per_minute = tasks_per_minute
        self.example_lookup = example_lookup
        self.worker_name = worker_name or env_id

        self.log_level = log_level
        self.vf_log_level = vf_log_level
        self.log_file = log_file
        self.max_restarts = max_restarts
        self.json_logging = json_logging

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
                self.max_concurrent_groups,
                self.tasks_per_minute,
                self.example_lookup,
                self.log_level,
                self.vf_log_level,
                self.log_file,
                self.worker_name,
                self.model_name,
                self.json_logging,
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
        sampling_args: dict,
    ) -> tuple[asyncio.Future, str]:
        """Submit a rollout request and return a (future, request_id) tuple."""
        request_id = uuid.uuid4().hex
        request = RolloutRequest(
            request_id=request_id,
            example_id=example_id,
            rollouts_per_example=rollouts_per_example,
            model_name=self.model_name,
            sampling_args=sampling_args,
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

    def update_model_name(self, model_name: str):
        """Update the model name for future requests."""
        get_logger().debug(f"Worker '{self.worker_name}' switching model: {self.model_name} -> {model_name}")
        self.model_name = model_name

    @property
    def pending_count(self) -> int:
        """Number of pending requests for this worker.

        Returns a large number if the worker is dead to prevent scheduler from selecting it.
        """
        if self._dead:
            return 999999  # Effectively infinite - scheduler will pick other workers
        return len(self.pending_futures)
