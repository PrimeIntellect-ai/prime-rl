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
from openai import AsyncOpenAI

from prime_rl.utils.client import setup_clients
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import intercept_verifiers_logging, reset_logger, setup_logger


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


@dataclass
class RolloutResponse:
    """Response containing rollout results."""

    request_id: str
    results: list[dict]  # Simplified state dicts
    lag_metrics: dict | None = None  # Event loop lag metrics from worker


def extract_result(state: vf.State) -> dict:
    """Extract only the fields needed from vf.State for IPC.

    The extracted dict must contain all fields needed by:
    - Buffer.update(): example_id, task, reward
    - orchestrator metrics: reward, is_truncated, error, timing, metrics, trajectory
    - interleave_rollout/branch_rollout: trajectory[*]["tokens"] with all token fields
    """
    # Get trajectory with tokens (needed for training)
    trajectory = []
    for step in state.get("trajectory", []):
        traj_step = {
            "prompt": step.get("prompt"),
            "completion": step.get("completion"),
            # tokens dict contains: prompt_ids, prompt_mask, completion_ids,
            # completion_mask, completion_logprobs, is_truncated
            "tokens": step.get("tokens"),
        }
        trajectory.append(traj_step)

    return {
        # Required by buffer
        "example_id": state.get("example_id"),
        "task": state.get("task"),
        "reward": state.get("reward"),
        # Required by orchestrator metrics
        "is_truncated": state.get("is_truncated", False),
        "error": type(state["error"]).__name__ if state.get("error") else None,
        "timing": dict(state.get("timing", {})),
        "metrics": state.get("metrics", {}),
        # Required for training examples
        "prompt": state.get("prompt"),
        "completion": state.get("completion"),
        "trajectory": trajectory,
    }


async def process_request(
    request: RolloutRequest,
    env: vf.Environment,
    client_cycle: cycle,
    semaphore: asyncio.Semaphore,
    example_lookup: dict[int, dict],
    sampling_args: dict,
) -> RolloutResponse:
    """Process a single rollout request."""
    client = next(client_cycle)
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

    results = [extract_result(state) for state in states]
    return RolloutResponse(request_id=request.request_id, results=results)


async def worker_loop(
    request_queue: Queue,
    response_queue: Queue,
    env: vf.Environment,
    clients: list[AsyncOpenAI],
    max_concurrent: int,
    env_id: str,
    example_lookup: dict[int, dict],
    sampling_args: dict,
):
    """Main async loop for processing requests."""
    from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor

    client_cycle = cycle(clients)
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else asyncio.Semaphore(10000)

    # Start event loop lag monitor for this worker
    lag_monitor = EventLoopLagMonitor(interval=0.1)  # More frequent sampling for workers
    lag_monitor_task = asyncio.create_task(lag_monitor.run())

    # Track in-flight tasks
    pending_tasks: dict[asyncio.Task, str] = {}

    def check_for_requests():
        """Non-blocking check for new requests."""
        while True:
            try:
                request = request_queue.get_nowait()
            except queue.Empty:
                break
            if request is None:  # Shutdown signal
                return False
            task = asyncio.create_task(
                process_request(request, env, client_cycle, semaphore, example_lookup, sampling_args)
            )
            pending_tasks[task] = request.request_id
        return True

    try:
        while True:
            # Check for new requests
            if not check_for_requests():
                break

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
):
    """Main entry point for worker process."""
    # Reset logger inherited from parent process, then setup fresh logger for this worker
    if log_file:
        reset_logger()
        setup_logger(log_level, log_file=Path(log_file), append=True, tag=worker_name)
        intercept_verifiers_logging(level=vf_log_level)

    # Load environment
    env = vf.load_environment(env_id, **env_args)
    env.set_max_seq_len(seq_len)
    env.set_interleaved_rollouts(interleaved_rollouts)

    # Create clients
    client_config = ClientConfig(**client_config_dict)
    clients = setup_clients(client_config)

    # Run async loop
    asyncio.run(
        worker_loop(
            request_queue,
            response_queue,
            env,
            clients,
            max_concurrent,
            env_id,
            example_lookup,
            sampling_args,
        )
    )


class EnvWorker:
    """Manages a worker subprocess for an environment."""

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
                raise error

            await asyncio.sleep(0.01)

    def update_model_name(self, model_name: str):
        """Update the model name for future requests."""
        self.model_name = model_name

    @property
    def pending_count(self) -> int:
        """Number of pending requests for this worker.

        Returns a large number if the worker is dead to prevent scheduler from selecting it.
        """
        if self._dead:
            return 999999  # Effectively infinite - scheduler will pick other workers
        return len(self.pending_futures)
