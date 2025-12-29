"""
Environment worker subprocess.

Runs environment rollouts in a separate process to isolate event loop lag.
"""

import asyncio
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import Process, Queue

import verifiers as vf
from openai import AsyncOpenAI

from prime_rl.utils.client import setup_clients
from prime_rl.utils.config import ClientConfig


@dataclass
class RolloutRequest:
    """Request to generate rollouts for an example."""

    id: int  # example_id
    rollouts_per_example: int


@dataclass
class RolloutResponse:
    """Response containing rollout results."""

    id: str
    results: list[dict]  # Simplified state dicts
    error: str | None = None
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
    clients: list[AsyncOpenAI],
    client_cycle: cycle,
    semaphore: asyncio.Semaphore,
    example_lookup: dict[int, dict],
    model_name: str,
    sampling_args: dict,
) -> RolloutResponse:
    """Process a single rollout request."""
    try:
        client = next(client_cycle)
        example = example_lookup[request.id]
        group_inputs = [vf.RolloutInput(**example) for _ in range(request.rollouts_per_example)]

        states = await env.run_group(
            group_inputs=group_inputs,
            client=client,
            model=model_name,
            gen_sampling_args=sampling_args,
            gen_sem=semaphore,
            score_sem=semaphore,
        )

        results = [extract_result(state) for state in states]
        return RolloutResponse(id=str(request.id), results=results)

    except Exception as e:
        return RolloutResponse(id=str(request.id), results=[], error=f"{type(e).__name__}: {e}")


async def worker_loop(
    request_queue: Queue,
    response_queue: Queue,
    env: vf.Environment,
    clients: list[AsyncOpenAI],
    max_concurrent: int,
    env_id: str,
    example_lookup: dict[int, dict],
    model_name: str,
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
    pending_tasks: dict[asyncio.Task, int] = {}

    def check_for_requests():
        """Non-blocking check for new requests."""
        try:
            while not request_queue.empty():
                request = request_queue.get_nowait()
                if request is None:  # Shutdown signal
                    return False
                task = asyncio.create_task(
                    process_request(
                        request, env, clients, client_cycle, semaphore, example_lookup, model_name, sampling_args
                    )
                )
                pending_tasks[task] = request.id
        except Exception:
            pass
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
                try:
                    response = task.result()
                    # Attach lag metrics to response
                    response.lag_metrics = lag_monitor.get_metrics()
                    response_queue.put(response)
                except Exception as e:
                    # Should not happen since process_request catches exceptions
                    response_queue.put(RolloutResponse(id="0", results=[], error=str(e)))
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
    model_name: str,
    seq_len: int,
    interleaved_rollouts: bool,
    max_concurrent: int,
    example_lookup: dict[int, dict],
    sampling_args: dict,
):
    """Main entry point for worker process."""
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
            model_name,
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

        self.request_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.process: Process | None = None

        # Track pending requests for response matching
        self.pending_futures: dict[str, asyncio.Future] = {}

        # Track latest lag metrics from this worker
        self.latest_lag_metrics: dict = {}

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
                self.model_name,
                self.seq_len,
                self.interleaved_rollouts,
                self.max_concurrent,
                self.example_lookup,
                self.sampling_args,
            ),
            daemon=True,
        )
        self.process.start()

    def stop(self):
        """Stop the worker process."""
        if self.process and self.process.is_alive():
            self.request_queue.put(None)  # Shutdown signal
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()

    async def submit_request(
        self,
        example_id: int,
        rollouts_per_example: int,
    ) -> asyncio.Future:
        """Submit a rollout request and return a future for the response."""
        request = RolloutRequest(
            id=example_id,
            rollouts_per_example=rollouts_per_example,
        )

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_futures[str(example_id)] = future

        self.request_queue.put(request)
        return future

    async def collect_responses(self):
        """Background task to collect responses and resolve futures."""
        while True:
            # Non-blocking check for responses
            try:
                while not self.response_queue.empty():
                    response: RolloutResponse = self.response_queue.get_nowait()
                    # Store latest lag metrics from worker
                    if response.lag_metrics:
                        self.latest_lag_metrics = response.lag_metrics
                    if response.id in self.pending_futures:
                        future = self.pending_futures.pop(response.id)
                        if response.error:
                            future.set_exception(RuntimeError(response.error))
                        else:
                            future.set_result(response.results)
            except Exception:
                pass
            await asyncio.sleep(0.01)

    def update_model_name(self, model_name: str):
        """Update the model name for future requests."""
        self.model_name = model_name

    @property
    def pending_count(self) -> int:
        """Number of pending requests for this worker."""
        return len(self.pending_futures)
