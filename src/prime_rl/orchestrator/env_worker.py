"""
Environment worker subprocess.

Runs environment rollouts in a separate process to isolate event loop lag.
"""

import asyncio
import queue
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import Process, Queue

import verifiers as vf

from prime_rl.utils.client import setup_clients
from prime_rl.utils.config import ClientConfig


# TODO: move to utils later
# future TODO: have vf guarantee that base fields of vf.State are serializable
def to_serializable_state(state: vf.State) -> dict:
    trajectory = []
    for step in state.get("trajectory", []):
        step = {
            "prompt": step.get("prompt"),
            "completion": step.get("completion"),
            "tokens": step.get("tokens"),
        }
        trajectory.append(step)

    return {
        # required by buffer
        "example_id": state.get("example_id"),
        "task": state.get("task"),
        "reward": state.get("reward"),
        # required by orchestrator metrics
        "is_truncated": state.get("is_truncated", False),
        "error": type(state["error"]).__name__ if state.get("error") else None,
        "timing": dict(state.get("timing", {})),
        "metrics": state.get("metrics", {}),
        # required for training examples
        "prompt": state.get("prompt"),
        "completion": state.get("completion"),
        "trajectory": trajectory,
    }


@dataclass
class EnvWorkerRequest:
    request_id: int
    example_id: int
    rollouts_per_example: int
    model_name: str  # required for multi-tenant run


@dataclass
class EnvWorkerResponse:
    request_id: int
    serialized_state: list[dict]  # serialized vf.State


class EnvWorkerHelper:
    """Consumes env worker requests from a queue, runs rollouts and returns responses to a response queue. Used by EnvWorker which starts the helper in a separate process."""

    def __init__(
        self,
        # all args need to be picklable
        env_id: str,
        env_args: dict,
        client_config_dict: dict,
        seq_len: int,
        interleaved_rollouts: bool,
        max_concurrent: int,
        sampling_args: dict,
        request_queue: "Queue[EnvWorkerRequest]",
        response_queue: "Queue[EnvWorkerResponse]",
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.client_config_dict = client_config_dict
        self.seq_len = seq_len
        self.interleaved_rollouts = interleaved_rollouts
        self.max_concurrent = max_concurrent
        self.sampling_args = sampling_args

        self.process: Process | None = None
        self.request_queue = request_queue
        self.response_queue = response_queue

    def start(self):
        # load environment
        env = vf.load_environment(self.env_id, **self.env_args)
        env.set_max_seq_len(self.seq_len)
        env.set_interleaved_rollouts(self.interleaved_rollouts)
        dataset = env.get_dataset()

        # create client
        client_config = ClientConfig(**self.client_config_dict)
        clients = setup_clients(client_config)
        client_cycle = cycle(clients)

        pending_requests: dict[asyncio.Task[EnvWorkerResponse], EnvWorkerRequest] = {}

        # create semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent) if self.max_concurrent > 0 else asyncio.Semaphore(10000)

        async def process_request(request: EnvWorkerRequest) -> EnvWorkerResponse:
            """Process a single rollout request."""
            client = next(client_cycle)
            example = dataset[request.example_id]
            group_inputs = [vf.RolloutInput(**example) for _ in range(request.rollouts_per_example)]

            states = await env.run_group(
                group_inputs=group_inputs,
                client=client,
                model=request.model_name,
                gen_sampling_args=self.sampling_args,
                gen_sem=semaphore,
                score_sem=semaphore,
            )

            serialized_state = [to_serializable_state(state) for state in states]
            return EnvWorkerResponse(request_id=request.request_id, serialized_state=serialized_state)

        def process_request_loop():
            """Non-blocking check for new requests."""
            while True:
                try:
                    request = self.request_queue.get_nowait()
                except queue.Empty:
                    break
                task = asyncio.create_task(process_request(request))
                pending_requests[task] = request
            return True

        async def worker_loop():
            while True:
                # process request
                running = process_request_loop()

                if not running:
                    break

                if not pending_requests:
                    # No pending tasks, wait a bit for new requests
                    await asyncio.sleep(0.01)
                    continue

                # Wait for at least one task to complete
                finished_requests, _ = await asyncio.wait(
                    pending_requests.keys(), timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )

                for request in finished_requests:
                    pending_requests.pop(request)
                    response = request.result()
                    self.response_queue.put(response)

        # Run async loop
        asyncio.run(worker_loop())


class EnvWorker:
    """Proxies vf.Environment.run_group, but delegates work to EnvWorkerHelper in a separate process."""

    def __init__(
        self,
        env_id: str,
        env_args: dict,
        client_config: ClientConfig,
        seq_len: int,
        interleaved_rollouts: bool,
        max_concurrent: int,
        sampling_args: dict,
    ):
        self.request_queue: Queue[EnvWorkerRequest] = Queue()
        self.response_queue: Queue[EnvWorkerResponse] = Queue()
        self.env_worker_helper = EnvWorkerHelper(
            env_id=env_id,
            env_args=env_args,
            client_config_dict=client_config.model_dump(),
            seq_len=seq_len,
            interleaved_rollouts=interleaved_rollouts,
            max_concurrent=max_concurrent,
            sampling_args=sampling_args,
            request_queue=self.request_queue,
            response_queue=self.response_queue,
        )
        self.env_worker_process: Process | None = None
        self.response_collector_task: asyncio.Task | None = None
        self.pending_futures: dict[int, asyncio.Future] = {}
        self.next_request_id: int = 0

    def start(self):
        self.env_worker_process = Process(target=self.env_worker_helper.start, daemon=True)
        self.env_worker_process.start()
        self.response_collector_task = asyncio.create_task(self.collect_responses())

    async def stop(self):
        if self.response_collector_task:
            self.response_collector_task.cancel()
            try:
                await self.response_collector_task
            except asyncio.CancelledError:
                pass

        if self.env_worker_process:
            self.env_worker_process.terminate()
            self.env_worker_process.join(timeout=1)

        self.response_queue.close()
        self.request_queue.close()

    async def run_group(
        self,
        example_id: int,
        rollouts_per_example: int,
        model_name: str,
    ) -> list[dict]:
        """Run a specified example from the environment."""
        request_id = self.next_request_id
        self.next_request_id += 1

        request = EnvWorkerRequest(
            request_id=request_id,
            example_id=example_id,
            rollouts_per_example=rollouts_per_example,
            model_name=model_name,
        )
        self.request_queue.put(request)
        future = asyncio.Future()
        self.pending_futures[request_id] = future

        return await future

    async def collect_responses(self):
        """Background task to collect responses from the response queue and resolve futures."""
        while True:
            while True:
                try:
                    response = self.response_queue.get_nowait()
                except queue.Empty:
                    break
                if response.request_id in self.pending_futures:
                    future = self.pending_futures.pop(response.request_id)
                    # Check if future was cancelled (e.g. by update_policy)
                    if not future.done():
                        future.set_result(response.serialized_state)

            await asyncio.sleep(0.01)


if __name__ == "__main__":
    # env helper
    # request_queue: "Queue[EnvWorkerRequest]" = Queue()
    # response_queue: "Queue[EnvWorkerResponse]" = Queue()
    # env_worker_helper = EnvWorkerHelper(
    #     env_id="gsm8k",
    #     env_args={},
    #     client_config_dict={},
    #     seq_len=1024,
    #     interleaved_rollouts=True,
    #     max_concurrent=10,
    #     sampling_args={},
    #     request_queue=request_queue,
    #     response_queue=response_queue,
    # )
    # Process(target=env_worker_helper.start, daemon=True).start()

    # env_worker_request = EnvWorkerRequest(
    #     example_id=0,
    #     rollouts_per_example=1,
    #     model_name="Qwen/Qwen3-4B-Instruct-2507",
    # )
    # request_queue.put(env_worker_request)
    # response = response_queue.get()
    # print(response)

    # env worker
    async def main():
        env_worker = EnvWorker(
            env_id="gsm8k",
            env_args={},
            client_config=ClientConfig(),
            seq_len=1024,
            interleaved_rollouts=True,
            max_concurrent=10,
            sampling_args={},
        )
        env_worker.start()

        tasks = []
        for i in range(10):
            tasks.append(
                env_worker.run_group(example_id=i, rollouts_per_example=1, model_name="Qwen/Qwen3-4B-Instruct-2507")
            )
        responses = await asyncio.gather(*tasks)
        for response in responses:
            print(response)

        # await env_worker.stop()

    asyncio.run(main())
