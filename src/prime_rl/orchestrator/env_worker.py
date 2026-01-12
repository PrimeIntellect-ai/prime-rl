"""
Environment worker subprocess.

Runs environment rollouts in a separate process to isolate event loop lag.
"""

import asyncio
import queue
from dataclasses import dataclass
from itertools import cycle
from multiprocessing import Event, Process, Queue, Value
from typing import Literal

import verifiers as vf

from prime_rl.eval.utils import prepare_sampling_args as get_eval_sampling_args  # TODO: rename
from prime_rl.orchestrator.config import EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.utils import get_sampling_args as get_train_sampling_args  # TODO: rename
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
class BaseRequestResponse:
    request_id: int


@dataclass
class EnvWorkerRunGroupRequest(BaseRequestResponse):
    example_id: int
    model_name: str  # required for multi-tenant run
    type: Literal["run_group"] = "run_group"


@dataclass
class EnvWorkerGetDatasetSizeRequest(BaseRequestResponse):
    type: Literal["get_dataset_size"] = "get_dataset_size"


EnvWorkerRequest = EnvWorkerRunGroupRequest | EnvWorkerGetDatasetSizeRequest


@dataclass
class EnvWorkerGetDatasetSizeResponse(BaseRequestResponse):
    dataset_size: int
    type: Literal["get_dataset_size"] = "get_dataset_size"


@dataclass
class EnvWorkerRunGroupResponse(BaseRequestResponse):
    serialized_state: list[dict]  # serialized vf.State
    type: Literal["run_group"] = "run_group"


EnvWorkerResponse = EnvWorkerGetDatasetSizeResponse | EnvWorkerRunGroupResponse


class EnvWorkerHelper:
    """Consumes env worker requests from a queue, runs rollouts and returns responses to a response queue. Used by EnvWorker which starts the helper in a separate process."""

    def __init__(
        self,
        env_config: TrainEnvConfig | EvalEnvConfig,
        request_queue: "Queue[EnvWorkerRequest]",
        response_queue: "Queue[EnvWorkerResponse]",
        num_examples,  # mp.Value
    ):
        self.ready = Event()
        self.env_config = env_config
        if env_config.type == "train":
            self.sampling_args = get_train_sampling_args(env_config.sampling)
        else:
            self.sampling_args = get_eval_sampling_args(env_config.sampling)
        print(self.sampling_args)

        self.process: Process | None = None
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.num_examples = num_examples

    def start(self):
        # install_env(self.env_config.id) # TODO: install env in subprocess

        # load environment
        env = vf.load_environment(self.env_config.id, **self.env_config.args)
        env.set_max_seq_len(self.env_config.seq_len)
        env.set_interleaved_rollouts(self.env_config.interleaved_rollouts)
        if self.env_config.type == "train":
            dataset = env.get_dataset()
        else:
            dataset = env.get_eval_dataset(n=self.env_config.num_examples)

        # Set dataset size in shared value for synchronous access from parent process
        self.num_examples.value = len(dataset)

        # create client
        client_config = ClientConfig(**self.env_config.client_config.model_dump())
        clients = setup_clients(client_config)
        client_cycle = cycle(clients)

        pending_requests: dict[asyncio.Task[EnvWorkerResponse], EnvWorkerRequest] = {}

        # create semaphore
        semaphore = (
            asyncio.Semaphore(self.env_config.max_concurrent)
            if self.env_config.max_concurrent > 0
            else asyncio.Semaphore(10000)
        )

        async def process_request(request: EnvWorkerRequest) -> EnvWorkerResponse:
            """Process a single rollout request."""

            def process_get_dataset_size_request(
                request: EnvWorkerGetDatasetSizeRequest,
            ) -> EnvWorkerGetDatasetSizeResponse:
                return EnvWorkerGetDatasetSizeResponse(request_id=request.request_id, dataset_size=len(dataset))

            async def process_run_group_request(request: EnvWorkerRunGroupRequest) -> EnvWorkerRunGroupResponse:
                client = next(client_cycle)
                example = dataset[request.example_id]
                group_inputs = [vf.RolloutInput(**example) for _ in range(self.env_config.rollouts_per_example)]

                states = await env.run_group(
                    group_inputs=group_inputs,
                    client=client,
                    model=request.model_name,
                    gen_sampling_args=self.sampling_args,
                    gen_sem=semaphore,
                    score_sem=semaphore,
                )

                serialized_state = [to_serializable_state(state) for state in states]
                return EnvWorkerRunGroupResponse(request_id=request.request_id, serialized_state=serialized_state)

            if request.type == "get_dataset_size":
                return process_get_dataset_size_request(request)
            elif request.type == "run_group":
                return await process_run_group_request(request)
            else:
                raise ValueError(f"Unknown request type: {request.type}")

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
        self.ready.set()
        asyncio.run(worker_loop())


class EnvWorker:
    """Proxies vf.Environment.run_group, but delegates work to EnvWorkerHelper in a separate process."""

    def __init__(self, env_config: TrainEnvConfig | EvalEnvConfig):
        self.env_config = env_config
        self.name = env_config.name or env_config.id

        self.request_queue: Queue[EnvWorkerRequest] = Queue()
        self.response_queue: Queue[EnvWorkerResponse] = Queue()
        self.num_examples = Value("i", 0)
        self.env_worker_helper = EnvWorkerHelper(
            env_config=env_config,
            request_queue=self.request_queue,
            response_queue=self.response_queue,
            num_examples=self.num_examples,
        )
        self.env_worker_process: Process | None = None

        self.response_collector_task: asyncio.Task | None = None
        self.pending_futures: dict[int, asyncio.Future] = {}
        self.next_request_id: int = 0

    def get_next_request_id(self) -> int:
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id

    def start(self):
        self.env_worker_process = Process(target=self.env_worker_helper.start, daemon=True)
        self.env_worker_process.start()
        self.env_worker_helper.ready.wait()
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
        model_name: str,
    ) -> list[dict]:
        request_id = self.get_next_request_id()
        request = EnvWorkerRunGroupRequest(
            request_id=request_id,
            example_id=example_id,
            model_name=model_name,
        )
        self.request_queue.put(request)
        future = asyncio.Future()
        self.pending_futures[request_id] = future

        return await future

    def get_dataset_size(self) -> int:
        """Returns the dataset size synchronously via shared memory."""
        assert self.env_worker_process and self.env_worker_process.is_alive()
        return self.num_examples.value

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
                        if response.type == "get_dataset_size":
                            future.set_result(response.dataset_size)
                        elif response.type == "run_group":
                            future.set_result(response.serialized_state)
                        else:
                            raise ValueError(f"Unknown response type: {response.type}")

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
        env_worker_config = EvalEnvConfig(id="gsm8k")
        env_worker = EnvWorker(env_worker_config)
        env_worker.start()

        dataset_size = env_worker.get_dataset_size()
        print(dataset_size)

        tasks = []
        for i in range(1):
            tasks.append(env_worker.run_group(example_id=i, model_name="Qwen/Qwen3-4B-Instruct-2507"))
        responses = await asyncio.gather(*tasks)
        for response in responses:
            print(response)

        # await env_worker.stop()

    asyncio.run(main())
