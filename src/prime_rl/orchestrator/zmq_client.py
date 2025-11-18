"""ZMQ client for communicating with remote environment workers."""

import asyncio
import uuid
from typing import Any

import msgpack
import zmq
import zmq.asyncio
from loguru import logger

from prime_rl.orchestrator.utils import serialize_for_msgpack


class ZMQEnvironmentClient:
    """Client for remote environment communication via ZMQ."""

    def __init__(self, endpoints: list[str], timeout: float = 60.0):
        """
        Initialize ZMQ client.

        Args:
            endpoints: List of ZMQ endpoints (e.g., ["tcp://localhost:5555"])
            timeout: Request timeout in seconds
        """
        self.endpoints = endpoints
        self.timeout = timeout
        self.ctx = zmq.asyncio.Context()

        # DEALER socket for async request/response
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)

        # Connect to all endpoints
        for endpoint in endpoints:
            logger.debug(f"Connecting ZMQ client to {endpoint}")
            self.socket.connect(endpoint)

        self.pending: dict[bytes, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None

    async def start(self):
        self._receiver_task = asyncio.create_task(self._receive_loop())
        logger.debug(f"ZMQ client started with {len(self.endpoints)} endpoint(s)")

    async def _receive_loop(self):
        """Continuously receive responses from environment servers."""
        while True:
            try:
                # Receive multipart: [request_id, payload]
                msg = await self.socket.recv_multipart()
                request_id, response_data = msg[0], msg[1]

                if request_id in self.pending:
                    future = self.pending.pop(request_id)
                    if not future.done():
                        future.set_result(msgpack.unpackb(response_data))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ZMQ receive loop: {e}")

    async def _send_request(self, action: str, **kwargs) -> dict:
        """
        Send request to environment.

        Args:
            action: Action type (generate, get_dataset, etc.)
            **kwargs: Action-specific parameters

        Returns:
            Response dict
        """
        # Auto-start receiver if not already running
        if self._receiver_task is None:
            await self.start()

        request_id = uuid.uuid4().bytes

        # Serialize the request payload to handle numpy types, etc.
        request_payload = serialize_for_msgpack({"action": action, **kwargs})
        payload_bytes = msgpack.packb(request_payload, use_bin_type=True)

        future = asyncio.Future()
        self.pending[request_id] = future

        await self.socket.send_multipart([request_id, payload_bytes])

        try:
            response = await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self.pending.pop(request_id, None)
            raise TimeoutError(f"Environment timeout for action: {action}")

        if response.get("status") == "error":
            raise RuntimeError(f"Environment error: {response.get('error')}")

        return response

    async def generate(
        self,
        problem: dict,
        model_name: str,
        rollouts_per_example: int,
        sampling_args: dict,
        inference_endpoint: str,
        processing_class: str,
        max_seq_len: int,
        mask_env_responses: bool = True,
        zero_truncated_completions: bool = False,
        mask_truncated_completions: bool = False,
    ) -> dict:
        """
        Generate and process rollouts for a problem.

        Args:
            problem: Problem dict
            model_name: Model identifier
            rollouts_per_example: Number of completions to generate
            sampling_args: Sampling configuration
            inference_endpoint: vLLM endpoint URL
            processing_class: Tokenizer name for processing
            max_seq_len: Maximum sequence length
            mask_env_responses: Whether to mask environment responses
            zero_truncated_completions: Whether to zero out truncated completion rewards
            mask_truncated_completions: Whether to mask truncated completions

        Returns:
            Dict with "generate_outputs", "processed_outputs", "is_truncated"
        """
        return await self._send_request(
            action="generate",
            problem=problem,
            model_name=model_name,
            rollouts_per_example=rollouts_per_example,
            sampling_args=sampling_args,
            inference_endpoint=inference_endpoint,
            processing_class=processing_class,
            max_seq_len=max_seq_len,
            mask_env_responses=mask_env_responses,
            zero_truncated_completions=zero_truncated_completions,
            mask_truncated_completions=mask_truncated_completions,
        )

    async def get_dataset(self, seed: int) -> list[dict]:
        """
        Get training dataset from environment.

        Args:
            seed: Random seed

        Returns:
            List of problem dicts
        """
        response = await self._send_request(action="get_dataset", seed=seed)
        return response["dataset"]

    async def get_eval_dataset(self, seed: int) -> list[dict]:
        """
        Get evaluation dataset from environment.

        Args:
            seed: Random seed

        Returns:
            List of problem dicts
        """
        response = await self._send_request(action="get_eval_dataset", seed=seed)
        return response["dataset"]

    async def close(self):
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass

        self.socket.close()
        self.ctx.term()
        logger.debug("ZMQ client closed")
