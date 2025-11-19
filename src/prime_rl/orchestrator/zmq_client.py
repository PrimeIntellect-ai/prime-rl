"""ZMQ client for communicating with remote environment workers."""

import asyncio
import uuid

import msgpack
import msgpack_numpy
import zmq
import zmq.asyncio
from loguru import logger
from verifiers.types import GenerateOutputs, ProcessedOutputs

from prime_rl.orchestrator.utils import msgpack_encoder

# Patch msgpack to use msgpack-numpy for efficient numpy array serialization
msgpack_numpy.patch()


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

        # TCP keepalive for faster dead server detection
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)  # Start probes after 10s idle
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 2)  # Probe every 2s
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)  # Give up after 3 failed probes

        # Connect to all endpoints
        for endpoint in endpoints:
            logger.debug(f"Connecting ZMQ client to {endpoint}")
            self.socket.connect(endpoint)

        self.pending: dict[bytes, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None

    async def start(self):
        self._receiver_task = asyncio.create_task(self._receive_loop())
        logger.debug(f"ZMQ client started with {len(self.endpoints)} endpoint(s)")

    def _fail_all_pending(self, reason: str):
        """Fail all pending futures with the given reason."""
        for request_id, future in list(self.pending.items()):
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.pending.clear()

    async def _receive_loop(self):
        """Continuously receive responses from environment servers."""
        while True:
            try:
                # Receive multipart: [request_id, payload]
                msg = await self.socket.recv_multipart()

                if len(msg) < 2:
                    logger.error(f"Invalid message format: expected 2 frames, got {len(msg)}")
                    continue

                request_id, response_data = msg[0], msg[1]

                if request_id in self.pending:
                    future = self.pending.pop(request_id)
                    if not future.done():
                        try:
                            response = msgpack.unpackb(response_data, raw=False)
                            future.set_result(response)
                        except Exception as unpack_error:
                            # Unpacking failed - fail the specific future
                            logger.error(f"Failed to unpack response for request {request_id.hex()}: {unpack_error}")
                            future.set_exception(RuntimeError(f"Failed to deserialize response: {unpack_error}"))
                else:
                    logger.warning(f"Received response for unknown request_id: {request_id.hex()}")

            except asyncio.CancelledError:
                break
            except zmq.ZMQError as e:
                # Socket-level error - fail all pending futures and exit
                logger.error(f"ZMQ socket error in receive loop: {e}")
                self._fail_all_pending(f"ZMQ socket error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in ZMQ receive loop: {e}", exc_info=True)
                # Don't break - log and continue for non-socket errors

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

        # Let msgpack traverse dicts/lists in C - only call msgpack_encoder for unknown types
        payload_bytes = msgpack.packb(
            {"action": action, **kwargs},
            default=msgpack_encoder,
            use_bin_type=True,
        )

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
    ) -> tuple[GenerateOutputs, ProcessedOutputs, list[bool]]:
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
            Tuple containing (GenerateOutputs, ProcessedOutputs, is_truncated list)
        """
        response = await self._send_request(
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

        # Reconstruct GenerateOutputs from ZMQ response
        gen_out = response["generate_outputs"]
        # Only include metadata if it's not None (Pydantic validation requirement)
        kwargs = {
            "prompt": gen_out["prompt"],
            "completion": gen_out["completion"],
            "answer": gen_out["answer"],
            "state": gen_out["state"],
            "reward": gen_out["reward"],
            "info": gen_out["info"],
            "task": gen_out["task"],
            "metrics": gen_out["metrics"],
            "example_id": gen_out["example_id"],
        }
        if gen_out.get("metadata") is not None:
            kwargs["metadata"] = gen_out["metadata"]

        # Use model_construct to skip validation overhead for trusted ZMQ data
        generate_outputs = GenerateOutputs.model_construct(**kwargs)

        # Reconstruct ProcessedOutputs from ZMQ response
        proc_out = response["processed_outputs"]
        is_truncated = response["is_truncated"]

        # Use model_construct to skip validation overhead for trusted ZMQ data
        processed_outputs = ProcessedOutputs.model_construct(
            prompt_ids=proc_out["prompt_ids"],
            completion_ids=proc_out["completion_ids"],
            prompt_mask=proc_out["prompt_mask"],
            completion_mask=proc_out["completion_mask"],
            completion_logprobs=proc_out["completion_logprobs"],
            rewards=proc_out["rewards"],
            is_truncated=is_truncated,
        )

        return generate_outputs, processed_outputs, is_truncated

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
