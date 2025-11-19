"""ZMQ server that wraps a verifiers environment for remote execution."""

import asyncio
import functools
import json
import signal
import time
from pathlib import Path

import msgpack
import msgpack_numpy
import verifiers as vf
import zmq
import zmq.asyncio
from datasets import Dataset
from loguru import logger
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from prime_rl.orchestrator.utils import msgpack_encoder, parse_is_truncated_completions
from prime_rl.utils.logger import setup_logger

# Patch msgpack to use msgpack-numpy for efficient numpy array serialization
msgpack_numpy.patch()


class ZMQEnvironmentServer:
    """Server that wraps a verifiers environment and exposes it via ZMQ."""

    def __init__(
        self,
        bind_address: str,
        env_id: str,
        instance_name: str,
        env_args: dict | None = None,
        max_concurrent: int = 10,
    ):
        """
        Initialize environment server.

        Args:
            bind_address: ZMQ bind address (e.g., "tcp://*:5555")
            env_id: Environment ID to load
            instance_name: Unique instance name
            env_args: Environment initialization arguments
            max_concurrent: Max concurrent requests
        """
        self.bind_address = bind_address
        self.env_id = env_id
        self.instance_name = instance_name
        self.env_args = env_args or {}
        self.max_concurrent = max_concurrent
        self.start_time = time.time()

        # Load environment
        logger.info(f"Loading environment: {env_id} (instance: {instance_name})")
        self.env = vf.load_environment(env_id, **self.env_args)

        # Setup ZMQ
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind_address)

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Cache inference clients and tokenizers
        self.inference_clients: dict[str, AsyncOpenAI] = {}
        self.tokenizers = {}

    def _get_inference_client(self, endpoint: str) -> AsyncOpenAI:
        """
        Get or create inference client for endpoint.

        Uses singleton pattern: if endpoint changes, closes old client to prevent leaks.
        Typical usage: one worker connects to one inference endpoint at a time.
        """
        if endpoint in self.inference_clients:
            return self.inference_clients[endpoint]

        # New endpoint - close all old clients to prevent resource leaks
        for old_endpoint, old_client in list(self.inference_clients.items()):
            try:
                # AsyncOpenAI uses httpx.AsyncClient internally - close it
                asyncio.create_task(old_client.close())
                logger.debug(f"Closed old inference client for {old_endpoint}")
            except Exception as e:
                logger.warning(f"Error closing old inference client {old_endpoint}: {e}")

        self.inference_clients.clear()

        # Create new client
        self.inference_clients[endpoint] = AsyncOpenAI(
            base_url=endpoint,
            api_key="EMPTY",
            max_retries=10,
        )
        logger.info(f"Created new inference client for {endpoint}")
        return self.inference_clients[endpoint]

    def _get_tokenizer(self, processing_class: str):
        if processing_class not in self.tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(processing_class)
            # Ensure pad_token is set (required for batching)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug(f"Set pad_token to eos_token for {processing_class}")
            self.tokenizers[processing_class] = tokenizer
        return self.tokenizers[processing_class]

    async def run(self, stop_event: asyncio.Event | None = None):
        """
        Main server loop.

        Args:
            stop_event: Optional event to signal graceful shutdown
        """
        logger.info(f"Environment server listening on {self.bind_address}")
        logger.info(f"Instance: {self.instance_name}")

        # Create a task to wait for stop signal
        stop_task = asyncio.create_task(stop_event.wait()) if stop_event else None

        try:
            while True:
                # If stop_event is set, exit gracefully
                if stop_event and stop_event.is_set():
                    logger.info("Stop event received, shutting down gracefully")
                    break

                try:
                    # Receive with timeout to periodically check stop_event
                    frames = await asyncio.wait_for(self.socket.recv_multipart(), timeout=1.0 if stop_event else None)

                    if len(frames) != 3:
                        logger.warning(f"Invalid message: expected 3 frames, got {len(frames)}")
                        continue

                    client_id, request_id, payload_bytes = frames

                    # Process in background with concurrency limit
                    asyncio.create_task(self._process_request(client_id, request_id, payload_bytes))

                except asyncio.TimeoutError:
                    # Timeout is expected when checking for stop_event
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in server loop: {e}", exc_info=True)
        finally:
            if stop_task and not stop_task.done():
                stop_task.cancel()

    async def _process_request(self, client_id: bytes, request_id: bytes, payload_bytes: bytes):
        async with self.semaphore:
            try:
                # Deserialize request
                request = msgpack.unpackb(payload_bytes, raw=False)
                action = request.get("action")

                # Route to handler
                if action == "generate":
                    response = await self._handle_generate(request)
                elif action == "get_dataset":
                    response = await self._handle_get_dataset(request)
                elif action == "get_eval_dataset":
                    response = await self._handle_get_eval_dataset(request)
                elif action == "health":
                    response = await self._handle_health(request)
                else:
                    response = {"status": "error", "error": f"Unknown action: {action}"}

                # Serialize response - msgpack traverses in C, calls msgpack_encoder for unknown types
                response_bytes = msgpack.packb(response, default=msgpack_encoder, use_bin_type=True)

                # Send response: [client_id, request_id, response]
                await self.socket.send_multipart([client_id, request_id, response_bytes])

                logger.debug(f"Sent {action} response ({len(response_bytes)} bytes)")

            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)

                # Send error response
                error_response = {"status": "error", "error": str(e)}
                error_bytes = msgpack.packb(error_response, default=msgpack_encoder, use_bin_type=True)
                await self.socket.send_multipart([client_id, request_id, error_bytes])

    async def _handle_generate(self, request: dict) -> dict:
        """Handle generate request."""
        problem = request["problem"]
        model_name = request["model_name"]
        rollouts_per_example = request["rollouts_per_example"]
        sampling_args = request["sampling_args"]
        inference_endpoint = request["inference_endpoint"]

        # Processing parameters
        processing_class = request["processing_class"]  # Tokenizer name
        max_seq_len = request["max_seq_len"]
        mask_env_responses = request.get("mask_env_responses", True)
        zero_truncated_completions = request.get("zero_truncated_completions", False)
        mask_truncated_completions = request.get("mask_truncated_completions", False)

        logger.info(
            f"[{problem.get('example_id')}] Starting generation of {rollouts_per_example} rollouts using {inference_endpoint}"
        )

        # Get inference client
        client = self._get_inference_client(inference_endpoint)

        # Generate rollouts using environment
        generate_outputs: vf.GenerateOutputs = await self.env.generate(
            inputs=Dataset.from_list([problem] * rollouts_per_example),
            client=client,
            model=model_name,
            sampling_args=sampling_args,
        )

        logger.info(
            f"[{problem.get('example_id')}] Generated {len(generate_outputs.completion)} rollouts, "
            f"avg_reward={sum(generate_outputs.reward) / len(generate_outputs.reward):.3f}"
        )

        # Get tokenizer for processing
        tokenizer = self._get_tokenizer(processing_class)

        logger.info(f"[{problem.get('example_id')}] Processing environment results")

        # Offload CPU-bound processing to thread pool to avoid blocking event loop
        # This prevents micro-stalls that would block ZMQ heartbeats and new requests
        loop = asyncio.get_running_loop()
        processed_outputs: vf.ProcessedOutputs = await loop.run_in_executor(
            None,  # Uses default ThreadPoolExecutor
            functools.partial(
                self.env.process_env_results_vllm,
                prompts=generate_outputs.prompt,
                completions=generate_outputs.completion,
                states=generate_outputs.state,
                rewards=generate_outputs.reward,
                processing_class=tokenizer,
                max_seq_len=max_seq_len,
                mask_env_responses=mask_env_responses,
                zero_truncated_completions=zero_truncated_completions,
                mask_truncated_completions=mask_truncated_completions,
            ),
        )

        # Parse truncation info
        responses = [state["responses"] for state in generate_outputs.state]
        is_truncated = parse_is_truncated_completions(responses=responses)

        logger.info(f"[{problem.get('example_id')}] Serializing response for transmission")

        # Return processed data - msgpack will traverse lists/dicts in C
        # msgpack_encoder only called for special types (Path, Pydantic, etc.)
        return {
            "status": "success",
            "generate_outputs": {
                "prompt": generate_outputs.prompt,
                "completion": generate_outputs.completion,
                "answer": generate_outputs.answer,
                "state": generate_outputs.state,
                "reward": generate_outputs.reward,
                "info": generate_outputs.info,
                "task": generate_outputs.task,
                "metrics": generate_outputs.metrics,
                "example_id": generate_outputs.example_id,
                "metadata": generate_outputs.metadata,
            },
            "processed_outputs": {
                "prompt_ids": processed_outputs.prompt_ids,
                "completion_ids": processed_outputs.completion_ids,
                "prompt_mask": processed_outputs.prompt_mask,
                "completion_mask": processed_outputs.completion_mask,
                "completion_logprobs": processed_outputs.completion_logprobs,
                "rewards": processed_outputs.rewards,
            },
            "is_truncated": is_truncated,
        }

    async def _handle_get_dataset(self, request: dict) -> dict:
        seed = request["seed"]

        logger.info(f"Loading dataset with seed={seed}")
        dataset = self.env.get_dataset(seed=seed)

        # Convert to list of dicts
        dataset_list = [dict(example) for example in dataset]

        logger.info(f"Loaded {len(dataset_list)} problems")

        return {"status": "success", "dataset": dataset_list}

    async def _handle_get_eval_dataset(self, request: dict) -> dict:
        seed = request["seed"]

        logger.info(f"Loading eval dataset with seed={seed}")
        eval_dataset = self.env.get_eval_dataset(seed=seed)

        # Convert to list of dicts
        dataset_list = [dict(example) for example in eval_dataset]

        logger.info(f"Loaded {len(dataset_list)} eval problems")

        return {"status": "success", "dataset": dataset_list}

    async def _handle_health(self, request: dict) -> dict:
        return {
            "status": "healthy",
            "instance_name": self.instance_name,
            "env_id": self.env_id,
            "uptime_seconds": time.time() - self.start_time,
        }

    async def close(self):
        # Close all inference clients
        for endpoint, client in self.inference_clients.items():
            try:
                await client.close()
                logger.debug(f"Closed inference client for {endpoint}")
            except Exception as e:
                logger.warning(f"Error closing inference client {endpoint}: {e}")

        self.inference_clients.clear()

        # Close ZMQ socket
        self.socket.close()
        self.ctx.term()
        logger.info("Environment server closed")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ZMQ Environment Server")
    parser.add_argument("--bind", required=True, help="ZMQ bind address (e.g., tcp://*:5555)")
    parser.add_argument("--env-id", required=True, help="Environment ID to load")
    parser.add_argument("--instance-name", required=True, help="Unique instance name")
    parser.add_argument("--env-args", default="{}", help="Environment args as JSON")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--log-level", default="info", help="Logging level (e.g., info, debug)")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file (optional)")

    args = parser.parse_args()

    # Setup logger with the specified log level and optional log file
    # Use append mode for shared log files (multiple workers write to same file)
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logger(args.log_level, log_file=log_file, append=True)
    # Update module-level logger so class methods use the configured logger with file output
    globals()['logger'] = logger

    # Parse env args
    env_args = json.loads(args.env_args)

    server = ZMQEnvironmentServer(
        bind_address=args.bind,
        env_id=args.env_id,
        instance_name=args.instance_name,
        env_args=env_args,
        max_concurrent=args.max_concurrent,
    )

    # Setup graceful shutdown for SIGTERM (K8s, Docker, Slurm) and SIGINT (Ctrl+C)
    stop_event = asyncio.Event()

    def signal_handler(sig):
        sig_name = signal.Signals(sig).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        stop_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    try:
        await server.run(stop_event=stop_event)
    finally:
        logger.info("Cleaning up resources...")
        await server.close()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
