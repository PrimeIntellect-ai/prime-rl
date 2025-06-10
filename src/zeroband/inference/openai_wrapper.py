import asyncio
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, Request
from vllm import LLM, SamplingParams


@dataclass
class PendingRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    future: asyncio.Future
    timestamp: float


class MockOpenAIServer:
    """
    Real HTTP server that implements OpenAI v1 API endpoints.
    Accumulates async requests and processes them in batches via vLLM.
    """

    def __init__(self, llm: LLM, tokenizer, config):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = config.batch_size
        self.max_wait_time = config.max_wait_time

        # Request accumulation
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.request_queue = asyncio.Queue()

        # FastAPI app for HTTP endpoints
        self.app = FastAPI()
        self._setup_routes()

        # Background batch processor
        self.batch_processor_task = None

        # Thread management
        self._server_thread = None
        self._loop = None
        self._server = None

    def _extract_sampling_params(self, body: Dict) -> SamplingParams:
        """Extract vLLM sampling parameters from OpenAI request"""
        return SamplingParams(
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens", 4096),
            top_p=body.get("top_p", 1.0),
            n=body.get("n", 1),  # Support n parameter from OpenAI request
            logprobs=body.get("logprobs", 0),
            stop=body.get("stop", None),
        )

    def _setup_routes(self):
        """Setup OpenAI-compatible HTTP endpoints"""

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()

            # Extract OpenAI request format
            messages = body["messages"]

            # Convert to vLLM format
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            sampling_params = self._extract_sampling_params(body)

            return await self._handle_completion_request(prompt, sampling_params)

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            body = await request.json()

            # Extract OpenAI request format
            prompt = body["prompt"]
            sampling_params = self._extract_sampling_params(body)

            return await self._handle_completion_request(prompt, sampling_params)

    async def _handle_completion_request(self, prompt: str, sampling_params: SamplingParams):
        """Handle individual completion request - queues for batching"""
        request_id = str(uuid.uuid4())

        # Create future for this request
        future = asyncio.Future()

        # Create pending request
        pending_req = PendingRequest(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params, future=future, timestamp=time.time()
        )

        # Add to pending requests and queue
        self.pending_requests[request_id] = pending_req
        await self.request_queue.put(request_id)

        # Wait for batch processing to complete
        try:
            response = await future
            return response
        except Exception as e:
            # Clean up on error
            self.pending_requests.pop(request_id, None)
            raise e

    async def _batch_processor(self):
        """Main batch processing loop - continuously processes requests"""
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
                else:
                    # No requests - small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                # Graceful shutdown
                break
            except Exception:
                # Log error and continue
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[PendingRequest]:
        """Collect requests for batching - waits for quiet period"""
        batch = []

        # Wait for first request
        try:
            request_id = await self.request_queue.get()
            if request_id in self.pending_requests:
                batch.append(self.pending_requests[request_id])
        except Exception:
            return batch

        # Keep collecting while new requests arrive within 0.1s
        while len(batch) < self.batch_size:
            try:
                # Wait 0.1s for another request
                request_id = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)

                # Got a new request - add it and continue waiting
                if request_id in self.pending_requests:
                    batch.append(self.pending_requests[request_id])

            except asyncio.TimeoutError:
                # No new request in 0.1s - process the batch
                break

        return batch

    async def _process_batch(self, batch: List[PendingRequest]):
        """Process a batch of requests through vLLM"""
        if not batch:
            return

        # Extract prompts and sampling params
        prompts = [req.prompt for req in batch]
        # Use first request's sampling params (could be more sophisticated)
        sampling_params = batch[0].sampling_params

        try:
            # Generate responses via vLLM
            outputs = await asyncio.to_thread(self.llm.generate, prompts, sampling_params, use_tqdm=False)

            # Distribute responses back to individual requests
            for request, output in zip(batch, outputs):
                response = self._format_openai_response(output, request.request_id)

                # Complete the future
                if not request.future.done():
                    request.future.set_result(response)

                # Clean up
                self.pending_requests.pop(request.request_id, None)

        except Exception as e:
            # Handle batch processing errors
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
                self.pending_requests.pop(request.request_id, None)

    def _format_openai_response(self, vllm_output, request_id: str) -> Dict:
        """Convert vLLM output to OpenAI API response format"""
        # Handle multiple completions (n > 1)
        choices = []
        total_completion_tokens = 0

        for i, output in enumerate(vllm_output.outputs):
            choices.append({"index": i, "message": {"role": "assistant", "content": output.text}, "finish_reason": output.finish_reason})
            total_completion_tokens += len(output.token_ids)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": len(vllm_output.prompt_token_ids),
                "completion_tokens": total_completion_tokens,
                "total_tokens": len(vllm_output.prompt_token_ids) + total_completion_tokens,
            },
        }

    def get_client_config(self, port: int = 8000) -> Dict[str, str]:
        """Get configuration for OpenAI client"""
        return {
            "base_url": f"http://localhost:{port}/v1",
            "api_key": "dummy-key",  # Not used but required by OpenAI client
        }

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the HTTP server and batch processor in a background thread (synchronous call)"""

        def run_server():
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Run the async server startup
            self._loop.run_until_complete(self._start_server_async(host, port))

        # Start server in background thread
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait a moment for server to start
        time.sleep(1.0)

    async def _start_server_async(self, host: str, port: int):
        """Internal async method to start the server"""
        # Start background batch processor
        self.batch_processor_task = asyncio.create_task(self._batch_processor())

        # Start HTTP server
        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        await self._server.serve()

    def shutdown(self):
        """Shutdown the server and cleanup resources"""
        if self._loop and self._server:
            # Cancel batch processor
            if self.batch_processor_task:
                self._loop.call_soon_threadsafe(self.batch_processor_task.cancel)

            # Shutdown server
            self._loop.call_soon_threadsafe(setattr, self._server, "should_exit", True)

            # Wait for thread to finish
            if self._server_thread:
                self._server_thread.join(timeout=5.0)
