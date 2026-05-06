import asyncio

from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput

from prime_rl.inference.vllm.serving_generate import GenerateRequest, OpenAIServingGenerate


class _FakeChatHandler:
    def __init__(self):
        self.lora_request = object()

    def _get_data_parallel_rank(self, raw_request):
        assert raw_request.headers["X-data-parallel-rank"] == "3"
        return 3

    async def _get_trace_headers(self, headers):
        assert headers["traceparent"] == "trace"
        return {"traceparent": headers["traceparent"]}

    def _maybe_get_adapters(self, request):
        assert request.model == "test-model"
        return self.lora_request


class _FakeEngineClient:
    def __init__(self):
        self.calls = []

    async def generate(self, engine_prompt, sampling_params, request_id, **kwargs):
        self.calls.append(
            {
                "engine_prompt": engine_prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
                "kwargs": kwargs,
            }
        )
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=list(engine_prompt["prompt_token_ids"]),
            prompt_logprobs=[None, {11: Logprob(-0.2)}, {12: Logprob(-0.4)}],
            outputs=[
                CompletionOutput(
                    index=0,
                    text="done",
                    token_ids=[99],
                    cumulative_logprob=-0.6,
                    logprobs=[{99: Logprob(-0.6)}],
                    finish_reason="stop",
                )
            ],
            finished=True,
        )


class _FakeRawRequest:
    def __init__(self):
        self.headers = {"X-data-parallel-rank": "3", "traceparent": "trace"}

    async def is_disconnected(self):
        return False


def test_generate_returns_prompt_logprobs_and_forwards_request_metadata():
    async def _run():
        engine_client = _FakeEngineClient()
        handler = OpenAIServingGenerate(engine_client, chat_handler=_FakeChatHandler())
        request = GenerateRequest(
            model="test-model",
            prompt_token_ids=[10, 11, 12],
            max_tokens=1,
            prompt_logprobs=True,
            priority=7,
        )

        response = await handler.generate(request, _FakeRawRequest())

        assert response.prompt_token_ids == [10, 11, 12]
        assert response.prompt_logprobs == [None, -0.2, -0.4]
        assert response.choices[0].token_ids == [99]
        assert response.choices[0].logprobs == [-0.6]

        call = engine_client.calls[0]
        assert call["engine_prompt"] == {"type": "token", "prompt_token_ids": [10, 11, 12]}
        assert call["sampling_params"].prompt_logprobs == 1
        assert call["kwargs"]["lora_request"] is handler.chat_handler.lora_request
        assert call["kwargs"]["priority"] == 7
        assert call["kwargs"]["data_parallel_rank"] == 3
        assert call["kwargs"]["trace_headers"] == {"traceparent": "trace"}

    asyncio.run(_run())
