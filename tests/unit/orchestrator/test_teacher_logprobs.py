import asyncio

import verifiers as vf
from openai import AsyncOpenAI
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse
from vllm.logprobs import Logprob

from prime_rl.orchestrator import utils as orchestrator_utils
from prime_rl.transport import TrainingSample


class _FakeOpenAIClient:
    def __init__(self):
        self.calls = []
        # Match what AsyncOpenAI exposes — utils.py reads ``str(client.base_url)``.
        self.base_url = "http://fake-host:8000/v1"

    async def post(self, path, *, cast_to, body):
        self.calls.append({"path": path, "cast_to": cast_to, "body": body})
        return GenerateResponse(
            request_id="gen-test",
            choices=[],
            # Upstream wire shape: list[dict[token_id, Logprob] | None]
            prompt_logprobs=[None, {11: Logprob(-0.7)}, {12: Logprob(-0.3)}],
        )


def test_compute_teacher_logprobs_uses_inference_generate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient()
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)

        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[1.0, 1.0],
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
        )

        assert result == [[0.0, -0.7, -0.3]]
        assert fake_client.calls == [
            {
                "path": "http://fake-host:8000/inference/v1/generate",
                "cast_to": GenerateResponse,
                "body": {
                    "model": "teacher-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]

        # Guard against the AsyncOpenAI client double-prefixing /v1 onto our
        # absolute URL. The SDK skips the merge when the path passes
        # ``httpx.URL.is_relative_url`` as False; assert the resolved URL ends
        # up at the disagg endpoint, not at base_url + path.
        real = AsyncOpenAI(api_key="test", base_url=fake_client.base_url)
        resolved = real._prepare_url(fake_client.calls[0]["path"])
        assert str(resolved) == "http://fake-host:8000/inference/v1/generate"

    asyncio.run(_run())
