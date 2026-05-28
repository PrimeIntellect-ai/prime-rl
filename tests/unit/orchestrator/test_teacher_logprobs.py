import asyncio
import json

import httpx
import verifiers as vf

from prime_rl.orchestrator import utils as orchestrator_utils
from prime_rl.transport import TrainingSample


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict):
        # Match what AsyncOpenAI exposes — utils.py reads ``str(client.base_url)``.
        self.base_url = "http://fake-host:8000/v1"
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=200,
            content=json.dumps(self._payload).encode(),
            request=request,
        )


def test_compute_teacher_logprobs_uses_inference_generate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [None, {"11": {"logprob": -0.7}}, {"12": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)

        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
        )

        assert result == [[0.0, -0.7, -0.3]]
        assert fake_client.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
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

    asyncio.run(_run())
