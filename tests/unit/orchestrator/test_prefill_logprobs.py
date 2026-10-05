import asyncio
import json

import httpx
from openai import AsyncOpenAI
from verifiers.v1.configs.client import TrainClientConfig

from prime_rl.orchestrator.clients import PrefillScorer, prefill_logprobs
from prime_rl.utils.weight_sync import WEIGHT_VERSION_HEADER


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict):
        # Match what AsyncOpenAI exposes — prefill_logprobs reads ``str(openai.base_url)``.
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


def test_prefill_logprobs_uses_inference_generate():
    async def _run():
        fake_openai = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [None, {"11": {"logprob": -0.7}}, {"12": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        result = await prefill_logprobs(fake_openai, "ref-model", [1, 2, 3])

        assert result == [0.0, -0.7, -0.3]
        assert fake_openai.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
                "body": {
                    "model": "ref-model",
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


def test_cached_prefill_client_refreshes_policy_version_headers():
    versions = []

    def respond(request):
        versions.append(request.headers[WEIGHT_VERSION_HEADER])
        return httpx.Response(
            200,
            json={
                "request_id": "score",
                "choices": [],
                "prompt_logprobs": [None, {"2": {"logprob": -0.7}}],
                "kv_transfer_params": None,
            },
        )

    async def run():
        scorer = PrefillScorer()
        scorer._client = AsyncOpenAI(
            base_url="http://test/v1",
            api_key="EMPTY",
            default_headers={WEIGHT_VERSION_HEADER: "1"},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        try:
            for version in ["1", "2"]:
                config = TrainClientConfig(
                    type="train",
                    base_url="http://test/v1",
                    headers={WEIGHT_VERSION_HEADER: version},
                )
                assert await scorer.score(config, "model", [1, 2]) == [0.0, -0.7]
            assert versions == ["1", "2"]
        finally:
            await scorer.aclose()

    asyncio.run(run())
