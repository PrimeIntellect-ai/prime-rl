import asyncio
import json

import httpx
import pytest

from prime_rl.orchestrator.clients import prefill_logprobs, prefill_logprobs_with_max


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
                "prompt_logprobs": [
                    None,
                    {"99": {"logprob": -0.1}, "2": {"logprob": -0.7}},
                    {"3": {"logprob": -0.3}},
                ],
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


@pytest.mark.parametrize("target_first", [True, False])
def test_prefill_target_and_maximum_are_independent_of_dictionary_order(target_first):
    pairs = [("2", {"logprob": -4.0}), ("99", {"logprob": -0.1})]
    entry = dict(pairs if target_first else reversed(pairs))
    client = _FakeOpenAIClient({"prompt_logprobs": [None, entry, {"3": {"logprob": -0.2}}]})
    assert asyncio.run(prefill_logprobs_with_max(client, "policy", [1, 2, 3])) == (
        [0.0, -4.0, -0.2],
        [0.0, -0.1, -0.2],
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt_logprobs": None},
        {"prompt_logprobs": []},
        {"prompt_logprobs": [None]},
        {"prompt_logprobs": [None, None]},
        {"prompt_logprobs": [None, {}]},
        {"prompt_logprobs": [None, {"99": {"logprob": -0.1}}]},
        {"prompt_logprobs": [None, {"2": {}}]},
        {"prompt_logprobs": [None, {"2": {"logprob": 0.1}}]},
        {"prompt_logprobs": [None, {"2": {"logprob": float("nan")}}]},
        {"prompt_logprobs": [None, {"2": {"logprob": -float("inf")}}]},
    ],
)
def test_prefill_rejects_incomplete_or_invalid_scores(payload):
    with pytest.raises(ValueError):
        asyncio.run(prefill_logprobs_with_max(_FakeOpenAIClient(payload), "policy", [1, 2]))


def test_prefill_rejects_empty_input_without_request():
    client = _FakeOpenAIClient({})
    with pytest.raises(ValueError, match="at least one token"):
        asyncio.run(prefill_logprobs_with_max(client, "policy", []))
    assert client.calls == []
