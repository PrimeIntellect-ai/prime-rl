import asyncio
import base64
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from prime_rl.inference.dynamo.worker import (
    _completion_logprobs,
    _extract_prompt_logprobs,
    _generate,
)


def test_extract_prompt_logprobs_prefers_supplied_token_id():
    prompt_logprobs = [
        None,
        {
            7: SimpleNamespace(logprob=-7.0),
            42: SimpleNamespace(logprob=-0.42),
        },
        {
            99: {"logprob": -0.99},
        },
    ]

    assert _extract_prompt_logprobs(prompt_logprobs, [1, 42, 99]) == [None, -0.42, -0.99]


def test_completion_logprobs_align_to_generated_token_ids():
    output = SimpleNamespace(
        token_ids=[5, 6],
        logprobs=[
            {5: SimpleNamespace(logprob=-0.5)},
            {7: SimpleNamespace(logprob=-0.7), 6: SimpleNamespace(logprob=-0.6)},
        ],
    )

    assert _completion_logprobs(output) == [-0.5, -0.6]


def test_logprobs_do_not_fall_back_to_unmatched_token():
    assert _extract_prompt_logprobs([{7: SimpleNamespace(logprob=-0.7)}], [42]) == [None]

    output = SimpleNamespace(token_ids=[42], logprobs=[{7: SimpleNamespace(logprob=-0.7)}])
    assert _completion_logprobs(output) == [0.0]


def test_generate_uses_tokens_input_with_cache_salt():
    calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.logprobs = None
            self.prompt_logprobs = None
            self.skip_special_tokens = True

    engine_module = ModuleType("vllm.inputs.engine")

    def fake_tokens_input(prompt_token_ids, cache_salt=None):
        return {"prompt_token_ids": prompt_token_ids, "cache_salt": cache_salt}

    engine_module.tokens_input = fake_tokens_input
    sampling_module = ModuleType("vllm.sampling_params")
    sampling_module.SamplingParams = FakeSamplingParams

    class FakeRoutedExperts:
        shape = (1, 2)

        def tobytes(self):
            return b"abc"

    class FakeEngineClient:
        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            calls.append(
                {
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "request_id": request_id,
                    "kwargs": kwargs,
                }
            )
            yield SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                prompt_logprobs=[None, {2: SimpleNamespace(logprob=-0.2)}],
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=[3],
                        logprobs=[{3: SimpleNamespace(logprob=-0.3)}],
                        finish_reason="length",
                        routed_experts=FakeRoutedExperts(),
                    )
                ],
            )

    handler = SimpleNamespace(
        default_sampling_params={},
        config=SimpleNamespace(model="served/model"),
        engine_client=FakeEngineClient(),
        _to_local_dp_rank=lambda dp_rank: dp_rank,
        _resolve_lora_request=lambda _model: None,
    )

    with patch.dict(
        sys.modules,
        {
            "vllm": ModuleType("vllm"),
            "vllm.inputs": ModuleType("vllm.inputs"),
            "vllm.inputs.engine": engine_module,
            "vllm.sampling_params": sampling_module,
        },
    ):
        response = asyncio.run(
            _generate(
                handler,
                {
                    "model": "served/model",
                    "prompt_token_ids": [1, 2],
                    "prompt_logprobs": True,
                    "max_tokens": 1,
                    "cache_salt": "42",
                    "priority": 7,
                    "routing": {"dp_rank": 3},
                },
            )
        )

    assert calls[0]["prompt"] == {"prompt_token_ids": [1, 2], "cache_salt": "42"}
    assert calls[0]["sampling_params"].logprobs == 1
    assert calls[0]["sampling_params"].prompt_logprobs == 1
    assert calls[0]["sampling_params"].skip_special_tokens is False
    assert calls[0]["kwargs"]["data_parallel_rank"] == 3
    assert calls[0]["kwargs"]["priority"] == -7
    assert response["prompt_logprobs"] == [None, -0.2]
    assert response["choices"][0]["logprobs"] == [-0.3]
    assert response["choices"][0]["routed_experts"] == {
        "data": base64.b85encode(b"abc").decode("ascii"),
        "shape": [1, 2],
    }


def test_dynamo_proxy_generate_preserves_data_parallel_header():
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace()

    with patch.dict(sys.modules, {"transformers": transformers_module}):
        from prime_rl.inference.dynamo.proxy import DynamoProxy

    captured = {}

    class FakeClient:
        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return SimpleNamespace(content=b"{}", status_code=200, headers={"content-type": "application/json"})

    class FakeRequest:
        headers = {"x-data-parallel-rank": "4"}

        async def json(self):
            return {"model": "served/model", "prompt_token_ids": [1, 2]}

    proxy = DynamoProxy.__new__(DynamoProxy)
    proxy.worker_url = "http://worker"
    proxy.client = FakeClient()

    response = asyncio.run(proxy.generate(FakeRequest()))

    assert response.status_code == 200
    assert captured["url"] == "http://worker/engine/generate"
    assert captured["json"]["routing"]["dp_rank"] == 4


def test_dynamo_proxy_exposes_generate_route():
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace()

    with patch.dict(sys.modules, {"transformers": transformers_module}):
        from prime_rl.inference.dynamo.proxy import build_app

    proxy = SimpleNamespace(close=lambda: None)
    app = build_app(proxy)

    post_routes = {route.path for route in app.routes if "POST" in getattr(route, "methods", set())}

    assert "/v1/generate" in post_routes
