import asyncio

import verifiers as vf

from prime_rl.inference.vllm.serving_generate import GenerateResponse
from prime_rl.orchestrator import utils as orchestrator_utils
from prime_rl.transport import TrainingSample


class _FakeOpenAIClient:
    def __init__(self):
        self.calls = []

    async def post(self, path, *, cast_to, body):
        self.calls.append({"path": path, "cast_to": cast_to, "body": body})
        return GenerateResponse(
            id="gen-test",
            model=body["model"],
            prompt_token_ids=list(body["prompt_token_ids"]),
            choices=[],
            usage={"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
            prompt_logprobs=[None, -0.7, -0.3],
        )


def test_compute_teacher_logprobs_uses_generate_endpoint(monkeypatch):
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
                "path": "/generate",
                "cast_to": GenerateResponse,
                "body": {
                    "model": "teacher-model",
                    "prompt_token_ids": [1, 2, 3],
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "prompt_logprobs": True,
                },
            }
        ]

    asyncio.run(_run())
