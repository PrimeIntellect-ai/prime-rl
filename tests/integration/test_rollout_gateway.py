from __future__ import annotations

import asyncio
from typing import Any

import httpx
import verifiers as vf
from fastapi import FastAPI
from vllm.entrypoints.openai.api_server import router

import prime_rl.inference.vllm.rollout_gateway as rollout_gateway
from prime_rl.inference.vllm.rollout_gateway import RolloutRegistry
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.trainer.batch import prepare_batch


class FakeAsyncOpenAI:
    def __init__(self, base_url: str, api_key: str, max_retries: int):
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.turn_count = 0
        self.closed = False

    def copy(self, base_url: str) -> "FakeAsyncOpenAI":
        self.base_url = base_url
        return self

    async def close(self) -> None:
        self.closed = True

    async def post(self, path: str, body: dict[str, Any], cast_to: Any):
        if path == "/chat/completions":
            prompt_token_ids = [11, 12]
            completion_ids = [21]
            content = "reply-1"
        elif path == "/chat/completions/tokens":
            prompt_token_ids = list(body["tokens"])
            completion_ids = [22 + (self.turn_count - 1)]
            content = f"reply-{self.turn_count + 1}"
        elif path == "/tokenize":
            payload = self._tokenize_payload(body)
            return cast_to.model_validate(payload)
        else:
            raise AssertionError(f"Unexpected path: {path}")

        self.turn_count += 1
        payload = {
            "id": f"cmpl-{self.turn_count}",
            "object": "chat.completion",
            "created": 1_700_000_000 + self.turn_count,
            "model": body.get("model", "test-model"),
            "prompt_token_ids": prompt_token_ids,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                    "token_ids": completion_ids,
                    "logprobs": {
                        "content": [
                            {
                                "token": str(token_id),
                                "logprob": -0.1,
                                "bytes": None,
                                "top_logprobs": [],
                            }
                            for token_id in completion_ids
                        ]
                    },
                }
            ],
        }
        return cast_to.model_validate(payload)

    def _tokenize_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        if "prompt" in body:
            if body["prompt"] == "World!":
                tokens = [901]
            else:
                tokens = [999]
        else:
            messages = body.get("messages", [])
            if messages == [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World!"},
            ]:
                tokens = [101, 901]
            elif messages == [{"role": "user", "content": "Turn 2"}]:
                tokens = [102]
            elif messages == [{"role": "user", "content": "Turn 3"}]:
                tokens = [103]
            else:
                tokens = []
        return {
            "count": len(tokens),
            "max_model_len": 4096,
            "tokens": tokens,
        }


def _new_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.rollout_registry = RolloutRegistry(port=8000)
    return app


async def _post_chat(
    client: httpx.AsyncClient,
    rollout_id: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
) -> httpx.Response:
    return await client.post(
        f"/v1/rollouts/{rollout_id}/chat/completions",
        json={"messages": messages, "stream": stream},
    )


def test_rollout_gateway_end_to_end_tito_and_trajectory(monkeypatch) -> None:
    from verifiers.utils.token_utils import get_tokens_client

    monkeypatch.setattr(rollout_gateway, "AsyncOpenAI", FakeAsyncOpenAI)
    get_tokens_client.cache_clear()
    app = _new_test_app()

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            rollout_id = "test-rollout-001"
            register = await client.post(
                f"/v1/rollouts/{rollout_id}/register",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "sampling_params": {"max_completion_tokens": 32},
                    "max_turns": 10,
                },
            )
            assert register.status_code == 200

            turn_0 = [{"role": "user", "content": "Hello"}]
            turn_1 = turn_0 + [{"role": "assistant", "content": "reply-1"}, {"role": "user", "content": "Turn 2"}]
            turn_2 = turn_1 + [{"role": "assistant", "content": "reply-2"}, {"role": "user", "content": "Turn 3"}]

            resp0 = await _post_chat(client, rollout_id, turn_0)
            assert resp0.status_code == 200
            assert resp0.json()["choices"][0]["message"]["content"] == "reply-1"

            resp1 = await _post_chat(client, rollout_id, turn_1)
            assert resp1.status_code == 200
            assert resp1.json()["choices"][0]["message"]["content"] == "reply-2"

            resp2 = await _post_chat(client, rollout_id, turn_2)
            assert resp2.status_code == 200
            assert resp2.json()["choices"][0]["message"]["content"] == "reply-3"

            trajectory_response = await client.get(f"/v1/rollouts/{rollout_id}/trajectory")
            assert trajectory_response.status_code == 200
            payload = trajectory_response.json()

            assert payload["num_turns"] == 3
            assert payload["prompt"] == turn_0
            assert len(payload["trajectory"]) == 3

            trajectory = payload["trajectory"]
            assert trajectory[0]["tokens"]["prompt_ids"] == [11, 12]
            assert trajectory[1]["tokens"]["prompt_ids"] == [11, 12, 21, 102]
            assert trajectory[2]["tokens"]["prompt_ids"] == [11, 12, 21, 102, 22, 103]

            for step in trajectory:
                tokens = step["tokens"]
                assert tokens is not None
                assert len(tokens["completion_ids"]) > 0
                assert len(tokens["completion_logprobs"]) == len(tokens["completion_ids"])
                assert all(mask == 0 for mask in tokens["prompt_mask"])
                assert all(mask == 1 for mask in tokens["completion_mask"])
                vf.TrajectoryStep(**step)

            output = vf.RolloutOutput(
                example_id=0,
                task="gateway-test",
                prompt=payload["prompt"],
                completion=payload["completion"],
                trajectory=[
                    vf.TrajectoryStep(
                        response=None,
                        **step,
                    )
                    for step in trajectory
                ],
                reward=1.0,
                timing={"generation_ms": 1.0, "scoring_ms": 1.0, "total_ms": 2.0},
                is_completed=True,
                is_truncated=payload["is_truncated"],
                metrics={},
                sampling_args={"temperature": 1.0},
                error=None,
            )

            interleaved = interleave_rollout(output)
            assert interleaved is not None
            assert len(interleaved) >= 1

            packed = prepare_batch(
                rollouts=interleaved,
                seq_len=128,
                num_train_workers=1,
                idxs=[0] * len(interleaved),
                num_loras=1,
            )
            assert len(packed) == 1
            assert len(packed[0]) >= 1

            cancel = await client.post(f"/v1/rollouts/{rollout_id}/cancel")
            assert cancel.status_code == 200

            blocked = await _post_chat(client, rollout_id, turn_2)
            assert blocked.status_code == 409

            cancelled_trajectory = await client.get(f"/v1/rollouts/{rollout_id}/trajectory")
            assert cancelled_trajectory.status_code == 200

            unregister = await client.post(f"/v1/rollouts/{rollout_id}/unregister")
            assert unregister.status_code == 200

            missing = await _post_chat(client, rollout_id, turn_2)
            assert missing.status_code == 404

    asyncio.run(_run())


def test_rollout_gateway_streaming_response(monkeypatch) -> None:
    from verifiers.utils.token_utils import get_tokens_client

    monkeypatch.setattr(rollout_gateway, "AsyncOpenAI", FakeAsyncOpenAI)
    get_tokens_client.cache_clear()
    app = _new_test_app()

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            rollout_id = "test-rollout-stream"
            register = await client.post(
                f"/v1/rollouts/{rollout_id}/register",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "sampling_params": {"max_completion_tokens": 32},
                    "max_turns": 10,
                },
            )
            assert register.status_code == 200

            stream_resp = await _post_chat(
                client,
                rollout_id,
                [{"role": "user", "content": "Hello"}],
                stream=True,
            )
            assert stream_resp.status_code == 200
            assert stream_resp.headers["content-type"].startswith("text/event-stream")
            assert "chat.completion.chunk" in stream_resp.text
            assert "data: [DONE]" in stream_resp.text

            unregister = await client.post(f"/v1/rollouts/{rollout_id}/unregister")
            assert unregister.status_code == 200

    asyncio.run(_run())


def test_rollout_gateway_non_extension_turn_falls_back_to_chat(monkeypatch) -> None:
    from verifiers.utils.token_utils import get_tokens_client

    monkeypatch.setattr(rollout_gateway, "AsyncOpenAI", FakeAsyncOpenAI)
    get_tokens_client.cache_clear()
    app = _new_test_app()

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            rollout_id = "test-rollout-non-extension"
            register = await client.post(
                f"/v1/rollouts/{rollout_id}/register",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "sampling_params": {"max_completion_tokens": 32},
                    "max_turns": 10,
                },
            )
            assert register.status_code == 200

            turn_0 = [{"role": "user", "content": "Hello"}]
            non_extension_turn = [
                {"role": "system", "content": "Use concise output"},
                {"role": "user", "content": "Follow up"},
            ]

            resp0 = await _post_chat(client, rollout_id, turn_0)
            assert resp0.status_code == 200

            resp1 = await _post_chat(client, rollout_id, non_extension_turn)
            assert resp1.status_code == 200

            trajectory_response = await client.get(f"/v1/rollouts/{rollout_id}/trajectory")
            assert trajectory_response.status_code == 200
            payload = trajectory_response.json()

            assert payload["num_turns"] == 2
            assert payload["trajectory"][1]["tokens"]["prompt_ids"] == [11, 12]

            unregister = await client.post(f"/v1/rollouts/{rollout_id}/unregister")
            assert unregister.status_code == 200

    asyncio.run(_run())
