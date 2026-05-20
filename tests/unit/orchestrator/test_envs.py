import asyncio
from collections import Counter, defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prime_rl.orchestrator.envs import EvalEnv


class _FakeMonitor:
    def log(self, *_args, **_kwargs):
        pass

    def log_eval_samples(self, *_args, **_kwargs):
        pass


class _FakeEvalEnv(EvalEnv):
    @property
    def requires_group_scoring(self) -> bool:
        return False


def _rollout(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "reward": 1.0,
        "completion": "ok",
        "is_truncated": False,
        "error": None,
        "trajectory": [
            {
                "tokens": {"prompt_ids": [1], "completion_ids": [2]},
                "response": {},
            }
        ],
    }


def test_eval_dynamic_refill_reuses_fast_client_without_exceeding_window():
    async def run() -> None:
        env = _FakeEvalEnv.__new__(_FakeEvalEnv)
        env.config = SimpleNamespace(
            resolved_name="fake-eval",
            rollouts_per_example=1,
            max_concurrent_rollouts_per_client=1,
        )
        env.examples = [{"id": str(i), "example_id": str(i)} for i in range(8)]

        clients = [
            SimpleNamespace(api_base_url="http://slow/v1", extra_headers={}),
            SimpleNamespace(api_base_url="http://fast/v1", extra_headers={}),
        ]
        active = defaultdict(int)
        max_active = defaultdict(int)
        completed = Counter()

        async def run_rollout(client, example, **_kwargs):
            active[client.api_base_url] += 1
            max_active[client.api_base_url] = max(max_active[client.api_base_url], active[client.api_base_url])
            await asyncio.sleep(0.03 if "slow" in client.api_base_url else 0.001)
            active[client.api_base_url] -= 1
            completed[client.api_base_url] += 1
            return _rollout(example["id"])

        async def unexpected_get_client():
            raise AssertionError("dynamic eval should use eval_clients directly")

        env.run_rollout = run_rollout

        with patch("prime_rl.utils.monitor.get_monitor", return_value=_FakeMonitor()):
            outputs = await env.evaluate(
                model_name="model",
                get_client=unexpected_get_client,
                ckpt_step=0,
                step=0,
                cache_salt="0",
                eval_clients=clients,
            )

        assert len(outputs) == 8
        assert max_active["http://slow/v1"] <= 1
        assert max_active["http://fast/v1"] <= 1
        assert completed["http://fast/v1"] > completed["http://slow/v1"]

    asyncio.run(run())


def test_eval_dynamic_refill_requires_explicit_eval_clients():
    async def run() -> None:
        env = _FakeEvalEnv.__new__(_FakeEvalEnv)
        env.config = SimpleNamespace(
            resolved_name="fake-eval",
            rollouts_per_example=1,
            max_concurrent_rollouts_per_client=1,
        )
        env.examples = [{"id": "0", "example_id": "0"}]

        async def get_client():
            return SimpleNamespace(api_base_url="http://fallback/v1", extra_headers={})

        async def run_rollout(*_args, **_kwargs):
            raise AssertionError("dynamic eval should not fall back to get_client")

        env.run_rollout = run_rollout

        with patch("prime_rl.utils.monitor.get_monitor", return_value=_FakeMonitor()):
            with pytest.raises(RuntimeError, match="requires at least one eval client"):
                await env.evaluate(
                    model_name="model",
                    get_client=get_client,
                    ckpt_step=0,
                    step=0,
                    cache_salt="0",
                    eval_clients=None,
                )

    asyncio.run(run())
