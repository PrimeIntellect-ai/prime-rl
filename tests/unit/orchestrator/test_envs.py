import asyncio
from collections import Counter, defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prime_rl.configs.orchestrator import EvalEnvConfig, EvalSamplingConfig, TrainSamplingConfig
from prime_rl.orchestrator.envs import Env, EvalEnv


class _FakeMonitor:
    def log(self, *_args, **_kwargs):
        pass

    def log_eval_samples(self, *_args, **_kwargs):
        pass


class _FakeEvalEnv(EvalEnv):
    @property
    def requires_group_scoring(self) -> bool:
        return False


class _FakeDataset:
    def to_list(self) -> list[dict]:
        return [{"example_id": "0"}]


def _rollout(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "reward": 1.0,
        "completion": "ok",
        "is_truncated": False,
        "error": None,
        "token_usage": {"final_output_tokens": 1},
        "trajectory": [
            {
                "tokens": {"prompt_ids": [1], "completion_ids": [2]},
                "response": {},
            }
        ],
    }


def test_eval_env_uses_config_seed_for_eval_dataset():
    env = SimpleNamespace(calls=[])

    def get_eval_dataset(n=-1, seed=None):
        env.calls.append((n, seed))
        return _FakeDataset()

    env.get_eval_dataset = get_eval_dataset

    with patch("prime_rl.orchestrator.envs.vf.load_environment", return_value=env):
        _FakeEvalEnv(EvalEnvConfig(id="fake", num_examples=3, seed=42))

    assert env.calls == [(3, 42)]


def test_train_sampling_top_p_reaches_sampling_args():
    sampling_args = TrainSamplingConfig(top_p=0.95).to_sampling_args()

    assert sampling_args["top_p"] == 0.95


def test_train_sampling_retains_vllm_thinking_knobs():
    sampling_args = TrainSamplingConfig(
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        thinking_token_budget=8192,
    ).to_sampling_args()

    assert sampling_args["top_p"] == 0.95
    assert sampling_args["presence_penalty"] == 1.5
    assert sampling_args["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "thinking_token_budget": 8192,
    }


def test_eval_sampling_retains_vllm_thinking_knobs():
    sampling_args = EvalSamplingConfig(
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        thinking_token_budget=8192,
    ).to_sampling_args()

    assert sampling_args["presence_penalty"] == 1.5
    assert sampling_args["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "thinking_token_budget": 8192,
    }


def test_eval_dynamic_refill_reuses_fast_client_without_exceeding_window():
    async def run() -> None:
        env = _FakeEvalEnv.__new__(_FakeEvalEnv)
        env.config = SimpleNamespace(
            resolved_name="fake-eval",
            group_size=1,
            max_concurrent_rollouts_per_client=1,
            state_columns=[],
            max_retries=3,
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

        with patch("prime_rl.orchestrator.envs.get_monitor", return_value=_FakeMonitor()):
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
            group_size=1,
            max_concurrent_rollouts_per_client=1,
            state_columns=[],
            max_retries=3,
        )
        env.examples = [{"id": "0", "example_id": "0"}]

        async def get_client():
            return SimpleNamespace(api_base_url="http://fallback/v1", extra_headers={})

        async def run_rollout(*_args, **_kwargs):
            raise AssertionError("dynamic eval should not fall back to get_client")

        env.run_rollout = run_rollout

        with patch("prime_rl.orchestrator.envs.get_monitor", return_value=_FakeMonitor()):
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


def test_fixed_member_sampling_args_strip_learner_only_fields_without_mutating_learner_args():
    env = Env.__new__(Env)
    env.sampling_args = {
        "temperature": 1.0,
        "logprobs": True,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "return_token_ids": True,
            "cache_salt": "salt-1",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

    fixed = env._fixed_member_sampling_args()

    # Learner-only fields (logprobs, return_token_ids, top_k, min_p,
    # cache_salt) are stripped; portable defaults and user extras survive.
    assert fixed == {
        "temperature": 1.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    # The learner's sampling args are untouched...
    assert env.sampling_args["logprobs"] is True
    assert env.sampling_args["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "return_token_ids": True,
        "cache_salt": "salt-1",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # ...and the returned extra_body is a copy, not an alias.
    fixed["extra_body"]["presence_penalty"] = 1.5
    assert "presence_penalty" not in env.sampling_args["extra_body"]


def test_fixed_member_sampling_args_drop_extra_body_when_only_learner_fields_remain():
    env = Env.__new__(Env)
    env.sampling_args = {
        "temperature": 1.0,
        "logprobs": True,
        "extra_body": {"top_k": -1, "min_p": 0.0, "return_token_ids": True},
    }

    assert env._fixed_member_sampling_args() == {"temperature": 1.0}
