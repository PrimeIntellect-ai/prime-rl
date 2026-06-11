from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prime_rl.configs.orchestrator import EvalEnvConfig, EvalSamplingConfig, TrainSamplingConfig
from prime_rl.orchestrator.envs import Env, EvalEnv


class _FakeDataset:
    def to_list(self) -> list[dict]:
        return [{"example_id": "0"}]


def test_eval_env_uses_config_seed_for_eval_dataset():
    env = SimpleNamespace(calls=[])

    def get_eval_dataset(n=-1, seed=None):
        env.calls.append((n, seed))
        return _FakeDataset()

    env.get_eval_dataset = get_eval_dataset

    with patch("prime_rl.orchestrator.envs.vf.load_environment", return_value=env):
        EvalEnv(EvalEnvConfig(id="fake", num_examples=3, seed=42))

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


def _debate_eval_env(name: str = "debate-eval", members: tuple[str, ...] = ("debater_a", "debater_b", "judge")):
    from verifiers.protocols.debate import DebateEnv

    inner = DebateEnv.__new__(DebateEnv)
    inner.is_multi_agent = True
    inner.members = list(members)
    env = EvalEnv.__new__(EvalEnv)
    env.config = SimpleNamespace(resolved_name=name)
    env._env = inner
    return env


def _eval_envs(*envs):
    from prime_rl.orchestrator.envs import EvalEnvs

    collection = EvalEnvs.__new__(EvalEnvs)
    collection._envs = {env.name: env for env in envs}
    return collection


def _judge_fixed_config():
    from prime_rl.configs.multi_agent import FixedMemberTargetConfig, MultiAgentConfig

    return MultiAgentConfig(
        fixed={
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge/v1"],
                request_mode="chat",
            )
        }
    )


def test_think_split_routing_raises_for_debate_eval_without_renderer():
    from prime_rl.orchestrator.envs import validate_eval_think_split_routing

    with pytest.raises(ValueError, match=r"debate-eval.*\['debater_a', 'debater_b'\]"):
        validate_eval_think_split_routing(
            _eval_envs(_debate_eval_env()),
            multi_agent=_judge_fixed_config(),
            renderer_configured=False,
        )


def test_think_split_routing_raises_for_debate_eval_without_multi_agent_config():
    from prime_rl.configs.multi_agent import MultiAgentConfig
    from prime_rl.orchestrator.envs import validate_eval_think_split_routing

    # Pure self-play (no fixed targets, no train_one): every member is trained
    with pytest.raises(ValueError, match="debate-eval"):
        validate_eval_think_split_routing(
            _eval_envs(_debate_eval_env()),
            multi_agent=MultiAgentConfig(),
            renderer_configured=False,
        )


def test_think_split_routing_passes_with_renderer_configured():
    from prime_rl.orchestrator.envs import validate_eval_think_split_routing

    validate_eval_think_split_routing(
        _eval_envs(_debate_eval_env()),
        multi_agent=_judge_fixed_config(),
        renderer_configured=True,
    )


def test_think_split_routing_passes_when_all_debate_members_are_fixed():
    from prime_rl.configs.multi_agent import FixedMemberTargetConfig, MultiAgentConfig
    from prime_rl.orchestrator.envs import validate_eval_think_split_routing

    config = MultiAgentConfig(
        fixed={
            "everyone": FixedMemberTargetConfig(
                members=["debater_a", "debater_b", "judge"],
                model="frozen-model",
                base_url=["http://frozen/v1"],
                request_mode="chat",
            )
        }
    )
    validate_eval_think_split_routing(
        _eval_envs(_debate_eval_env()),
        multi_agent=config,
        renderer_configured=False,
    )


def test_think_split_routing_ignores_non_debate_multi_agent_envs():
    from prime_rl.orchestrator.envs import validate_eval_think_split_routing

    # Multi-agent but not a DebateEnv — no think-channel split contract
    inner = SimpleNamespace(is_multi_agent=True, members=["solver", "critic"])
    env = EvalEnv.__new__(EvalEnv)
    env.config = SimpleNamespace(resolved_name="ma-eval")
    env._env = inner

    validate_eval_think_split_routing(
        _eval_envs(env),
        multi_agent=_judge_fixed_config(),
        renderer_configured=False,
    )


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
