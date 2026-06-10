import asyncio
from types import SimpleNamespace

import pytest
import verifiers as vf
from pydantic import ValidationError
from renderers import Qwen3RendererConfig

from prime_rl.configs.multi_agent import FixedMemberTargetConfig, MultiAgentConfig, TrainOneConfig
from prime_rl.orchestrator.envs import TrainEnv
from prime_rl.orchestrator.member_generation import (
    DISPATCH_ID_FIELD,
    compile_member_generation_plan,
    is_trainable_member,
    uncovered_trainable_members,
    validate_member_references,
)


def test_compile_member_generation_plan_routes_train_one_and_fixed_members():
    config = MultiAgentConfig(
        train_one=TrainOneConfig(
            members=["debater_a", "debater_b"],
            unselected="opponent",
        ),
        fixed={
            "opponent": FixedMemberTargetConfig(
                model="opponent-model",
                base_url=["http://opponent/v1"],
                request_mode="chat",
                sampling={"temperature": 0.0},
            ),
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge-1/v1", "http://judge-2/v1"],
                request_mode="token",
                sampling={"temperature": 0.0, "max_completion_tokens": 256},
            ),
        },
    )
    dispatch_id = "dispatch-compile"

    plan = compile_member_generation_plan(
        config,
        member_ids=["debater_a", "debater_b", "judge"],
        default_client=vf.ClientConfig(
            client_type="renderer",
            api_base_url="http://learner/v1",
            api_key_var="VLLM_API_KEY",
        ),
        default_model="learner-lora",
        learner_sampling_args={"temperature": 1.0, "extra_body": {"cache_salt": "7"}},
        fixed_sampling_args={"temperature": 1.0, "max_completion_tokens": 1024},
        dispatch_id=dispatch_id,
    )

    assert plan is not None
    trainable = [
        member
        for member in ("debater_a", "debater_b")
        if is_trainable_member(config, {DISPATCH_ID_FIELD: dispatch_id}, member)
    ]
    assert len(trainable) == 1
    selected = trainable[0]
    frozen = ({"debater_a", "debater_b"} - {selected}).pop()

    assert plan.members[selected].model == "learner-lora"
    assert plan.members[selected].client.client_type == "renderer"
    assert plan.members[selected].sampling_args["extra_body"] == {"cache_salt": "7"}
    assert plan.members[frozen].model == "opponent-model"
    assert plan.members[frozen].client.client_type == "openai_chat_completions"
    assert plan.members[frozen].sampling_args == {
        "temperature": 0.0,
        "max_completion_tokens": 1024,
    }
    assert plan.members["judge"].model == "judge-model"
    assert plan.members["judge"].client.client_type == "openai_chat_completions_token"
    assert plan.members["judge"].sampling_args == {
        "temperature": 0.0,
        "max_completion_tokens": 256,
    }


def test_fixed_member_renderer_config_reaches_client_config():
    renderer = Qwen3RendererConfig()
    config = MultiAgentConfig(
        fixed={
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge/v1"],
                request_mode="renderer",
                renderer=renderer,
                renderer_model_name="Qwen/Qwen3-0.6B",
                renderer_pool_size=2,
            )
        }
    )

    plan = compile_member_generation_plan(
        config,
        member_ids=["debater", "judge"],
        default_client=vf.ClientConfig(api_base_url="http://learner/v1"),
        default_model="learner-model",
        learner_sampling_args={},
        fixed_sampling_args={},
        dispatch_id="dispatch-renderer",
    )

    assert plan is not None
    client = plan.members["judge"].client
    assert client.client_type == "renderer"
    assert client.renderer_config == renderer
    assert client.renderer_model_name == "Qwen/Qwen3-0.6B"
    assert client.renderer_pool_size == 2


def test_train_one_unselected_renderer_target_preserves_renderer_config():
    renderer = Qwen3RendererConfig()
    config = MultiAgentConfig(
        train_one=TrainOneConfig(
            members=["debater_a", "debater_b"],
            unselected="opponent",
        ),
        fixed={
            "opponent": FixedMemberTargetConfig(
                model="opponent-model",
                base_url=["http://opponent/v1"],
                request_mode="renderer",
                renderer=renderer,
                renderer_model_name="Qwen/Qwen3-0.6B",
                renderer_pool_size=3,
            )
        },
    )
    dispatch_id = "dispatch-renderer-train-one"

    plan = compile_member_generation_plan(
        config,
        member_ids=["debater_a", "debater_b"],
        default_client=vf.ClientConfig(api_base_url="http://learner/v1"),
        default_model="learner-model",
        learner_sampling_args={},
        fixed_sampling_args={},
        dispatch_id=dispatch_id,
    )

    assert plan is not None
    trainable = [
        member
        for member in ("debater_a", "debater_b")
        if is_trainable_member(config, {DISPATCH_ID_FIELD: dispatch_id}, member)
    ]
    assert len(trainable) == 1
    frozen = ({"debater_a", "debater_b"} - {trainable[0]}).pop()
    client = plan.members[frozen].client
    assert client.client_type == "renderer"
    assert client.renderer_config == renderer
    assert client.renderer_model_name == "Qwen/Qwen3-0.6B"
    assert client.renderer_pool_size == 3


def test_fixed_member_renderer_fields_require_renderer_mode():
    with pytest.raises(ValidationError, match="fixed target renderer fields require request_mode = 'renderer'"):
        FixedMemberTargetConfig(
            members=["judge"],
            model="judge-model",
            base_url=["http://judge/v1"],
            request_mode="chat",
            renderer_model_name="Qwen/Qwen3-0.6B",
        )


def test_env_compile_generation_strips_learner_only_fields_from_fixed_targets():
    env = TrainEnv.__new__(TrainEnv)
    env.config = SimpleNamespace(resolved_name="ma", state_columns=[], max_retries=0)
    env._env = SimpleNamespace(
        is_multi_agent=True,
        members=["debater", "judge"],
        rubric=SimpleNamespace(),
    )
    env.sampling_args = {
        "temperature": 1.0,
        "max_completion_tokens": 1024,
        "logprobs": True,
        "extra_body": {
            "top_k": -1,
            "min_p": 0.0,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    config = MultiAgentConfig(
        fixed={
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge/v1"],
                sampling={"temperature": 0.0},
            )
        }
    )

    plan = env.compile_generation(
        config,
        client=vf.ClientConfig(api_base_url="http://learner/v1"),
        model_name="learner-model",
        cache_salt="7",
        dispatch_id="dispatch-1",
    )

    assert plan is not None
    # The learner keeps every training-loop field.
    assert plan.members["debater"].sampling_args["logprobs"] is True
    assert plan.members["debater"].sampling_args["extra_body"] == {
        "top_k": -1,
        "min_p": 0.0,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": "7",
    }
    # The fixed target inherits portable defaults and user extras only —
    # no logprobs, no return_token_ids/top_k/min_p/cache_salt — while its
    # explicit sampling override (temperature) still wins.
    assert plan.members["judge"].sampling_args == {
        "temperature": 0.0,
        "max_completion_tokens": 1024,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    # The fixed target's extra_body is a copy — mutating it must not leak
    # into the env's learner sampling args.
    plan.members["judge"].sampling_args["extra_body"]["cache_salt"] = "9"
    assert env.sampling_args["extra_body"] == {
        "top_k": -1,
        "min_p": 0.0,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_run_rollout_forwards_generation_and_records_dispatch_id():
    async def run() -> None:
        env = TrainEnv.__new__(TrainEnv)
        env.config = SimpleNamespace(resolved_name="ma", state_columns=[], max_retries=0)
        env._env_client = object()
        runner = SimpleNamespace(run_rollout=None)
        plan = vf.MemberGenerationPlan()

        async def run_rollout(*_args, **kwargs):
            runner.kwargs = kwargs
            return vf.RolloutOutput(example_id="ex-1")

        runner.run_rollout = run_rollout
        env._env = runner
        env.sampling_args = {}

        output = await env.run_rollout(
            client=vf.ClientConfig(api_base_url="http://learner/v1"),
            example={"prompt": [{"role": "user", "content": "q"}], "example_id": "ex-1"},
            model_name="learner-model",
            cache_salt="7",
            generation=plan,
            dispatch_id="dispatch-1",
        )

        assert runner.kwargs["generation"] is plan
        assert output[DISPATCH_ID_FIELD] == "dispatch-1"

    asyncio.run(run())


def test_validate_member_references_rejects_typo_member_ids_listing_valid_ids():
    config = MultiAgentConfig(
        train_one=TrainOneConfig(members=["debater_a", "debater_c"], unselected="opponent"),
        fixed={
            "opponent": FixedMemberTargetConfig(model="opponent-model", base_url=["http://opponent/v1"]),
            "judge": FixedMemberTargetConfig(members=["jugde"], model="judge-model", base_url=["http://judge/v1"]),
        },
    )

    with pytest.raises(ValueError, match=r"\['debater_c', 'jugde'\]") as exc_info:
        validate_member_references(config, {"gpqa-debate": ["debater_a", "debater_b", "judge"]})

    assert "gpqa-debate=['debater_a', 'debater_b', 'judge']" in str(exc_info.value)


def test_validate_member_references_rejects_enabled_config_without_multi_agent_envs():
    config = MultiAgentConfig(
        fixed={"judge": FixedMemberTargetConfig(members=["judge"], model="judge-model", base_url=["http://judge/v1"])}
    )

    with pytest.raises(ValueError, match="no loaded environment"):
        validate_member_references(config, {})


def test_validate_member_references_accepts_known_members_and_disabled_config():
    validate_member_references(MultiAgentConfig(), {})

    config = MultiAgentConfig(
        train_one=TrainOneConfig(members=["debater_a", "debater_b"], unselected="opponent"),
        fixed={
            "opponent": FixedMemberTargetConfig(model="opponent-model", base_url=["http://opponent/v1"]),
            "judge": FixedMemberTargetConfig(members=["judge"], model="judge-model", base_url=["http://judge/v1"]),
        },
    )
    validate_member_references(config, {"gpqa-debate": ["debater_a", "debater_b", "judge"]})


def test_uncovered_trainable_members_flags_default_trainable_under_train_one():
    config = MultiAgentConfig(
        train_one=TrainOneConfig(members=["debater_a", "debater_b"], unselected="opponent"),
        fixed={"opponent": FixedMemberTargetConfig(model="opponent-model", base_url=["http://opponent/v1"])},
    )

    assert uncovered_trainable_members(config, ["debater_a", "debater_b", "judge"]) == ["judge"]
    assert uncovered_trainable_members(MultiAgentConfig(), ["debater_a", "judge"]) == []
