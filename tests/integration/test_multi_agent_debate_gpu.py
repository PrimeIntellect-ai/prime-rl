from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
import verifiers as vf
from renderers import Qwen3RendererConfig

from prime_rl.configs.multi_agent import (
    FixedMemberTargetConfig,
    MultiAgentConfig,
    TrainOneConfig,
    stable_train_member,
)
from prime_rl.configs.orchestrator import TrainEnvConfig, TrainSamplingConfig
from prime_rl.orchestrator.envs import TrainEnv
from prime_rl.orchestrator.member_generation import DISPATCH_ID_FIELD

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

BASE_URL_ENV = "PRIME_RL_MA_GPU_SMOKE_BASE_URL"
API_KEY_ENV_ENV = "PRIME_RL_MA_GPU_SMOKE_API_KEY_VAR"
DEFAULT_API_KEY_ENV = "VLLM_API_KEY"


def test_prime_env_server_multi_agent_debate_uses_gpu_openai_endpoint(tmp_path: Path) -> None:
    base_url = os.environ.get(BASE_URL_ENV)
    if not base_url:
        pytest.skip(f"set {BASE_URL_ENV} to a GPU-backed OpenAI-compatible endpoint")

    api_key_var = os.environ.get(API_KEY_ENV_ENV, DEFAULT_API_KEY_ENV)
    if api_key_var not in os.environ:
        pytest.skip(f"set {api_key_var} for the GPU-backed endpoint")

    asyncio.run(_run_gpu_backed_multi_agent_smoke(tmp_path, base_url=base_url, api_key_var=api_key_var))


async def _run_gpu_backed_multi_agent_smoke(tmp_path: Path, *, base_url: str, api_key_var: str) -> None:
    _prepend_fixture_path()

    learner_model = os.environ.get("PRIME_RL_MA_GPU_SMOKE_LEARNER_MODEL", "Qwen/Qwen3-0.6B")
    learner_client_type = os.environ.get("PRIME_RL_MA_GPU_SMOKE_LEARNER_CLIENT_TYPE", "renderer")
    learner_renderer_model = os.environ.get("PRIME_RL_MA_GPU_SMOKE_RENDERER_MODEL", learner_model)
    opponent_model = os.environ.get("PRIME_RL_MA_GPU_SMOKE_OPPONENT_MODEL", "opponent-model")
    judge_model = os.environ.get("PRIME_RL_MA_GPU_SMOKE_JUDGE_MODEL", "judge-model")
    fixed_request_mode = os.environ.get("PRIME_RL_MA_GPU_SMOKE_FIXED_REQUEST_MODE", "renderer")
    fixed_renderer_model = os.environ.get("PRIME_RL_MA_GPU_SMOKE_FIXED_RENDERER_MODEL", learner_renderer_model)
    rollout_count = int(os.environ.get("PRIME_RL_MA_GPU_SMOKE_ROLLOUTS", "4"))
    request_timeout = float(os.environ.get("PRIME_RL_MA_GPU_SMOKE_TIMEOUT", "1200.0"))
    connect_timeout = float(os.environ.get("PRIME_RL_MA_GPU_SMOKE_CONNECT_TIMEOUT", "30.0"))

    env = TrainEnv(
        TrainEnvConfig(
            id="ma-e2e-env",
            num_workers=1,
            sampling=TrainSamplingConfig(
                temperature=0.7,
                max_completion_tokens=32,
            ),
        )
    )
    try:
        await env.start(log_dir=tmp_path, log_level="warning")
        fixed_renderer_kwargs = _fixed_renderer_kwargs(fixed_request_mode, fixed_renderer_model)
        config = MultiAgentConfig(
            train_one=TrainOneConfig(
                members=["debater_a", "debater_b"],
                unselected="opponent",
            ),
            fixed={
                "opponent": FixedMemberTargetConfig(
                    model=opponent_model,
                    base_url=[base_url],
                    api_key_var=api_key_var,
                    request_mode=fixed_request_mode,
                    sampling={"temperature": 0.0, "max_completion_tokens": 32},
                    timeout=request_timeout,
                    connect_timeout=connect_timeout,
                    **fixed_renderer_kwargs,
                ),
                "judge": FixedMemberTargetConfig(
                    members=["judge"],
                    model=judge_model,
                    base_url=[base_url],
                    api_key_var=api_key_var,
                    request_mode=fixed_request_mode,
                    sampling={"temperature": 0.0, "max_completion_tokens": 32},
                    timeout=request_timeout,
                    connect_timeout=connect_timeout,
                    **fixed_renderer_kwargs,
                ),
            },
        )
        learner_client = vf.ClientConfig(
            client_type=learner_client_type,
            api_base_url=base_url,
            api_key_var=api_key_var,
            renderer_model_name=learner_renderer_model if learner_client_type == "renderer" else None,
            renderer_pool_size=1 if learner_client_type == "renderer" else None,
            timeout=request_timeout,
            connect_timeout=connect_timeout,
        )
        outputs = []
        dispatch_ids = _dispatch_ids_covering_both_trainable_members(rollout_count)
        for idx, dispatch_id in enumerate(dispatch_ids):
            generation = env.compile_generation(
                config,
                client=learner_client,
                model_name=learner_model,
                cache_salt=f"gpu-smoke-{idx}",
                dispatch_id=dispatch_id,
            )
            assert generation is not None

            outputs.append(
                await env.run_rollout(
                    client=learner_client,
                    example={
                        "prompt": [{"role": "user", "content": "Debate whether answer A or answer B is better."}],
                        "answer": "A",
                        "example_id": f"gpu-smoke-{idx}",
                        "task": {"env_id": "ma-e2e-env"},
                    },
                    model_name=learner_model,
                    cache_salt=f"gpu-smoke-{idx}",
                    generation=generation,
                    dispatch_id=dispatch_id,
                )
            )
    finally:
        if env._env_client is not None:
            await env._env_client.close()
        env.shutdown()

    assert len(outputs) == rollout_count
    seen_trainable_members = set()
    for output, dispatch_id in zip(outputs, dispatch_ids):
        selected = stable_train_member(["debater_a", "debater_b"], seed=0, dispatch_id=dispatch_id)
        frozen = ({"debater_a", "debater_b"} - {selected}).pop()
        seen_trainable_members.add(selected)

        assert output[DISPATCH_ID_FIELD] == dispatch_id
        assert len(output["trajectory"]) == 5

        member_rollouts = vf.rollout_to_member_rollouts(output)
        by_member = {rollout["member_id"]: rollout for rollout in member_rollouts}
        assert set(by_member) == {"debater_a", "debater_b", "judge"}
        assert by_member[selected]["model"] == learner_model
        assert by_member[frozen]["model"] == opponent_model
        assert by_member["judge"]["model"] == judge_model

    assert seen_trainable_members == {"debater_a", "debater_b"}


def _prepend_fixture_path() -> None:
    fixture_path = str(Path(__file__).parents[1] / "fixtures")
    if fixture_path not in sys.path:
        sys.path.insert(0, fixture_path)

    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [part for part in pythonpath.split(os.pathsep) if part]
    if fixture_path not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([fixture_path, *parts])


def _fixed_renderer_kwargs(request_mode: str, renderer_model: str) -> dict[str, object]:
    if request_mode != "renderer":
        return {}
    return {
        "renderer": Qwen3RendererConfig(),
        "renderer_model_name": renderer_model,
        "renderer_pool_size": 1,
    }


def _dispatch_ids_covering_both_trainable_members(count: int) -> list[str]:
    if count < 2:
        raise ValueError("PRIME_RL_MA_GPU_SMOKE_ROLLOUTS must be at least 2")

    dispatch_ids: list[str] = []
    seen: set[str] = set()
    candidate_idx = 0
    while len(dispatch_ids) < count:
        dispatch_id = f"gpu-smoke-rollout-{candidate_idx}"
        candidate_idx += 1
        selected = stable_train_member(["debater_a", "debater_b"], seed=0, dispatch_id=dispatch_id)
        if len(seen) < 2 and selected in seen:
            continue
        dispatch_ids.append(dispatch_id)
        seen.add(selected)

    return dispatch_ids
