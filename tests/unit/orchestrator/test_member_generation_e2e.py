from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import verifiers as vf
from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import MARScore, MemberScore, Messages, State

from prime_rl.configs.multi_agent import (
    FixedMemberTargetConfig,
    MultiAgentConfig,
    TrainOneConfig,
    stable_train_member,
)
from prime_rl.configs.orchestrator import TrainEnvConfig, TrainSamplingConfig
from prime_rl.orchestrator.envs import Envs, EvalEnv, TrainEnv
from prime_rl.orchestrator.member_generation import DISPATCH_ID_FIELD


class _MembersOnlyRubric:
    members = ["solo"]


class _MembersOnlySingleAgentEnv:
    is_multi_agent = False
    members = ["solo"]
    rubric = _MembersOnlyRubric()


class _DetectionMultiAgentRubric(MultiAgentRubric):
    async def build_marscore(self, state: State) -> MARScore:
        return MARScore(
            members=[MemberScore(member_id="debater", reward=1.0)],
            episode_scalar=1.0,
        )


class _DetectionMultiAgentEnv(MultiAgentEnv):
    async def build_prompt(self, state: State, member_id: str, slot: TurnSlot) -> Messages:
        return state["prompt"]

    async def render_completion(self, state: State) -> None:
        state["completion"] = []


def _multi_agent_config(base_url: str = "http://fixed/v1") -> MultiAgentConfig:
    return MultiAgentConfig(
        train_one=TrainOneConfig(
            members=["debater_a", "debater_b"],
            unselected="opponent",
        ),
        fixed={
            "opponent": FixedMemberTargetConfig(
                model="opponent-model",
                base_url=[base_url],
                request_mode="chat",
            ),
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=[base_url],
                request_mode="chat",
            ),
        },
    )


def test_multi_agent_detection_ignores_members_attrs_on_single_agent_env() -> None:
    env = TrainEnv.__new__(TrainEnv)
    env._env = _MembersOnlySingleAgentEnv()

    assert env.is_multi_agent is False


def test_multi_agent_detection_uses_env_marker_not_rubric_type() -> None:
    env = TrainEnv.__new__(TrainEnv)
    env._env = types.SimpleNamespace(
        is_multi_agent=False,
        members=["debater"],
        rubric=_DetectionMultiAgentRubric(members=["debater"]),
    )

    assert env.is_multi_agent is False


def test_multi_agent_detection_accepts_verifiers_multi_agent_env() -> None:
    env = TrainEnv.__new__(TrainEnv)
    env._env = _DetectionMultiAgentEnv(
        schedule=StaticSchedule((TurnSlot(slot_id=0, agents=("debater",), phase="answer"),)),
        members=["debater"],
        dataset=lambda: None,
        rubric=_DetectionMultiAgentRubric(members=["debater"]),
    )

    assert env._env.is_multi_agent is True
    assert env.is_multi_agent is True


def test_env_collection_multi_agent_names_uses_env_predicate() -> None:
    envs = Envs.__new__(Envs)
    envs._envs = {
        "single": types.SimpleNamespace(name="single", is_multi_agent=False),
        "members-only": types.SimpleNamespace(name="members-only", is_multi_agent=True),
    }

    assert envs.multi_agent_names == {"members-only"}


class RecordingOpenAIServer(ThreadingHTTPServer):
    records: list[dict[str, Any]]
    counts: dict[str, int]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), RecordingOpenAIHandler)
        self.records = []
        self.counts = {}

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/v1"


class RecordingOpenAIHandler(BaseHTTPRequestHandler):
    server: RecordingOpenAIServer

    def log_message(self, *_args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        member_id = self.headers.get("X-Verifiers-Member-ID", "")
        self.server.counts[member_id] = self.server.counts.get(member_id, 0) + 1
        self.server.records.append(
            {
                "member_id": member_id,
                "model": body["model"],
                "temperature": body.get("temperature"),
                "max_completion_tokens": body.get("max_completion_tokens"),
                "logprobs": body.get("logprobs"),
            }
        )

        if member_id == "judge":
            content = "judged locally"
        elif self.server.counts[member_id] == 1:
            content = f"{member_id} proposes"
        else:
            content = f"{member_id} critiques"

        payload = {
            "id": f"cmpl-{len(self.server.records)}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def test_prime_env_server_carries_member_generation_plan_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asyncio.run(_run_prime_env_server_member_generation_smoke(monkeypatch, tmp_path))


async def _run_prime_env_server_member_generation_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_dir = Path(__file__).parents[2] / "fixtures"
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{fixture_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )
    monkeypatch.setenv("E2E_API_KEY", "local")
    sys.path.insert(0, str(fixture_dir))

    server = RecordingOpenAIServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    env = TrainEnv(
        TrainEnvConfig(
            id="ma-e2e-env",
            num_workers=1,
            sampling=TrainSamplingConfig(
                temperature=0.7,
                max_completion_tokens=64,
            ),
        )
    )
    try:
        await env.start(log_dir=tmp_path, log_level="warning")
        dispatch_id = "cpu-e2e-rollout-0"
        config = MultiAgentConfig(
            train_one=TrainOneConfig(
                members=["debater_a", "debater_b"],
                unselected="opponent",
            ),
            fixed={
                "opponent": FixedMemberTargetConfig(
                    model="opponent-model",
                    base_url=[server.base_url],
                    api_key_var="E2E_API_KEY",
                    request_mode="chat",
                    sampling={"temperature": 0.0},
                ),
                "judge": FixedMemberTargetConfig(
                    members=["judge"],
                    model="judge-model",
                    base_url=[server.base_url],
                    api_key_var="E2E_API_KEY",
                    request_mode="chat",
                    sampling={"temperature": 0.0, "max_completion_tokens": 12},
                ),
            },
        )
        learner_client = vf.ClientConfig(
            client_type="openai_chat_completions",
            api_base_url=server.base_url,
            api_key_var="E2E_API_KEY",
        )
        generation = env.compile_generation(
            config,
            client=learner_client,
            model_name="learner-model",
            cache_salt="7",
            dispatch_id=dispatch_id,
            group_id="group-e2e",
        )
        assert generation is not None

        output = await env.run_rollout(
            client=learner_client,
            example={
                "prompt": [{"role": "user", "content": "Which answer wins?"}],
                "answer": "A",
                "example_id": "cpu-e2e",
                "task": {"env_id": "ma-e2e-env"},
            },
            model_name="learner-model",
            cache_salt="7",
            generation=generation,
            dispatch_id=dispatch_id,
        )
    finally:
        if env._env_client is not None:
            await env._env_client.close()
        env.shutdown()
        server.shutdown()
        server.server_close()

    selected = stable_train_member(["debater_a", "debater_b"], seed=0, dispatch_id=dispatch_id)
    frozen = ({"debater_a", "debater_b"} - {selected}).pop()

    assert output[DISPATCH_ID_FIELD] == dispatch_id
    assert len(output["trajectory"]) == 5
    models_by_member = {
        step["extras"]["member_id"]: step["extras"]["generation"]["model"] for step in output["trajectory"]
    }
    generation_by_member = {step["extras"]["member_id"]: step["extras"]["generation"] for step in output["trajectory"]}
    assert models_by_member[selected] == "learner-model"
    assert models_by_member[frozen] == "opponent-model"
    assert models_by_member["judge"] == "judge-model"
    assert generation_by_member[selected]["sampling_args"]["extra_body"] == {"cache_salt": "7"}
    assert generation_by_member[frozen]["sampling_args"]["temperature"] == 0.0
    assert generation_by_member["judge"]["sampling_args"]["max_completion_tokens"] == 12

    seen = {(record["member_id"], record["model"]) for record in server.records}
    assert (selected, "learner-model") in seen
    assert (frozen, "opponent-model") in seen
    assert ("judge", "judge-model") in seen

    # On the wire, only learner requests carry the learner-only logprobs field.
    logprobs_by_member = {record["member_id"]: record["logprobs"] for record in server.records}
    assert logprobs_by_member[selected] is True
    assert logprobs_by_member[frozen] is None
    assert logprobs_by_member["judge"] is None

    member_rollouts = vf.rollout_to_member_rollouts(output)
    by_member = {rollout["member_id"]: rollout for rollout in member_rollouts}
    assert set(by_member) == {selected, frozen, "judge"}


class _RecordingInnerEnv:
    """Inner vf-env fake: records the kwargs the orchestrator Env forwards."""

    requires_group_rollouts = False

    def __init__(self, *, is_multi_agent: bool, members: list[str] | None = None) -> None:
        self.is_multi_agent = is_multi_agent
        if members is not None:
            self.members = members
        self.kwargs: dict[str, Any] | None = None

    async def run_rollout(self, _input: Any, **kwargs: Any) -> vf.RolloutOutput:
        self.kwargs = kwargs
        return vf.RolloutOutput(example_id="ex-1")


class _RecordingInnerMAEnv(_RecordingInnerEnv, vf.MultiAgentEnv):
    """MA variant: real ``vf.MultiAgentEnv`` subclass so the wrapper's
    ``isinstance``-based ``is_multi_agent`` property sees it."""

    def build_prompt(self, *args, **kwargs):  # pragma: no cover - unused fake
        raise NotImplementedError

    def render_completion(self, *args, **kwargs):  # pragma: no cover - unused fake
        raise NotImplementedError

    def __init__(self, *, members: list[str]) -> None:
        _RecordingInnerEnv.__init__(self, is_multi_agent=True, members=members)


def _recording_eval_env(name: str, inner: _RecordingInnerEnv) -> EvalEnv:
    env = EvalEnv.__new__(EvalEnv)
    env.config = types.SimpleNamespace(
        resolved_name=name,
        group_size=1,
        max_retries=0,
        state_columns=[],
    )
    env._env = inner
    env._env_client = types.SimpleNamespace()
    env.sampling_args = {}
    return env


def _dispatcher_for_eval(eval_env: EvalEnv, multi_agent: MultiAgentConfig):
    from renderers import Qwen3RendererConfig

    from prime_rl.configs.shared import ClientConfig as PrimeClientConfig
    from prime_rl.orchestrator.dispatcher import RolloutDispatcher
    from prime_rl.orchestrator.types import Policy
    from prime_rl.utils.client import StaticInferencePool

    pool = StaticInferencePool(
        PrimeClientConfig(base_url=["http://student/v1"], api_key_var="VLLM_API_KEY"),
        model_name="learner-model",
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=Qwen3RendererConfig(),
        pool_size=2,
    )
    envs = Envs.__new__(Envs)
    envs._envs = {eval_env.name: eval_env}
    dispatcher = RolloutDispatcher(
        train_envs=envs,
        eval_envs=envs,
        train_source=None,  # schedule_group_rollout never touches the sources
        eval_source=None,
        inference=pool,
        eval_inference=pool,
        policy=Policy(version=0, model_name="learner-model"),
        max_inflight_rollouts=4,
        tasks_per_minute=None,
        max_off_policy_steps=4,
        training_mode="rl",
        multi_agent=multi_agent,
    )
    return dispatcher, pool


async def _schedule_eval_rollout(dispatcher: Any, env_name: str, dispatch_ids: list[str]) -> None:
    import uuid

    from prime_rl.orchestrator.types import GroupState

    gid = uuid.uuid4()
    group = GroupState(
        kind="eval",
        env_name=env_name,
        example={"prompt": [{"role": "user", "content": "q"}], "example_id": "ex-1"},
        rollouts_to_schedule=1,
        target_rollouts=1,
        eval_step=0,
        dispatch_ids=dispatch_ids,
    )
    dispatcher.groups[gid] = group
    assert await dispatcher.schedule_group_rollout(gid, group)
    await next(iter(dispatcher.inflight))


def test_dispatcher_eval_routes_trained_members_through_train_typed_client() -> None:
    """F4: at eval, the trained debate members must generate through the same
    client type as training (renderer/TITO) while fixed members keep their
    configured chat targets."""

    async def run() -> None:
        dispatch_id = "eval:0:ma-eval:ex-1:0"
        inner = _RecordingInnerMAEnv(members=["debater_a", "debater_b", "judge"])
        env = _recording_eval_env("ma-eval", inner)
        dispatcher, pool = _dispatcher_for_eval(env, _multi_agent_config())

        await _schedule_eval_rollout(dispatcher, "ma-eval", [dispatch_id])

        assert inner.kwargs is not None
        # The pinned eval client was re-typed to the train path's renderer client
        rollout_client = inner.kwargs["client"]
        assert rollout_client.client_type == "renderer"
        assert rollout_client.api_base_url == "http://student/v1"

        plan = inner.kwargs["generation"]
        assert plan is not None
        assert sorted(plan.members) == ["debater_a", "debater_b", "judge"]
        selected = stable_train_member(["debater_a", "debater_b"], seed=0, dispatch_id=dispatch_id)
        frozen = ({"debater_a", "debater_b"} - {selected}).pop()
        assert plan.members[selected].client.client_type == "renderer"
        assert plan.members[selected].client.renderer_config == pool._renderer_config
        assert plan.members[selected].client.renderer_model_name == "learner-model"
        assert plan.members[selected].client.api_base_url == "http://student/v1"
        assert plan.members[frozen].client.client_type == "openai_chat_completions"
        assert plan.members["judge"].client.client_type == "openai_chat_completions"

    asyncio.run(run())


def test_dispatcher_eval_keeps_chat_client_for_single_agent_envs() -> None:
    async def run() -> None:
        inner = _RecordingInnerEnv(is_multi_agent=False)
        env = _recording_eval_env("single-eval", inner)
        dispatcher, _pool = _dispatcher_for_eval(env, _multi_agent_config())

        await _schedule_eval_rollout(dispatcher, "single-eval", [])

        assert inner.kwargs is not None
        assert inner.kwargs["client"].client_type == "openai_chat_completions"
        assert inner.kwargs["generation"] is None

    asyncio.run(run())
