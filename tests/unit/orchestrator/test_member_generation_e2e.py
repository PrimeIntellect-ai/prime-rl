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


class _EvalMonitor:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def log_eval_samples(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _stub_eval_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("prime_rl.utils.monitor")
    module.get_monitor = lambda: _EvalMonitor()
    monkeypatch.setitem(sys.modules, "prime_rl.utils.monitor", module)


class _FakeSingleAgentEval(EvalEnv):
    def __init__(self) -> None:
        self.config = type(
            "Config",
            (),
            {"resolved_name": "single-eval", "rollouts_per_example": 1},
        )()
        self.examples = [
            {
                "example_id": "single",
                "prompt": [{"role": "user", "content": "q"}],
                "answer": "a",
            }
        ]
        self.sampling_args = {}
        self.generations: list[vf.MemberGenerationPlan | None] = []

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def requires_group_scoring(self) -> bool:
        return False

    @property
    def is_multi_agent(self) -> bool:
        return False

    async def run_rollout(self, **kwargs: Any) -> vf.RolloutOutput:
        self.generations.append(kwargs["generation"])
        return vf.RolloutOutput(
            example_id="single",
            reward=1.0,
            completion=[{"role": "assistant", "content": "ok"}],
            is_truncated=False,
            trajectory=[],
        )


class _FakeMultiAgentEval(_FakeSingleAgentEval):
    @property
    def name(self) -> str:
        return "ma-eval"

    @property
    def is_multi_agent(self) -> bool:
        return True

    def multi_agent_members(self) -> list[str]:
        return ["debater_a", "debater_b", "judge"]


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


async def _eval_client() -> vf.ClientConfig:
    return vf.ClientConfig(api_base_url="http://learner/v1")


def test_single_agent_eval_ignores_train_multi_agent_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_eval_monitor(monkeypatch)
    env = _FakeSingleAgentEval()

    outputs = asyncio.run(
        env.evaluate(
            model_name="learner-model",
            get_client=_eval_client,
            ckpt_step=1,
            step=1,
            cache_salt="1",
            multi_agent=_multi_agent_config(),
        )
    )

    assert len(outputs) == 1
    assert env.generations == [None]


def test_multi_agent_eval_receives_member_generation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_eval_monitor(monkeypatch)
    env = _FakeMultiAgentEval()

    outputs = asyncio.run(
        env.evaluate(
            model_name="learner-model",
            get_client=_eval_client,
            ckpt_step=1,
            step=1,
            cache_salt="1",
            multi_agent=_multi_agent_config(),
        )
    )

    assert len(outputs) == 1
    assert env.generations[0] is not None
    assert sorted(env.generations[0].members) == ["debater_a", "debater_b", "judge"]


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
    assert models_by_member[selected] == "learner-model"
    assert models_by_member[frozen] == "opponent-model"
    assert models_by_member["judge"] == "judge-model"

    seen = {(record["member_id"], record["model"]) for record in server.records}
    assert (selected, "learner-model") in seen
    assert (frozen, "opponent-model") in seen
    assert ("judge", "judge-model") in seen

    member_rollouts = vf.rollout_to_member_rollouts(output)
    by_member = {rollout["member_id"]: rollout for rollout in member_rollouts}
    assert by_member[selected]["model"] == "learner-model"
    assert by_member[selected]["sampling_args"]["extra_body"] == {"cache_salt": "7"}
    assert by_member[frozen]["model"] == "opponent-model"
    assert by_member[frozen]["sampling_args"]["temperature"] == 0.0
    assert by_member["judge"]["model"] == "judge-model"
    assert by_member["judge"]["sampling_args"]["max_completion_tokens"] == 12
