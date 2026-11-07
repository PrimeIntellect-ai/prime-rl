"""The RL adapter routes native Harbor scoring without a server or judge request."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from cyberstrike_rl import CyberStrikeRL, CyberStrikeRLConfig
from prime_flow.qc.relay import Call, GraderRelay
from verifiers.v1 import JudgeResponse
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.env import Env
from verifiers.v1.tasksets.harbor.taskset import (
    HarborData,
    HarborTask,
    HarborTaskConfig,
)
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.utils.generic import concrete_type
from verifiers.v1.utils.loaders import environment_class


@pytest_asyncio.fixture
async def relay_case(monkeypatch):
    start, stop = AsyncMock(), AsyncMock()
    monkeypatch.setattr(GraderRelay, "start", start)
    monkeypatch.setattr(GraderRelay, "stop", stop)
    monkeypatch.setattr(GraderRelay, "url", AsyncMock(return_value="https://relay.fixture"))
    env = CyberStrikeRL(CyberStrikeRLConfig(taskset={"id": "harbor"}))
    task = HarborTask(
        HarborData(task_dir="/fixture", verifier_env={"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}),
        HarborTaskConfig(),
    )
    task.verifier_staged = True
    trace = Trace(task=TraceTask(type="HarborTask", data=task.data), agent=AgentInfo(config=AgentConfig()))
    runtime = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")),
        read=AsyncMock(return_value=json.dumps({"reward": 1.0, "deterministic": 1.0}).encode()),
        set_network=AsyncMock(),
    )
    await env.start()
    try:
        yield env, task, trace, runtime, start, stop
    finally:
        await env.stop()
        stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_grant_spans_native_scoring_and_preserves_task_identity(relay_case):
    env, task, trace, runtime, start, _ = relay_case
    assert concrete_type(CyberStrikeRL, EnvConfig, origin=Env) is CyberStrikeRLConfig
    assert environment_class("harbor", "prime/cyberstrike-rl@0.1.0") is CyberStrikeRL
    start.assert_awaited_once()
    declared = task.data.model_dump(mode="json")

    async def run(routed):
        token = routed.data.verifier_env["ANTHROPIC_API_KEY"]
        assert token in env._grader.tokens
        assert routed.data.verifier_env["ANTHROPIC_BASE_URL"] == "https://relay.fixture"
        assert routed.data.model_dump(mode="json") == declared
        await routed.score(trace, runtime)
        verifier_call = next(call for call in runtime.run.await_args_list if "ANTHROPIC_API_KEY" in call.args[1])
        assert verifier_call.args[1]["ANTHROPIC_API_KEY"] == token
        assert token in env._grader.tokens
        env._grader.tokens[token].append(Call({"model": "fixture"}, JudgeResponse(text="fixture")))
        return trace

    await env.run(task, SimpleNamespace(agent=SimpleNamespace(run=run)))
    assert env._grader.tokens == {}
    assert trace.reward == 1.0
    assert trace.metrics["deterministic"] == 1.0
    assert len(trace.info["judge_calls"]) == 1
    assert task.data.model_dump(mode="json") == declared


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("fixture failure"), asyncio.CancelledError()])
async def test_grant_revoked_on_failed_or_cancelled_episode(relay_case, error):
    env, task, *_ = relay_case

    async def run(routed):
        assert routed.data.verifier_env["ANTHROPIC_API_KEY"] in env._grader.tokens
        raise error

    with pytest.raises(type(error)):
        await env.run(task, SimpleNamespace(agent=SimpleNamespace(run=run)))
    assert env._grader.tokens == {}
