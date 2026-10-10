import asyncio
import gc
import weakref
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Tool, ToolMessage, Usage, UserMessage

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress
from prime_rl.utils.logger import setup_logger

setup_logger("warning")


def trace(*, advantage=1.0, ok=True, trainable=True, sampled=True):
    tools = [Tool(name="search", description="schema" * 100, parameters={"type": "object"})]
    nodes = [
        MessageNode(message=UserMessage(content="prompt"), tools=tools, token_ids=[1], mask=[False]),
        MessageNode(
            parent=0,
            message=ToolMessage(content="result", tool_call_id="call1", name="search"),
            token_ids=[2],
            mask=[False],
        ),
        MessageNode(
            parent=1,
            message=AssistantMessage(content="answer"),
            sampled=sampled,
            token_ids=[3, 4],
            mask=[sampled, sampled],
            is_content=[True, True],
            logprobs=[-0.1, -0.2],
            advantages=[advantage, advantage],
            reference_logprobs=[-0.3, -0.4],
            routed_experts=np.ones((2, 2, 2), dtype=np.uint8),
        ),
    ]
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=trainable),
        tools=tools,
        nodes=nodes,
        ok=ok,
        is_completed=True,
        rewards={"correct": vf.Reward(score=1.0)},
        calls=[ModelCall(node=2, usage=Usage(prompt_tokens=2, completion_tokens=2), finish_reason="stop")],
        errors=[] if ok else [vf.Error(type="ProviderError", message="failed")],
    )


def episode(traces, *, group="g", version=0):
    return vf.Episode(
        env=vf.EnvInfo(id="test", name="test"),
        task=traces[0].task,
        group=vf.GroupInfo(id=group),
        traces=traces,
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=version, end=version))),
    )


def sink(*, constant=True, admitted=True, batch_size=100):
    config = OrchestratorConfig(batch_size=batch_size, constant_trainer_batch_size=constant, max_off_policy_steps=1)
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=1),
        sampling_args={"temperature": 1.0},
        requires_sampling_masks=False,
        algorithm=Algorithm(GRPOAlgoConfig(), None),
    )
    return TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env),
        progress=Progress(),
        batch_size=batch_size,
        on_result=lambda group: admitted,
    )


@pytest.mark.parametrize("constant,admitted", [(True, True), (False, True), (True, False)])
def test_zero_output_finalization_releases_objects_without_reporting(constant, admitted):
    async def run():
        s = sink(constant=constant, admitted=admitted, batch_size=4)
        refs = []
        for i in range(40):
            t = trace(advantage=0)
            ep = episode([t], group=str(i))
            refs.extend([weakref.ref(t), weakref.ref(ep)])
            before = msgspec.msgpack.encode(trace_to_samples(t, env_name="test"))
            batch = await s.add(ep)
            assert batch is None or not batch.samples
            assert msgspec.msgpack.encode(trace_to_samples(t, env_name="test")) == before
            assert s.take_rollout_window() is None
            assert s.rollout_window.attempts == i + 1
            del t, ep, batch
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not s.pending_groups and not s.pending_group_failures and not s.episode_by_trace
        assert not s.pending_batch
        window = s.take_rollout_window(force=True)
        assert window.attempts == 40
        assert window.traces == 40
        assert s.take_rollout_window(force=True) is None
        assert s.progress.step == 1
        if constant:
            assert window.discarded == 40
        else:
            assert window.metrics.to_dict()["rollout/queued/pruned"] == 40

    asyncio.run(run())


def test_queued_trace_does_not_own_discarded_siblings():
    async def run():
        kept = trace()
        discarded = [trace(advantage=0), trace(ok=False), trace(trainable=False), trace(sampled=False)]
        ep = episode([kept, *discarded])
        refs = [weakref.ref(t) for t in discarded] + [weakref.ref(ep)]
        s = sink()
        before = msgspec.msgpack.encode(trace_to_samples(kept, env_name="test"))
        assert await s.add(ep) is None
        assert all(t.nodes[2].token_ids == [3, 4] for t in discarded)
        del discarded, ep
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert s.episode_by_trace[kept.id].traces == [kept]
        assert msgspec.msgpack.encode(trace_to_samples(kept, env_name="test")) == before
        window = s.take_rollout_window(force=True)
        assert window.traces == 5 and window.attempts == 1
        assert window.metrics.to_dict()["rollout/traces/discarded"] == 4

    asyncio.run(run())


def test_stale_queue_releases_trace_without_recounting_arrivals():
    async def run():
        s = sink(batch_size=4)
        old = trace()
        ref = weakref.ref(old)
        await s.add(episode([old]))
        assert s.take_rollout_window(force=True).attempts == 1
        del old
        s.progress.step = 4
        assert s.take_batch() is None
        gc.collect()
        assert ref() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == 0 and window.traces == 0
        assert window.metrics.to_dict()["off_policy/dropped"] == 1
        assert s.take_rollout_window(force=True) is None

    asyncio.run(run())


def test_deferred_pruning_and_multiple_batches_preserve_episode_boundaries():
    async def run():
        s = sink(constant=False, batch_size=2)
        zero, kept, sibling, another = trace(advantage=0), trace(), trace(), trace()
        ep = episode([zero, kept, sibling, another])
        ep_id = ep.id
        batch = await s.add(ep)
        assert len(batch.samples) == 1 and batch.samples[0].trace_id == kept.id
        assert len(batch.cohort) == 1 and batch.cohort.episodes[0].id == ep_id
        assert sibling.id in s.pending_batch
        assert zero.nodes[2].token_ids == [3, 4]
        batch = s.take_batch()
        assert len(batch.samples) == 2
        assert len(batch.cohort) == 1 and batch.cohort.num_traces == 2
        assert batch.cohort.episodes[0].id == ep_id
        assert not s.pending_batch and not s.episode_by_trace

    asyncio.run(run())


def test_failures_and_cancellations_accumulate_until_reporting():
    async def run():
        s = sink(batch_size=4)
        for i in range(8):
            failure = DispatchFailure(
                kind="train",
                env_name="test",
                group_id=str(i),
                step=1,
                policy_version=0,
                task_type="Task",
                task_key="key",
                task_hash="hash",
                error=vf.Error(type="TransportError", message="large" * 1000),
            )
            assert await s.fail(failure) is None
            assert s.take_rollout_window() is None
            assert not s.pending_group_failures
        for i in range(4):
            await s.cancel(GroupCancellation("train", "test", f"cancel{i}", 1, 1, "stale"))
        assert s.take_rollout_window() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == window.discarded == 12
        assert window.stale == 4
        assert window.errored == 8
        assert window.metrics.to_dict()["train/agg/all/dispatch_failure/mean"] == 1
        assert not s.pending_group_cancellations

    asyncio.run(run())


def test_partial_window_and_shutdown_release():
    async def run():
        s = sink(batch_size=4)
        t = trace()
        ref = weakref.ref(t)
        await s.add(episode([t]))
        assert s.take_rollout_window() is None
        del t
        s.discard_queued()
        gc.collect()
        assert ref() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == 1
        assert window.metrics.to_dict()["rollout/queued/cancelled"] == 1
        assert s.take_rollout_window(force=True) is None

    asyncio.run(run())
