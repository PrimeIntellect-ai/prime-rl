import asyncio
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Tool, ToolMessage, Usage, UserMessage, WeightVersionSpan

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.control.types import ConfigUpdate
from prime_rl.env.heads import EpisodeHead, TraceHead, episode_head
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.train_sink import TrainSink, queued_samples, write_token_advantages
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress, TrainBatch
from prime_rl.orchestrator.utils import episode_group_id, min_fresh_version
from prime_rl.transports import payload
from prime_rl.transports.batch.types import MicroBatchPlan, TrainingSample
from prime_rl.transports.payload import PayloadHook, build_sample, sample_url, wait_for_deletes
from prime_rl.utils.logger import setup_logger

setup_logger("warning")

# The env worker that keeps every test episode's samples.
HOOK = PayloadHook()


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
        info={"advantage": advantage} if sampled else {},
    )


def received(ep: vf.Episode, *, keep: bool = True) -> EpisodeHead:
    """The head of an episode as the orchestrator receives it: with ``keep``, an env worker
    kept its samples. A trace's ``advantage`` info stands for the credit its group earned."""
    if keep:
        HOOK.keep(ep, "rl")
    head = episode_head(ep)
    for trace_head, trace in zip(head.traces, ep.traces, strict=True):
        if "advantage" in trace.info:
            trace_head.info["advantage"] = trace.info["advantage"]
    return head


def whole_episode(traces, *, group="g", version=0) -> vf.Episode:
    """A train episode as the env worker that ran it holds it."""
    return vf.Episode(
        env=vf.EnvInfo(id="test", name="test"),
        task=traces[0].task,
        group=vf.GroupInfo(id=group),
        traces=traces,
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=version, end=version))),
    )


def episode(traces, *, group="g", version=0, keep=True) -> EpisodeHead:
    """A train episode as the orchestrator receives it."""
    return received(whole_episode(traces, group=group, version=version), keep=keep)


def train_trace(trace: TraceHead) -> list[TrainingSample]:
    """The samples a trainer rank builds from one trace an env worker kept, referenced with
    the trace's credit by the orchestrator."""
    assert not write_token_advantages([trace])
    refs = [
        sample.ref
        for sample in queued_samples(trace, env_name="test", temperature=1.0, policy_start=None, seq_len=1 << 20)
    ]
    plan = MicroBatchPlan(samples=refs, trained_version=0, min_version=0)
    return [build_sample(payload._fetch(sample_url(ref)), ref, plan) for ref in refs]


def trained(batch: TrainBatch, step: int, max_staleness: int = 1) -> list[TrainingSample]:
    """The samples a trainer rank builds from a batch the sink cut at ``step``."""
    plan = MicroBatchPlan(
        samples=batch.samples, trained_version=step - 1, min_version=min_fresh_version(step, max_staleness)
    )
    return [build_sample(payload._fetch(sample_url(ref)), ref, plan) for ref in batch.samples]


def sink(*, constant=True, admitted=True, batch_size=100, closed=None, group_size=None):
    config = OrchestratorConfig(batch_size=batch_size, constant_trainer_batch_size=constant, max_staleness=1)
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=1),
        sampling_at=lambda step: vf.SamplingConfig(temperature=1.0),
        requires_sampling_masks=False,
        requires_sampler_logprobs=False,
        algorithm=Algorithm(GRPOAlgoConfig(), None),
    )
    return TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env),
        progress=Progress(),
        batch_size=batch_size,
        on_result=lambda group: admitted,
        on_groups_closed=closed.extend if closed is not None else None,
        group_size=group_size,
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
            batch = await s.add(ep)
            assert batch is None or not batch.samples
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
        assert await s.add(ep) is None
        assert all(t.nodes[2].token_ids.tolist() == [3, 4] for t in discarded)
        del discarded, ep
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert [trace.id for trace in s.episode_by_trace[kept.id].traces] == [kept.id]
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
        assert await s.take_batch() is None
        gc.collect()
        assert ref() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == 0 and window.traces == 0
        assert window.metrics.to_dict()["staleness/rollout/dropped"] == 1
        assert s.take_rollout_window(force=True) is None

    asyncio.run(run())


def test_partially_stale_trace_ships_fresh_tokens():
    async def run():
        s = sink(batch_size=1)
        s.progress.step = 4
        t = trace()
        # Unlabelled tokens take the episode's dispatch version.
        t.nodes[2].weight_versions = [
            WeightVersionSpan(version="default", start=0, end=1),
            WeightVersionSpan(version="3", start=1, end=2),
        ]
        t.nodes[2].loss_weights = {"ref_kl": np.ones(2, dtype=np.float32)}
        batch = await s.add(episode([t], version=1))
        [sample] = trained(batch, step=4)
        assert sample.mask.tolist() == [False, False, False, True]
        assert sample.ref_kl_weights.tolist() == [0.0, 0.0, 0.0, 1.0]
        # Batch 4 trains on v3: the unlabelled token took dispatch version v1.
        assert sample.staleness.tolist() == [-1, -1, 2, 0]
        assert s.take_rollout_window(force=True).metrics.to_dict()["staleness/token/masked"] == 0.5

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
        assert list(zero.nodes[2].token_ids) == [3, 4]
        batch = await s.take_batch()
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
            await s.cancel(GroupCancellation("train", "test", f"cancel{i}", 1, 1, "overload"))
        assert s.take_rollout_window() is None
        window = s.take_rollout_window(force=True)
        assert window.attempts == window.discarded == 12
        assert window.cancelled == 4
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


def test_reconfigure_applies_new_batch_size_and_staleness():
    async def run():
        s = sink(batch_size=4)
        await s.add(episode([trace()], group="a"))
        await s.add(episode([trace()], group="b"))
        config = OrchestratorConfig(batch_size=2, max_staleness=1)
        s.reconfigure(ConfigUpdate(revision=1, step=1, config=config, changes=()))
        assert len((await s.take_batch()).samples) == 2

        s.progress.step = 2
        await s.add(episode([trace()], group="c"))
        config = OrchestratorConfig(batch_size=2, max_staleness=0)
        s.reconfigure(ConfigUpdate(revision=2, step=2, config=config, changes=()))
        await s.take_batch()
        assert not s.pending_batch

    asyncio.run(run())


def test_groups_close_when_they_finalize_without_shipping():
    async def run():
        closed = []
        rejected = sink(admitted=False, closed=closed)
        await rejected.add(episode([trace()], group="rejected"))
        assert closed == ["rejected"]

        s = sink(batch_size=2, closed=closed)
        await s.add(episode([trace(ok=False)], group="failed"))
        await s.cancel(GroupCancellation("train", "test", "overload", 1, 1, "overload"))
        assert closed == ["rejected", "failed", "overload"]

        # A shutdown drop re-runs on resume, so the group stays open.
        await s.cancel(GroupCancellation("train", "test", "shutdown", 1, 1, "shutdown"))
        assert closed == ["rejected", "failed", "overload"]

        s.progress.step = 4
        await s.add(episode([trace()], group="stale"))
        assert closed[-1] == "stale" and not s.pending_batch

        closed.clear()
        pruned = sink(constant=False, batch_size=1, closed=closed)
        batch = await pruned.add(episode([trace(advantage=0)], group="pruned"))
        assert not batch.samples and closed == ["pruned"]

    asyncio.run(run())


def test_shipped_groups_close_when_the_consumer_ships_them():
    async def run():
        closed = []
        s = sink(constant=False, batch_size=2, closed=closed)
        batch = await s.add(episode([trace(), trace(advantage=0), trace()], group="partial"))
        assert len(batch.samples) == 1 and len(s.pending_batch) == 1
        assert {episode_group_id(episode) for episode in batch.cohort.episodes} == {"partial"}
        # Taking the batch does not close it: a batch that never ships re-runs on resume.
        assert closed == []
        s.shipped(batch)
        assert closed == ["partial"]

    asyncio.run(run())


def test_unshipped_groups_stay_open_at_shutdown():
    async def run():
        closed = []
        s = sink(constant=False, batch_size=2, closed=closed)
        await s.add(episode([trace(), trace(advantage=0), trace()], group="partial"))
        s.discard_queued()
        assert closed == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("make_trace", "admitted", "reason"),
    [
        (lambda: trace(advantage=0), True, "zero advantage"),
        (lambda: trace(), False, "rejected by curriculum"),
        (lambda: trace(trainable=False), True, "no trainable survivors"),
    ],
)
def test_zero_output_warning_names_the_drop_reason(monkeypatch, make_trace, admitted, reason):
    warnings = []
    monkeypatch.setattr(
        "prime_rl.orchestrator.train_sink.get_logger",
        lambda: SimpleNamespace(debug=lambda message: None, warning=warnings.append),
    )
    s = sink(admitted=admitted, batch_size=4)

    async def run():
        for i in range(4):
            await s.add(episode([make_trace()], group=str(i)))

    asyncio.run(run())

    assert warnings == [
        f"No admitted train payload after 4 finalized units (1 zero-output batch equivalents), by drop reason: {reason}=4"
    ]


def test_open_group_closes_at_the_size_it_was_drawn_with():
    async def run():
        closed = []
        sizes = {"old": 2, "new": 3}
        s = sink(batch_size=100, closed=closed, group_size=sizes.get)
        # A live update raises the env's group size while group "old" is open.
        s.train_envs.get("test").config.group_size = 3
        for _ in range(2):
            await s.add(episode([trace()], group="old"))
        assert "old" not in s.pending_groups
        for _ in range(2):
            await s.add(episode([trace()], group="new"))
        assert len(s.pending_groups["new"]) == 2
        await s.add(episode([trace()], group="new"))
        assert "new" not in s.pending_groups
        # A group the source does not know takes the env's current size.
        for _ in range(2):
            await s.add(episode([trace()], group="unknown"))
        assert len(s.pending_groups["unknown"]) == 2

    asyncio.run(run())


@pytest.mark.parametrize("support", ["missing", "head", "replayed"])
def test_truncated_sampling_requires_replayed_support(support):
    async def run():
        t = trace()
        if support != "missing":
            t.nodes[-1].sampler_support = vf.SamplerSupport(
                ids=np.array([3, 4], dtype=np.int32),
                counts=np.array([1, 1], dtype=np.int32),
                replayed=np.full(2, support == "replayed", dtype=np.bool_),
                logprobs=np.array([-0.1, -0.2], dtype=np.float32),
            )
        ep = episode([t])
        s = sink()
        s.train_envs.get("test").requires_sampling_masks = True
        try:
            if support == "replayed":
                await s.add(ep)
                assert s.pending_batch
            else:
                with pytest.raises(RuntimeError, match="carry no sampling masks"):
                    await s.add(ep)
        finally:
            HOOK.discard(ep)
            wait_for_deletes()

    asyncio.run(run())
