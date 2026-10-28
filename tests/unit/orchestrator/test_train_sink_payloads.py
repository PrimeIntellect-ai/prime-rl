"""Strict payload accounting in the train sink: every array an env server kept for a
delivered trace is either shipped in a batch (the trainer deletes it) or deleted by the
sink when it drops the sample or trace."""

import asyncio
from types import SimpleNamespace

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Progress
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY
from prime_rl.transports.payload import PayloadHook, sample_payload_urls, wait_for_deletes
from tests.unit.orchestrator.test_train_sink import episode, trace


def kept(hook: PayloadHook, **kwargs) -> vf.Trace:
    """A trace whose arrays an env server kept."""
    t = trace(**kwargs)
    for node in t.nodes:
        node.routed_experts = np.ones((len(node.token_ids), 2, 2), dtype=np.uint8)
    hook.keep(episode([t]), vf.SamplingConfig())
    return t


def payload_sink(*, constant=True, batch_size=100) -> TrainSink:
    config = OrchestratorConfig(batch_size=batch_size, constant_trainer_batch_size=constant, max_staleness=1)
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=1, sampling=SimpleNamespace(temperature=1.0)),
        requires_sampling_masks=False,
        requires_top_logprobs=False,
        algorithm=Algorithm(GRPOAlgoConfig(), None),
    )
    return TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env),
        progress=Progress(),
        batch_size=batch_size,
        on_result=lambda group: True,
    )


def settle(sink: TrainSink) -> None:
    wait_for_deletes()


def test_sink_deletes_exactly_the_arrays_it_drops():
    async def run():
        hook = PayloadHook()
        queued = kept(hook)
        dropped = [kept(hook, advantage=0), kept(hook, ok=False), kept(hook)]
        dropped[-1].drop("judge: outcome=task_defect")
        s = payload_sink()
        await s.add(episode([queued, *dropped]))
        settle(s)
        assert set(hook.server.blobs) == {queued.info[PAYLOAD_INFO_KEY]["branches"]["0"]["id"]}

    asyncio.run(run())


def test_sink_deletes_stale_pruned_and_discarded_arrays_but_not_shipped_ones():
    async def run():
        hook = PayloadHook()
        s = payload_sink(constant=False, batch_size=2)
        stale = kept(hook)
        await s.add(episode([stale], group="stale", version=0))
        s.progress.step = 4
        assert await s.take_batch() is None
        shipped, pruned = kept(hook), kept(hook, advantage=0)
        await s.add(episode([shipped], group="a", version=3))
        batch = await s.add(episode([pruned], group="b", version=3))
        leftover = kept(hook)
        await s.add(episode([leftover], group="c", version=3))
        s.discard_queued()
        settle(s)
        assert batch is not None
        ids = {url.rsplit("/", 1)[1] for url in sample_payload_urls(batch.samples)}
        assert ids == {shipped.info[PAYLOAD_INFO_KEY]["branches"]["0"]["id"]}
        assert set(hook.server.blobs) == ids

    asyncio.run(run())


def test_eval_sink_deletes_the_arrays_of_eval_episodes():
    """Env servers keep arrays for every episode; eval never trains, so its sink frees them."""
    hook = PayloadHook()
    evaluated = episode([kept(hook)])
    evaluated.run = vf.TrainRunInfo(id="run", work=vf.EvalWorkInfo(step=0))
    env = SimpleNamespace(examples=[None, None], config=SimpleNamespace(group_size=1))
    sink = EvalSink(eval_envs=SimpleNamespace(get=lambda name: env))
    assert sink.add(evaluated) is None
    wait_for_deletes()
    assert not hook.server.blobs


def test_dispatcher_deletes_the_arrays_of_returned_episodes_it_drops():
    """A dropped group's task may already have returned its episode: no sink sees it."""

    async def run() -> None:
        hook = PayloadHook()
        returned = episode([kept(hook)])
        dispatcher = Dispatcher.__new__(Dispatcher)

        async def done():
            return returned

        finished = asyncio.ensure_future(done())
        await finished
        running = asyncio.ensure_future(asyncio.sleep(60))
        await dispatcher._cancel_claimed([finished, running])
        wait_for_deletes()
        assert running.cancelled()
        assert not hook.server.blobs

    asyncio.run(run())


def test_dispatcher_skips_claimed_tasks_still_pending(monkeypatch):
    """`safe_cancel` can return with a task still pending when the caller is cancelled."""

    async def noop(tasks) -> None:
        pass

    async def run() -> None:
        dispatcher = Dispatcher.__new__(Dispatcher)
        monkeypatch.setattr("prime_rl.orchestrator.dispatcher.safe_cancel_all", noop)
        pending = asyncio.ensure_future(asyncio.sleep(60))
        await dispatcher._cancel_claimed([pending])
        pending.cancel()

    asyncio.run(run())
