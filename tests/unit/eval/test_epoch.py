import asyncio
import gc
import weakref
from types import SimpleNamespace

import prime_runs as pr
import pytest
import verifiers.v1 as vf

from prime_rl import monitors
from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.eval.runner import EvalRunner
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_index_path, get_trace_stream
from prime_rl.orchestrator.annotations import stamp_batch
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation
from tests.unit.orchestrator.test_train_sink import episode, trace


@pytest.mark.parametrize("trainable", [True, False])
def test_eval_epoch_releases_episodes_and_matches_full_episode_reporting(monkeypatch, tmp_path, trainable):
    good = trace(trainable=trainable)
    good.metrics = {"overlap": 0.2, "missing": None}
    good.rewards = {"overlap": vf.Reward(score=0.5, weight=2.0), "null": None}
    judge = trace(trainable=False)
    judge.agent.name = "judge"
    bad = trace(ok=False, trainable=trainable)
    dropped = trace(trainable=trainable)
    dropped.drop("excluded")
    first = episode([good, judge])
    second = episode([bad, dropped])
    third = episode([trace(trainable=trainable)], group="another")
    third.traces[0].rewards["correct"].score = 0
    empty = episode([trace()])
    empty.traces = []
    empty.ok = False
    original = [first, second, third, empty]
    for item in original:
        item.run = vf.TrainRunInfo(id="run", work=vf.EvalWorkInfo(step=3, policy=vf.PolicySpan(start=2, end=2)))
    expected = EvalEpisodes(original, group_size=2)
    expected_metrics = {
        **expected.metrics.to_wandb(prefix="eval/test", subset="all"),
        **expected.effective.metrics.to_wandb(prefix="eval/test", subset="effective"),
    }
    expected_platform = pr.metrics.from_episodes(original)
    original_updates = stamp_batch(expected.effective.vf_episodes, 3)
    trace_ref = weakref.ref(good)
    episode_ref = weakref.ref(first)
    held = [first]
    del original, expected, good, first, bad, dropped, judge, item

    env = SimpleNamespace(examples=[None] * 3, config=SimpleNamespace(group_size=2))
    runner = EvalRunner.__new__(EvalRunner)
    runner.eval_sink = EvalSink(eval_envs=SimpleNamespace(get=lambda name: env))
    runner.eval_triggered_at = {}
    published = {}
    updates = []
    platform = {}

    async def log(data, step, **kwargs):
        if isinstance(data, dict):
            published.update(data)

    async def log_annotations(rows):
        updates.extend(rows)

    async def finish(env_name, step, rows):
        platform.update(pr.metrics.from_episodes(rows))

    writer = FileMonitor(FileMonitorConfig(compress=False))
    capture = SimpleNamespace(log=log, log_annotations=log_annotations, log_eval_epoch=finish)
    monkeypatch.setattr(monitors, "MONITORS", [writer, capture])

    async def land():
        # Drop the caller's last strong reference before the epoch finishes.
        await writer.init(tmp_path, producer="eval")
        pending = {"test"}
        await runner.land(held.pop(), pending)
        assert len(get_index_path(get_trace_stream(tmp_path)).read_bytes().splitlines()) == 1
        gc.collect()
        assert episode_ref() is None and trace_ref() is None
        assert not updates
        for item in (second, third, empty):
            await runner.land(item, pending)
        failure = DispatchFailure(
            "eval", "test", "g", 3, 2, "Task", "key", "hash", vf.Error(type="Transport", message="x")
        )
        assert runner.eval_sink.fail(failure) is None
        batch = runner.eval_sink.cancel(GroupCancellation("eval", "test", "g", 3, 1, "superseded"))
        assert batch is not None
        assert all(row.policy_version == 2 for row in batch.episodes)
        await runner.finalize_eval_batch(batch)
        await writer.finalize()

    asyncio.run(land())
    assert platform == expected_platform
    assert published == {
        **expected_metrics,
        "eval/test/all/dispatch_failure/mean": 1 / 6,
        "eval/test/all/dispatch_failure/Transport": 1.0,
        "eval/test/all/cancelled/count": 1.0,
        "eval/test/all/cancelled/mean": 1 / 6,
        "eval/test/policy_version": 3.0,
        "step": 3.0,
    }
    for actual, old in zip(updates, original_updates, strict=True):
        assert actual["info"]["ship"].pop("time") >= old["info"]["ship"].pop("time")
        for row in (actual, old):
            for branch in row["branches"]:
                branch["advantages"] = branch["advantages"].tolist()
        assert actual == old


def test_eval_file_write_failure_stops_before_episode_release(monkeypatch):
    writer = FileMonitor(FileMonitorConfig())
    writer.producer = "eval"

    async def disk_full(*args, **kwargs):
        raise OSError("no space")

    monkeypatch.setattr(writer, "log", disk_full)
    monkeypatch.setattr(monitors, "MONITORS", [writer])
    with pytest.raises(OSError, match="no space"):
        asyncio.run(monitors.log([episode([trace()])], 0, "eval", "all"))
