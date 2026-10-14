"""A streaming eval epoch reaches the platform window by window: the Prime monitor
folds each window into one summary, equal to ``prime_runs``' summary of every episode."""

import prime_runs as pr
import pytest
import verifiers.v1 as vf

from prime_rl.monitors.prime import EvalSummary


def _episode(ok: bool, *seats: tuple[bool, float, float | None]) -> vf.Episode:
    """An episode with one trace per seat: ``(trainable, score, metric)``."""
    task = vf.TraceTask(type="Task", data=vf.WireTaskData(id="t"), key="t", hash="h")
    traces = [
        vf.Trace(
            task=task,
            agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=trainable),
            nodes=[],
            rewards={"solved": vf.Reward(score=score)},
            metrics={} if metric is None else {"turns": metric},
            ok=ok,
        )
        for trainable, score, metric in seats
    ]
    return vf.Episode(task=task, traces=traces, ok=ok)


@pytest.mark.parametrize("judged", [True, False])
def test_windows_fold_into_the_summary_of_every_episode(judged: bool) -> None:
    episodes = [
        _episode(True, (True, 1.0, 3.0), (not judged, 0.5, None)),
        _episode(False, (True, 0.0, 5.0)),
        _episode(True, (True, 1.0, None), (not judged, 0.0, 2.0)),
    ]
    fold = EvalSummary()
    fold.add(episodes[:2])
    fold.add([])
    fold.add(episodes[2:])

    assert fold.summary() == pr.metrics.from_episodes(episodes)


def test_an_epoch_with_no_trainable_trace_scores_every_trace() -> None:
    episodes = [_episode(True, (False, 1.0, None)), _episode(True, (False, 0.0, None))]
    fold = EvalSummary()
    for episode in episodes:
        fold.add([episode])

    assert fold.summary() == pr.metrics.from_episodes(episodes)
    assert fold.summary()["avg_reward"] == 0.5
