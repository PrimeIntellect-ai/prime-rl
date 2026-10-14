"""Several streaming sources share the dispatcher: ``EvalSource.next_task`` hands out
their tasks in turn, so a stream that always has a task ready cannot starve another."""

import itertools
import time
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.task_feed import TaskFeed

TIMEOUT_S = 10


def _stream(name: str, tasks) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, examples=[], feed=TaskFeed(tasks, name=name), config=SimpleNamespace(group_size=1)
    )


def _endless(prefix: str):
    for i in itertools.count():
        yield vf.Task(vf.TaskData(id=f"{prefix}-{i}"))


def _source(*streams: SimpleNamespace) -> EvalSource:
    envs = EvalEnvs.__new__(EvalEnvs)
    envs._envs = {stream.name: stream for stream in streams}
    source = EvalSource(envs)
    source.trigger(0)
    return source


def _take(source: EvalSource, n: int) -> list[str]:
    taken: list[str] = []
    deadline = time.monotonic() + TIMEOUT_S
    while len(taken) < n and time.monotonic() < deadline:
        if (request := source.next_task()) is not None:
            taken.append(request.env_name)
        else:
            time.sleep(0.001)  # the feeds read one task ahead on their own threads
    return taken


def test_two_ready_streams_take_turns() -> None:
    streams = _stream("a", _endless("a")), _stream("b", _endless("b"))
    source = _source(*streams)

    taken = []
    for _ in range(10):
        deadline = time.monotonic() + TIMEOUT_S
        # both feeds have their next task read ahead: either could give it
        while any(stream.feed._queue.empty() for stream in streams) and time.monotonic() < deadline:
            time.sleep(0.001)
        taken.append(source.next_task().env_name)

    assert taken == ["a", "b"] * 5


def test_a_stream_with_nothing_ready_does_not_hold_up_the_other() -> None:
    source = _source(_stream("a", _endless("a")), _stream("b", iter([])))

    assert _take(source, 5) == ["a"] * 5
