from types import SimpleNamespace

from prime_rl.orchestrator.train_source import TrainSource


class _CountingStream:
    """Stands in for an infinite taskset generator backed by live external
    state: every next() is a real draw the source must not replay."""

    def __init__(self):
        self.pulls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.pulls += 1
        return SimpleNamespace(data=SimpleNamespace(idx=self.pulls))


def _infinite_env(stream: _CountingStream, name: str = "stream"):
    return SimpleNamespace(name=name, tasks=stream, num_tasks=None, config=SimpleNamespace(ratio=1))


def _finite_env(name: str = "table", n: int = 5):
    tasks = [SimpleNamespace(data=SimpleNamespace(idx=i)) for i in range(n)]
    return SimpleNamespace(name=name, tasks=tasks, num_tasks=n, config=SimpleNamespace(ratio=1))


def test_infinite_env_resume_does_not_replay_the_generator():
    stream = _CountingStream()
    source = TrainSource([_infinite_env(stream)])
    for _ in range(7):
        source.next_example()
    assert stream.pulls == 7
    state = source.state_dict()

    fresh_stream = _CountingStream()
    resumed = TrainSource([_infinite_env(fresh_stream)])
    resumed.load_state_dict(state)
    assert fresh_stream.pulls == 0, "resume must not consume the generator"
    assert resumed.cursors["stream"] == 7

    resumed.next_example()
    assert fresh_stream.pulls == 1
    assert resumed.cursors["stream"] == 8


def test_finite_env_resume_replays_epoch_shuffle_and_position():
    source = TrainSource([_finite_env()])
    drawn = [source.next_example()["task"].data.idx for _ in range(3)]
    state = source.state_dict()

    resumed = TrainSource([_finite_env()])
    resumed.load_state_dict(state)
    continued = [resumed.next_example()["task"].data.idx for _ in range(2)]
    baseline = [source.next_example()["task"].data.idx for _ in range(2)]
    assert continued == baseline
    assert len(set(drawn)) == 3
