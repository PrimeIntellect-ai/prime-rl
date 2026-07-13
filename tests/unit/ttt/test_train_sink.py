"""TTT TrainSink containment without changing ordinary Prime-RL failure behavior."""

import pytest

pytest.importorskip("verifiers")

from prime_rl.orchestrator.metrics import TrainRollouts  # noqa: E402
from prime_rl.orchestrator.train_sink import TrainSink  # noqa: E402
from prime_rl.orchestrator.types import Rollout  # noqa: E402
from prime_rl.transport import TrainingSample  # noqa: E402


def task(idx: int = 0):
    import verifiers.v1 as vf

    return vf.TraceTask(type="Task", data=vf.TaskData(idx=idx, prompt="t"))


def sample(n: int, env_name: str = "e") -> TrainingSample:
    return TrainingSample(
        token_ids=list(range(n)),
        mask=[False] + [True] * (n - 1),
        logprobs=[0.0] * n,
        temperatures=[1.0] * n,
        env_name=env_name,
    )


def make_bad_ttt_rollout() -> Rollout:
    """A rollout whose trace has a branch stamped ttt_version=2 but no recorded v2
    checkpoint — trace_to_samples raises ValueError on it."""
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, UserMessage

    rollout = Rollout(task=task(), env_name="e")
    parent = None
    for i, message in enumerate([UserMessage(content="u"), AssistantMessage(content="a")]):
        sampled = i == 1
        rollout.nodes.append(
            MessageNode(
                parent=parent,
                message=message,
                sampled=sampled,
                token_ids=[10 + i, 20 + i],
                mask=[False, sampled],
                logprobs=[-0.5] if sampled else [],
                ttt_version=2,
            )
        )
        parent = len(rollout.nodes) - 1
    rollout.info["ttt"] = {"updates": [{"version": 1, "ckpt_path": "/x"}]}
    return rollout


class FakeAlgorithm:
    def __init__(self):
        self.finalized = []

    async def finalize_rollout(self, rollout):
        self.finalized.append(rollout)


class FakeEnv:
    def __init__(self, *, ttt: bool = False):
        if ttt:
            active_ttt = type("TTT", (), {"enabled": True, "qa": None})()
            self.config = type("Cfg", (), {"ttt": active_ttt})()
        else:
            self.config = object()
        self.algorithm = FakeAlgorithm()
        self.sampling_args = {"temperature": 0.7}


class FakeTrainEnvs:
    def __init__(self, env):
        self._env = env

    def get(self, name):
        return self._env


def make_sink(env: FakeEnv) -> TrainSink:
    """Light TrainSink: only the attributes process_rollout touches."""
    sink = TrainSink.__new__(TrainSink)
    sink.train_envs = FakeTrainEnvs(env)
    sink.mm_token_type_ids_mapping = None
    return sink


@pytest.mark.asyncio
async def test_tokenization_failure_becomes_rollout_error():
    env = FakeEnv(ttt=True)
    sink = make_sink(env)
    rollout = make_bad_ttt_rollout()
    await sink.process_rollout(rollout)  # must not raise
    assert rollout.has_error
    assert rollout.error.type == "TokenizationError"
    assert rollout.samples == []
    # finalize_rollout must be skipped for the contained failure.
    assert env.algorithm.finalized == []


@pytest.mark.asyncio
async def test_plain_tokenization_failure_keeps_core_fail_fast_behavior():
    env = FakeEnv()
    sink = make_sink(env)
    with pytest.raises(ValueError, match="no checkpoint path is recorded"):
        await sink.process_rollout(make_bad_ttt_rollout())


class RaisingTokenizer:
    """apply_chat_template chokes (odd tool schemas) — recycling must skip, not error."""

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("unsupported tool schema")


class QAEnv:
    """Env with recycle_to_policy on; group finalization is a no-op."""

    class _QA:
        recycle_to_policy = True
        meta_lessons = False
        temperature = None

    class _TTT:
        enabled = True

    def __init__(self):
        self._TTT.qa = self._QA()
        self.config = type("Cfg", (), {"ttt": self._TTT()})()
        self.requires_group_scoring = False
        self.sampling_args = {"temperature": 0.7}

        class Algo:
            async def finalize_group(self, survivors):
                pass

        self.algorithm = Algo()


def make_qa_sink(env) -> TrainSink:
    """Light TrainSink with the group-processing state process_group touches."""
    sink = TrainSink.__new__(TrainSink)
    sink.train_envs = FakeTrainEnvs(env)
    sink.tokenizer = RaisingTokenizer()
    sink.pre_filters = []
    sink.pending_batch = []
    sink.pending_tokens = 0
    sink.token_batch_size = None
    sink.pre_filter_seen = 0
    sink.pre_filter_dropped = 0
    sink.pre_filter_dropped_by_name = {}
    sink.pending_groups = {}
    sink._meta_clients = {}
    return sink


@pytest.mark.asyncio
async def test_qa_recycle_failure_skips_rollout_not_group():
    import uuid

    env = QAEnv()
    sink = make_qa_sink(env)

    rollout = Rollout(task=task(), env_name="e")
    rollout.info["ttt"] = {"updates": [{"version": 1, "qa_pairs": [{"question": "q?", "answer": "a"}]}]}
    group_id = uuid.uuid4()
    sink.pending_groups[group_id] = [rollout]

    await sink.process_group(group_id)  # must not raise
    # Recycling was skipped (enrichment) but the rollout still reached the batch.
    assert rollout.samples == []
    assert sink.pending_batch == [rollout]


@pytest.mark.asyncio
async def test_meta_lesson_failure_skips_group_not_run():
    """The A5 path gets the same containment as recycling: a provider failure must skip
    meta lessons, increment the dropped metric, and never kill the group/main loop."""
    import uuid

    env = QAEnv()
    env.config.ttt.qa.recycle_to_policy = False
    env.config.ttt.qa.meta_lessons = True

    class ExplodingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("meta provider unavailable")

    class ExplodingClient:
        chat = type("Chat", (), {"completions": ExplodingCompletions()})()

    class Pool:
        model_name = "m"

    env.sampler = type("S", (), {"pool": Pool()})()

    sink = make_qa_sink(env)
    sink._meta_clients = {"e": ExplodingClient()}
    sink.meta_groups_ok = 0
    sink.meta_groups_dropped = 0

    group_id = uuid.uuid4()
    rollouts = []
    for i in range(2):  # meta extraction needs >= 2 survivors with pairs
        rollout = Rollout(task=task(), env_name="e")
        rollout.info["ttt"] = {"updates": [{"version": 1, "qa_pairs": [{"question": f"q{i}?", "answer": "a"}]}]}
        rollouts.append(rollout)
    sink.pending_groups[group_id] = rollouts

    await sink.process_group(group_id)  # must not raise
    assert sink.pending_batch == rollouts  # the group still ships
    assert all(r.samples == [] for r in rollouts)  # no meta samples, no corruption
    # The drop is counted (surfaced as ttt/meta_groups_dropped): a silently rising
    # dropped rate is the arm quietly running without lessons.
    assert (sink.meta_groups_ok, sink.meta_groups_dropped) == (0, 1)


@pytest.mark.asyncio
async def test_disabled_ttt_runs_no_qa_to_policy_paths():
    """ttt.enabled=false (the wiring ablation) must run neither recycling nor meta
    lessons even with both QA flags set — same predicate as validate_ttt."""
    import uuid

    env = QAEnv()
    env.config.ttt.enabled = False
    env.config.ttt.qa.recycle_to_policy = True

    sink = make_qa_sink(env)  # RaisingTokenizer would raise if recycling ran uncontained

    recycle_calls = []
    rollout = Rollout(task=task(), env_name="e")
    rollout.info["ttt"] = {"updates": [{"version": 1, "qa_pairs": [{"question": "q?", "answer": "a"}]}]}
    group_id = uuid.uuid4()
    sink.pending_groups[group_id] = [rollout]

    import prime_rl.orchestrator.train_sink as ts

    original = ts.qa_recycle_samples
    ts.qa_recycle_samples = lambda *a, **k: recycle_calls.append(a) or []
    try:
        await sink.process_group(group_id)
    finally:
        ts.qa_recycle_samples = original
    assert recycle_calls == []  # disabled TTT: recycling never invoked
    assert sink.pending_batch == [rollout]


class TTTEnv:
    config = type("Cfg", (), {"ttt": type("TTT", (), {"enabled": True})()})()


def test_token_batches_use_ttt_payload_but_leave_core_accounting_unchanged():
    """Recycled/auxiliary samples count toward TTT batches; plain Prime-RL keeps its
    established trace-token accounting."""
    ttt_sink = TrainSink.__new__(TrainSink)
    ttt_sink.train_envs = FakeTrainEnvs(TTTEnv())
    ttt_sink.batch_size = None
    ttt_sink.token_batch_size = 6
    ttt_sink.post_filters = []
    ttt_sink._group_meta_samples = {}

    first = Rollout(task=task(), env_name="e", samples=[sample(7)])
    second = Rollout(task=task(), env_name="e", samples=[sample(7)])
    ttt_sink.pending_batch = [first, second]
    ttt_sink.pending_tokens = 14
    ttt_sink.pending_rollouts = TrainRollouts([first, second])

    batch = ttt_sink.process_batch()
    assert batch.samples == first.samples
    assert ttt_sink.pending_batch == [second]
    assert ttt_sink.pending_tokens == 7

    core_sink = TrainSink.__new__(TrainSink)
    core_sink.train_envs = FakeTrainEnvs(FakeEnv())
    ordinary = Rollout(task=task(), env_name="e", samples=[sample(7)])
    assert core_sink._rollout_batch_tokens(ordinary) == ordinary.num_total_tokens == 0


def test_meta_samples_wait_for_a_post_filter_group_survivor(monkeypatch):
    """A5 lessons survive a batch split when the first group member is filtered."""
    import prime_rl.orchestrator.train_sink as train_sink_module

    group_id = __import__("uuid").uuid4()
    first = Rollout(task=task(), env_name="e", group_id=group_id, samples=[sample(2)])
    second = Rollout(task=task(), env_name="e", group_id=group_id, samples=[sample(3)])
    lesson = sample(4)

    sink = TrainSink.__new__(TrainSink)
    sink.train_envs = FakeTrainEnvs(TTTEnv())
    sink.batch_size = 1
    sink.token_batch_size = None
    sink.post_filters = [object()]
    sink.pending_batch = [first, second]
    sink.pending_tokens = 0
    sink.pending_rollouts = TrainRollouts([first, second])
    sink._group_meta_samples = {group_id: [lesson]}

    def filter_first(_filters, cohort):
        if cohort == [first]:
            first.is_filtered = True

    monkeypatch.setattr(train_sink_module, "apply_filters", filter_first)

    empty = sink.process_batch()
    assert empty.samples == []
    assert sink._group_meta_samples[group_id] == [lesson]

    shipped = sink.process_batch()
    assert shipped.samples == [*second.samples, lesson]
    assert group_id not in sink._group_meta_samples
