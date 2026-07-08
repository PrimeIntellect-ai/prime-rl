"""TrainSink failure containment: one bad rollout (tokenization ValueError) or a
choking QA-recycle chat template must never propagate to the orchestrator main loop."""

import pytest

pytest.importorskip("verifiers")

from prime_rl.orchestrator.train_sink import TrainSink  # noqa: E402
from prime_rl.orchestrator.types import Rollout  # noqa: E402


def make_bad_ttt_rollout() -> Rollout:
    """A rollout whose trace has a branch stamped ttt_version=2 but no recorded v2
    checkpoint — trace_to_samples raises ValueError on it."""
    import verifiers.v1 as vf
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, UserMessage

    rollout = Rollout(task=vf.Task(idx=0, prompt="t"), env_name="e")
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
    def __init__(self):
        self.config = object()  # no .ttt attr — qa_temperature resolves to None
        self.algorithm = FakeAlgorithm()


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
    env = FakeEnv()
    sink = make_sink(env)
    rollout = make_bad_ttt_rollout()
    await sink.process_rollout(rollout)  # must not raise
    assert rollout.has_error
    assert rollout.error.type == "TokenizationError"
    assert rollout.samples == []
    # finalize_rollout must be skipped for the contained failure.
    assert env.algorithm.finalized == []


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


@pytest.mark.asyncio
async def test_qa_recycle_failure_skips_rollout_not_group():
    import uuid

    import verifiers.v1 as vf

    env = QAEnv()
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

    rollout = Rollout(task=vf.Task(idx=0, prompt="t"), env_name="e")
    rollout.info["ttt"] = {"updates": [{"version": 1, "qa_pairs": [{"question": "q?", "answer": "a"}]}]}
    group_id = uuid.uuid4()
    sink.pending_groups[group_id] = [rollout]

    await sink.process_group(group_id)  # must not raise
    # Recycling was skipped (enrichment) but the rollout still reached the batch.
    assert rollout.samples == []
    assert sink.pending_batch == [rollout]


@pytest.mark.asyncio
async def test_meta_lesson_failure_skips_group_not_run():
    """The A5 path gets the same containment as recycling: a chat-template failure in
    meta_lesson_samples (or client construction) must skip meta lessons, never kill the
    group / propagate to the main loop."""
    import uuid

    import verifiers.v1 as vf

    env = QAEnv()
    env.config.ttt.qa.recycle_to_policy = False
    env.config.ttt.qa.meta_lessons = True

    class ExplodingPool:
        model_name = "m"

        async def get_eval_client(self):
            raise RuntimeError("no eval client configured")

    env.sampler = type("S", (), {"pool": ExplodingPool()})()

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

    group_id = uuid.uuid4()
    rollouts = []
    for i in range(2):  # meta extraction needs >= 2 survivors with pairs
        rollout = Rollout(task=vf.Task(idx=0, prompt="t"), env_name="e")
        rollout.info["ttt"] = {"updates": [{"version": 1, "qa_pairs": [{"question": f"q{i}?", "answer": "a"}]}]}
        rollouts.append(rollout)
    sink.pending_groups[group_id] = rollouts

    await sink.process_group(group_id)  # must not raise
    assert sink.pending_batch == rollouts  # the group still ships
    assert all(r.samples == [] for r in rollouts)  # no meta samples, no corruption


@pytest.mark.asyncio
async def test_disabled_ttt_runs_no_qa_to_policy_paths():
    """ttt.enabled=false (the wiring ablation) must run neither recycling nor meta
    lessons even with both QA flags set — same predicate as validate_ttt."""
    import uuid

    import verifiers.v1 as vf

    env = QAEnv()
    env.config.ttt.enabled = False
    env.config.ttt.qa.recycle_to_policy = True

    sink = TrainSink.__new__(TrainSink)
    sink.train_envs = FakeTrainEnvs(env)
    sink.tokenizer = RaisingTokenizer()  # would raise if recycling ran and weren't contained
    sink.pre_filters = []
    sink.pending_batch = []
    sink.pending_tokens = 0
    sink.token_batch_size = None
    sink.pre_filter_seen = 0
    sink.pre_filter_dropped = 0
    sink.pre_filter_dropped_by_name = {}
    sink.pending_groups = {}
    sink._meta_clients = {}

    recycle_calls = []
    rollout = Rollout(task=vf.Task(idx=0, prompt="t"), env_name="e")
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
