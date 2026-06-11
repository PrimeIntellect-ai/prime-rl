"""Sink-level tests for filter actions: ``process_group`` ordering (penalty
visible to the advantage baseline + sample propagation) and ``process_batch``
post-filter sample re-sync.

``TrainSink``'s heavy constructor args (tokenizer / renderer / real envs)
are only used by ``add()`` / ``process_rollout``; these tests bypass them by
pre-building ``rollout.samples`` and driving ``process_group`` /
``process_batch`` directly.
"""

import math
import uuid
from types import SimpleNamespace

from prime_rl.configs.orchestrator import DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import setup_advantage_fn
from prime_rl.orchestrator.filters import GibberishFilter, ZeroAdvantageFilter
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import TrainRollout
from prime_rl.transport import TrainingSample

VOCAB_SIZE = 128_000


def _make_gibberish_filter(action="penalize", penalty_reward=-1.0):
    return GibberishFilter(
        name="gibberish",
        token_id_threshold=100_000,
        logprob_threshold=-math.log(VOCAB_SIZE) - 2.0,
        action=action,
        penalty_reward=penalty_reward,
    )


def _make_sample(reward=None):
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[True, True],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        env_name="test",
        reward=reward,
    )


def _make_rollout(*, dirty: bool, reward: float = 1.0, group_id=None, gibberish_filter=None) -> TrainRollout:
    """Rollout with one pre-built sample; ``dirty=True`` triggers the
    gibberish filter (rare token at high entropy)."""
    if dirty:
        assert gibberish_filter is not None
        completion_ids = [120_000]
        completion_logprobs = [gibberish_filter.logprob_threshold - 1.0]
    else:
        completion_ids = [50]
        completion_logprobs = [-1.0]
    raw = {
        "trajectory": [
            {
                "tokens": {
                    "completion_ids": completion_ids,
                    "completion_logprobs": completion_logprobs,
                    "completion_mask": [1] * len(completion_ids),
                }
            }
        ],
        "reward": reward,
        "stop_condition": None,
        "metrics": {},
    }
    rollout = TrainRollout(
        raw=raw,
        env_name="test",
        example_id=0,
        group_id=group_id or uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
    )
    rollout.samples = [_make_sample(reward=None)]
    return rollout


class _FakeEnv:
    def __init__(self, group_size=2):
        self.config = SimpleNamespace(group_size=group_size)
        self.requires_group_scoring = False
        self.advantage_fn = setup_advantage_fn(DefaultAdvantageConfig())
        self.sampling_args = {"temperature": 1.0}


class _FakeTrainEnvs:
    def __init__(self, env: _FakeEnv):
        self._env = env

    def get(self, name: str) -> _FakeEnv:
        return self._env


def _make_sink(*, pre_filters=(), post_filters=(), batch_size=1) -> TrainSink:
    return TrainSink(
        config=SimpleNamespace(training_mode="rl"),  # process_group reads only training_mode
        tokenizer=None,
        renderer=None,
        train_envs=_FakeTrainEnvs(_FakeEnv()),
        mm_token_type_ids_mapping=None,
        batch_size=batch_size,
        token_batch_size=None,
        pre_filters=list(pre_filters),
        post_filters=list(post_filters),
    )


# --- process_group: pre-advantage penalty ordering ---


def test_process_group_penalize_lands_before_advantage_and_samples():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    sink = _make_sink(pre_filters=[gibberish_filter, ZeroAdvantageFilter(name="zero_advantage", action="drop")])

    group_id = uuid.uuid4()
    clean = _make_rollout(dirty=False, reward=1.0, group_id=group_id)
    dirty = _make_rollout(dirty=True, reward=1.0, group_id=group_id, gibberish_filter=gibberish_filter)
    sink.pending_groups[group_id] = [clean, dirty]

    sink.process_group(group_id)

    # Penalty applied before the group baseline: rewards (1.0, -1.0) → advantages (+1, -1)
    assert dirty.reward == -1.0
    assert dirty.raw_reward == 1.0
    assert clean.advantage == 1.0
    assert dirty.advantage == -1.0

    # Samples stamped with post-penalty reward and advantage
    assert clean.samples[0].reward == 1.0
    assert clean.samples[0].advantage == 1.0
    assert dirty.samples[0].reward == -1.0
    assert dirty.samples[0].advantage == -1.0

    # Penalized rollout stays trainable; nonzero advantages → no drops
    assert sink.pending_batch == [clean, dirty]
    assert sink.pre_filter_seen == 2
    assert sink.pre_filter_dropped == 0


def test_process_group_equally_penalized_group_dropped_by_zero_advantage():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    sink = _make_sink(pre_filters=[gibberish_filter, ZeroAdvantageFilter(name="zero_advantage", action="drop")])

    group_id = uuid.uuid4()
    rollouts = [
        _make_rollout(dirty=True, reward=1.0, group_id=group_id, gibberish_filter=gibberish_filter) for _ in range(2)
    ]
    sink.pending_groups[group_id] = rollouts

    sink.process_group(group_id)

    # Both capped to the same reward → zero advantages → post-advantage drop
    for rollout in rollouts:
        assert rollout.reward == -1.0
        assert rollout.advantage == 0.0
        assert rollout.is_filtered is True
    assert sink.pending_batch == []
    assert sink.pre_filter_dropped == 2
    # Drop attribution counts only drop-action filters, not the penalty
    assert sink.pre_filter_dropped_by_name == {"zero_advantage": 2}


# --- process_batch: post-batch penalize re-syncs samples ---


def test_process_batch_penalize_resyncs_sample_reward():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    sink = _make_sink(post_filters=[gibberish_filter], batch_size=2)

    clean = _make_rollout(dirty=False, reward=1.0)
    dirty = _make_rollout(dirty=True, reward=1.0, gibberish_filter=gibberish_filter)
    # Simulate process_group's propagation: samples stamped with pre-penalty reward
    for rollout in (clean, dirty):
        rollout.samples[0].reward = rollout.reward
        rollout.samples[0].advantage = 0.5
    sink.pending_batch = [clean, dirty]

    batch = sink.process_batch()

    # Shipped samples agree with the (penalized) rollout reward used in metrics
    assert dirty.reward == -1.0
    assert dirty.samples[0].reward == -1.0
    assert clean.samples[0].reward == 1.0
    # Penalize keeps the rollout trainable; advantage is metadata-only post-batch
    assert dirty.is_filtered is False
    assert dirty.samples[0].advantage == 0.5
    assert batch.metrics.n_trainable == 2
    assert len(batch.samples) == 2


def test_process_batch_drop_still_excludes_samples():
    gibberish_filter = _make_gibberish_filter(action="drop")
    sink = _make_sink(post_filters=[gibberish_filter], batch_size=2)

    clean = _make_rollout(dirty=False, reward=1.0)
    dirty = _make_rollout(dirty=True, reward=1.0, gibberish_filter=gibberish_filter)
    for rollout in (clean, dirty):
        rollout.samples[0].reward = rollout.reward
    sink.pending_batch = [clean, dirty]

    batch = sink.process_batch()

    assert dirty.is_filtered is True
    assert dirty.reward == 1.0  # drop leaves reward untouched
    assert batch.metrics.n_trainable == 1
    assert len(batch.samples) == 1
