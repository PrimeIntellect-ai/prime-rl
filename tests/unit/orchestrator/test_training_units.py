import uuid
from collections import defaultdict

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import TrainRollout, rollouts_for_logging
from prime_rl.transport import TrainingSample


def _sample(prompt_len: int, completion_mask: list[bool]) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(range(prompt_len)),
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(len(completion_mask))),
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * len(completion_mask),
        completion_temperatures=[1.0] * len(completion_mask),
        env_name="unset",
    )


def _rollout(
    *,
    samples: list[TrainingSample],
    env_name: str = "debate",
    reward: float = 0.0,
    advantage: float | None = None,
    is_filtered: bool = False,
    source_rollout_id: uuid.UUID | None = None,
) -> TrainRollout:
    rollout = TrainRollout(
        raw={"reward": reward, "trajectory": []},
        env_name=env_name,
        example_id=1,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        samples=samples,
        advantage=advantage,
        is_filtered=is_filtered,
        source_rollout_id=source_rollout_id,
    )
    for sample in samples:
        sample.reward = reward
        sample.advantage = advantage
        sample.env_name = env_name
        sample.training_mode = "rl"
    TrainSink._fill_token_usage_from_samples(rollout)
    return rollout


def _sink_for(rollouts: list[TrainRollout], episodes: list[TrainRollout] | None = None) -> TrainSink:
    sink = TrainSink.__new__(TrainSink)
    sink.batch_size = len(rollouts)
    sink.token_batch_size = None
    sink.pending_batch = list(rollouts)
    sink.pending_batch_tokens = sum(TrainSink.rollout_token_count(rollout) for rollout in rollouts)
    sink.post_filters = []
    sink.pending_episode_rollouts = {episode.rollout_id: episode for episode in episodes or []}
    sink.arrivals_by_env = defaultdict(int, {"debate": len(rollouts)})
    sink.errors_by_env = defaultdict(int)
    return sink


def test_process_batch_ships_training_units_and_logs_source_episode():
    episode = _rollout(samples=[], reward=0.5)
    member_a_sample = _sample(prompt_len=3, completion_mask=[True, True])
    member_b_sample = _sample(prompt_len=5, completion_mask=[True])
    member_a = _rollout(
        samples=[member_a_sample],
        reward=0.8,
        advantage=0.3,
        source_rollout_id=episode.rollout_id,
    )
    member_b = _rollout(
        samples=[member_b_sample],
        reward=0.2,
        advantage=-0.1,
        is_filtered=True,
        source_rollout_id=episode.rollout_id,
    )

    batch = _sink_for([member_a, member_b], episodes=[episode]).process_batch()

    assert batch.rollouts == [member_a, member_b]
    assert batch.samples == [member_a_sample]
    assert batch.samples[0].reward == 0.8
    assert batch.samples[0].advantage == 0.3
    assert batch.samples[0].env_name == "debate"
    assert batch.samples[0].training_mode == "rl"
    assert batch.metrics.rollout_prefill_lens == [3, 5]
    assert batch.metrics.rollout_decode_lens == [2, 1]
    assert batch.metrics.samples_per_rollout == [1, 1]
    assert batch.metrics.num_prefill_tokens == 8
    assert batch.metrics.num_decode_tokens == 3
    assert batch.metrics.n_trainable == 1
    assert batch.episode_rollouts == [episode]
    assert rollouts_for_logging(batch) == [episode]


def test_rollouts_for_logging_preserves_single_agent_rollouts_in_mixed_batches():
    single_agent = _rollout(samples=[_sample(prompt_len=2, completion_mask=[True])], env_name="math", reward=1.0)
    episode = _rollout(samples=[], reward=0.5)
    member_a = _rollout(samples=[_sample(prompt_len=3, completion_mask=[True])], source_rollout_id=episode.rollout_id)
    member_b = _rollout(samples=[_sample(prompt_len=4, completion_mask=[True])], source_rollout_id=episode.rollout_id)

    batch = _sink_for([single_agent, member_a, member_b], episodes=[episode]).process_batch()

    assert rollouts_for_logging(batch) == [single_agent, episode]


def test_process_batch_tracks_pending_tokens_for_token_batching():
    first = _rollout(samples=[_sample(prompt_len=3, completion_mask=[True, True])])
    second = _rollout(samples=[_sample(prompt_len=5, completion_mask=[True])])
    third = _rollout(samples=[_sample(prompt_len=7, completion_mask=[True, True, True])])
    sink = _sink_for([first, second, third])
    sink.batch_size = None
    sink.token_batch_size = 11

    assert sink.batch_progress() == (21, 11, "tokens")
    batch = sink.process_batch()

    assert batch.rollouts == [first, second]
    assert batch.metrics.num_prefill_tokens == 8
    assert batch.metrics.num_decode_tokens == 3
    assert sink.pending_batch == [third]
    assert sink.batch_progress() == (10, 11, "tokens")
