from prime_rl.orchestrator.train_sink import _sample_has_trainable_tokens
from prime_rl.transport.types import TrainingSample


def _sample(completion_mask: list[bool], echo_alpha: list[float | None] | None = None) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=completion_mask,
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
        env_name="test-env",
        echo_alpha=echo_alpha,
    )


def test_sample_trainable_with_primary_tokens():
    assert _sample_has_trainable_tokens(_sample([True, False]))


def test_sample_trainable_with_echo_only():
    # No primary (completion) tokens, but an echo token past position 0 -> still trains.
    assert _sample_has_trainable_tokens(_sample([False, False], echo_alpha=[None, None, 0.5, None]))


def test_sample_not_trainable_without_primary_or_echo():
    assert not _sample_has_trainable_tokens(_sample([False, False]))
    assert not _sample_has_trainable_tokens(_sample([False, False], echo_alpha=[None, None, None, None]))


def test_sample_echo_at_position_zero_is_not_trainable():
    # Position 0 has no shifted current-token logprob, so an alpha there never trains; matches the
    # trainer's echo_mask construction, which skips index 0.
    assert not _sample_has_trainable_tokens(_sample([False, False], echo_alpha=[0.5, None, None, None]))
