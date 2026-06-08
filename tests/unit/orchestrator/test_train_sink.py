from types import SimpleNamespace

from prime_rl.orchestrator.train_sink import TrainSink, _sample_has_trainable_tokens
from prime_rl.transport.types import TrainingSample


def _sample(completion_mask: list[bool], term_advantage: list[float] | None = None) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=completion_mask,
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
        env_name="test-env",
        term_advantages={"echo": term_advantage} if term_advantage is not None else None,
    )


def test_sample_trainable_with_primary_tokens():
    assert _sample_has_trainable_tokens(_sample([True, False]))


def test_sample_trainable_with_overlay_only():
    # No primary (completion) tokens, but an overlay token past position 0 -> still trains.
    assert _sample_has_trainable_tokens(_sample([False, False], term_advantage=[0.0, 0.0, 0.5, 0.0]))


def test_sample_not_trainable_without_primary_or_overlay():
    assert not _sample_has_trainable_tokens(_sample([False, False]))
    assert not _sample_has_trainable_tokens(_sample([False, False], term_advantage=[0.0, 0.0, 0.0, 0.0]))


def test_sample_zero_weight_overlay_not_trainable():
    # A zero overlay weight (e.g. an advantage-weighted overlay scaled by zero advantage) yields no
    # gradient, so it must not keep an otherwise-empty batch alive.
    assert not _sample_has_trainable_tokens(_sample([False, False], term_advantage=[0.0, 0.0, 0.0, 0.0]))


def test_sample_overlay_at_position_zero_is_not_trainable():
    # Position 0 has no shifted current-token logprob, so an alpha there never trains; matches the
    # trainer's overlay mask construction, which skips index 0.
    assert not _sample_has_trainable_tokens(_sample([False, False], term_advantage=[0.5, 0.0, 0.0, 0.0]))


def test_rl_primary_disabled_in_non_rl_modes():
    sink = TrainSink.__new__(TrainSink)
    sink._rl_primary_cache = {}
    sink.config = SimpleNamespace(training_mode="sft")
    assert not sink._rl_primary_enabled("test-env")

    sink = TrainSink.__new__(TrainSink)
    sink._rl_primary_cache = {}
    sink.config = SimpleNamespace(training_mode="opd")
    assert not sink._rl_primary_enabled("test-env")
