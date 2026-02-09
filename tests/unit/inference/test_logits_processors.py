import math

import torch

from prime_rl.inference.vllm.logits_processors.gibberish import GibberishDetector
from prime_rl.inference.vllm.logits_processors.repetition import RepetitionDetector

VOCAB_SIZE = 128_256
EOS_TOKEN_ID = 0
TOKEN_ID_THRESHOLD = 100_000
LOGPROB_THRESHOLD = -math.log(VOCAB_SIZE) - 2.0


def make_uniform_logits(vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Logits where every token has equal probability."""
    return torch.zeros(vocab_size)


def make_peaked_logits(token_id: int, peak_logit: float = 20.0, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Logits where one token has very high probability."""
    logits = torch.zeros(vocab_size)
    logits[token_id] = peak_logit
    return logits


# --- GibberishDetector tests ---


def test_gibberish_first_step_saves_logits():
    detector = GibberishDetector(EOS_TOKEN_ID, TOKEN_ID_THRESHOLD, LOGPROB_THRESHOLD)
    logits = make_uniform_logits()
    result = detector([], logits)

    assert not detector.aborted
    assert detector.prev_logits is not None
    assert result is logits


def test_gibberish_normal_token_passes_through():
    detector = GibberishDetector(EOS_TOKEN_ID, TOKEN_ID_THRESHOLD, LOGPROB_THRESHOLD)

    # Step 0: save logits
    logits_0 = make_peaked_logits(token_id=50, peak_logit=10.0)
    detector([], logits_0)

    # Step 1: token 50 was sampled (low id, high prob) — no abort
    logits_1 = make_uniform_logits()
    result = detector([50], logits_1)

    assert not detector.aborted
    assert result is logits_1


def test_gibberish_triggers_on_rare_low_prob_token():
    detector = GibberishDetector(EOS_TOKEN_ID, TOKEN_ID_THRESHOLD, LOGPROB_THRESHOLD)

    # Step 0: uniform logits — every token has logprob = -log(vocab_size) ≈ -11.76
    # which is > threshold (-11.76 - 2.0 = -13.76), so uniform alone doesn't trigger.
    # Use logits that make a high-id token have very low probability.
    logits_0 = torch.zeros(VOCAB_SIZE)
    logits_0[110_000] = -50.0  # This token will have extremely low probability
    detector([], logits_0)

    # Step 1: token 110_000 was sampled — high id AND low logprob → trigger
    logits_1 = make_uniform_logits()
    result = detector([110_000], logits_1)

    assert detector.aborted
    # EOS should be forced: all -inf except EOS
    assert result[EOS_TOKEN_ID] == 0.0
    assert result[1].item() == float("-inf")


def test_gibberish_no_trigger_for_low_id_token():
    detector = GibberishDetector(EOS_TOKEN_ID, TOKEN_ID_THRESHOLD, LOGPROB_THRESHOLD)

    # Step 0: uniform logits with low prob for token 500
    logits_0 = torch.zeros(VOCAB_SIZE)
    logits_0[500] = -50.0
    detector([], logits_0)

    # Step 1: token 500 was sampled — low id, so no trigger even though prob is low
    logits_1 = make_uniform_logits()
    result = detector([500], logits_1)

    assert not detector.aborted
    assert result is logits_1


def test_gibberish_no_trigger_for_high_prob_rare_token():
    detector = GibberishDetector(EOS_TOKEN_ID, TOKEN_ID_THRESHOLD, LOGPROB_THRESHOLD)

    # Step 0: peaked logits where token 110_000 has high probability
    logits_0 = make_peaked_logits(token_id=110_000, peak_logit=20.0)
    detector([], logits_0)

    # Step 1: token 110_000 was sampled — high id BUT high prob → no trigger
    logits_1 = make_uniform_logits()
    result = detector([110_000], logits_1)

    assert not detector.aborted
    assert result is logits_1


def test_gibberish_aborted_stays_sticky():
    detector = GibberishDetector(EOS_TOKEN_ID, TOKEN_ID_THRESHOLD, LOGPROB_THRESHOLD)
    detector.aborted = True

    logits = make_uniform_logits()
    result = detector([42], logits)

    assert detector.aborted
    assert result[EOS_TOKEN_ID] == 0.0
    assert result[1].item() == float("-inf")


# --- RepetitionDetector tests ---


def test_repetition_normal_steps_no_abort():
    detector = RepetitionDetector(EOS_TOKEN_ID, window=5, prob_threshold=0.99)

    # Uniform logits → max_prob ≈ 1/vocab_size, well below 0.99
    for _ in range(10):
        logits = make_uniform_logits()
        result = detector(list(range(10)), logits)
        assert not detector.aborted
        assert result is logits


def test_repetition_triggers_after_window():
    detector = RepetitionDetector(EOS_TOKEN_ID, window=5, prob_threshold=0.99)

    # Peaked logits → max_prob ≈ 1.0 > 0.99
    for i in range(4):
        logits = make_peaked_logits(token_id=42, peak_logit=20.0)
        result = detector([42] * (i + 1), logits)
        assert not detector.aborted

    # 5th step triggers
    logits = make_peaked_logits(token_id=42, peak_logit=20.0)
    result = detector([42] * 5, logits)
    assert detector.aborted
    assert result[EOS_TOKEN_ID] == 0.0
    assert result[1].item() == float("-inf")


def test_repetition_counter_resets_on_low_prob():
    detector = RepetitionDetector(EOS_TOKEN_ID, window=5, prob_threshold=0.99)

    # 4 consecutive high-prob steps
    for i in range(4):
        logits = make_peaked_logits(token_id=42, peak_logit=20.0)
        detector([42] * (i + 1), logits)
    assert detector.consecutive_count == 4

    # Break the streak with uniform logits
    logits = make_uniform_logits()
    detector([42] * 5, logits)
    assert detector.consecutive_count == 0
    assert not detector.aborted

    # Start again — need full window to trigger
    for i in range(4):
        logits = make_peaked_logits(token_id=42, peak_logit=20.0)
        detector([42] * (6 + i), logits)
    assert not detector.aborted
    assert detector.consecutive_count == 4


def test_repetition_aborted_stays_sticky():
    detector = RepetitionDetector(EOS_TOKEN_ID, window=5, prob_threshold=0.99)
    detector.aborted = True

    logits = make_uniform_logits()
    result = detector([42], logits)

    assert detector.aborted
    assert result[EOS_TOKEN_ID] == 0.0
    assert result[1].item() == float("-inf")
