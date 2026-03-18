from prime_rl.utils.usage_signing import sign_usage, verify_usage

SECRET = "test-secret-key-12345"
RUN_ID = "abc123"


def test_sign_produces_hex_string():
    sig = sign_usage(SECRET, RUN_ID, 100, 200)
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA-256 hex digest


def test_verify_correct_signature():
    sig = sign_usage(SECRET, RUN_ID, 100, 200)
    assert verify_usage(SECRET, RUN_ID, 100, 200, sig) is True


def test_verify_rejects_wrong_tokens():
    sig = sign_usage(SECRET, RUN_ID, 100, 200)
    assert verify_usage(SECRET, RUN_ID, 101, 200, sig) is False
    assert verify_usage(SECRET, RUN_ID, 100, 201, sig) is False


def test_verify_rejects_wrong_run_id():
    sig = sign_usage(SECRET, RUN_ID, 100, 200)
    assert verify_usage(SECRET, "other_run", 100, 200, sig) is False


def test_verify_rejects_wrong_secret():
    sig = sign_usage(SECRET, RUN_ID, 100, 200)
    assert verify_usage("wrong-secret", RUN_ID, 100, 200, sig) is False


def test_verify_rejects_garbage_signature():
    assert verify_usage(SECRET, RUN_ID, 100, 200, "not-a-real-signature") is False


def test_sign_is_deterministic():
    sig1 = sign_usage(SECRET, RUN_ID, 500, 1000)
    sig2 = sign_usage(SECRET, RUN_ID, 500, 1000)
    assert sig1 == sig2


def test_different_inputs_produce_different_signatures():
    sig1 = sign_usage(SECRET, RUN_ID, 100, 200)
    sig2 = sign_usage(SECRET, RUN_ID, 200, 100)
    assert sig1 != sig2
