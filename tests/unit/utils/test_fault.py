import pytest

from prime_rl.utils.fault import parse


def test_parse():
    assert parse("") == {}
    assert parse("trainer_crash@5, env_error@reverse-text-b") == {"trainer_crash": "5", "env_error": "reverse-text-b"}
    for bad in ("trainer_crash", "trainer_crash@", "typo_crash@5"):
        with pytest.raises(ValueError):
            parse(bad)
