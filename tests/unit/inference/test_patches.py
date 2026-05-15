from types import SimpleNamespace

from prime_rl.inference.patches import (
    _sanitize_request_spec_token_ids,
    _sanitize_scheduler_spec_token_ids,
    _trim_invalid_spec_token_suffix,
)


def test_trim_invalid_spec_token_suffix_keeps_valid_tokens():
    spec_token_ids = [123, 456]

    assert _trim_invalid_spec_token_suffix(spec_token_ids) is spec_token_ids


def test_trim_invalid_spec_token_suffix_drops_invalid_suffix():
    assert _trim_invalid_spec_token_suffix([-1]) == []
    assert _trim_invalid_spec_token_suffix([123, -1]) == [123]
    assert _trim_invalid_spec_token_suffix([123, -1, 456]) == [123]


def test_sanitize_request_spec_token_ids_updates_only_invalid_tokens():
    request = SimpleNamespace(spec_token_ids=[123, -1, 456], num_computed_tokens=7)

    assert _sanitize_request_spec_token_ids(request) is True
    assert request.spec_token_ids == [123]
    assert request.num_computed_tokens == 7

    assert _sanitize_request_spec_token_ids(request) is False
    assert request.spec_token_ids == [123]


def test_sanitize_scheduler_spec_token_ids_updates_all_requests():
    scheduler = SimpleNamespace(
        requests={
            "valid": SimpleNamespace(spec_token_ids=[11]),
            "all_invalid": SimpleNamespace(spec_token_ids=[-1]),
            "mixed": SimpleNamespace(spec_token_ids=[22, -1, 33]),
            "empty": SimpleNamespace(spec_token_ids=[]),
        }
    )

    assert _sanitize_scheduler_spec_token_ids(scheduler) == 2
    assert scheduler.requests["valid"].spec_token_ids == [11]
    assert scheduler.requests["all_invalid"].spec_token_ids == []
    assert scheduler.requests["mixed"].spec_token_ids == [22]
    assert scheduler.requests["empty"].spec_token_ids == []
