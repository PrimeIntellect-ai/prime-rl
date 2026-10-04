from types import SimpleNamespace

import pytest

from prime_rl.inference.patches import _routed_experts_load_limit


def _request(prompt_start, kv_transfer_params=None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(routed_experts_prompt_start=prompt_start),
        kv_transfer_params=kv_transfer_params,
    )


@pytest.mark.parametrize(
    ("prompt_start", "num_computed_tokens", "kv_transfer_params", "limit"),
    [
        (0, 0, None, 0),  # first turn: no external load
        (100, 0, None, 64),  # multi-turn: load up to the block before prompt_start
        (100, 64, None, 0),  # local prefix hit already covers it
        (200, 64, None, 128),
        (200, 0, {"do_remote_decode": True, "remote_engine_id": None}, 192),  # P/D prefill
        (0, 0, {"do_remote_prefill": True, "remote_engine_id": "p0"}, None),  # P/D decode
        (0, 0, {"do_remote_prefill": False, "remote_engine_id": "p0"}, None),  # P/D decode after pull
    ],
)
def test_routed_experts_load_limit(prompt_start, num_computed_tokens, kv_transfer_params, limit):
    request = _request(prompt_start, kv_transfer_params)
    assert _routed_experts_load_limit(request, num_computed_tokens, block_size=64) == limit
