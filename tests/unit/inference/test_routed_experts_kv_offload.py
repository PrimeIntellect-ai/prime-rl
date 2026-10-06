from types import SimpleNamespace

import pytest

from prime_rl.inference.patches import _first_loaded_block, _routed_experts_load_limit, _track_externally_loaded_blocks


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
        (0, 64, {"do_remote_prefill": False, "remote_engine_id": "p0"}, None),  # P/D decode
    ],
)
def test_routed_experts_load_limit(prompt_start, num_computed_tokens, kv_transfer_params, limit):
    request = _request(prompt_start, kv_transfer_params)
    assert _routed_experts_load_limit(request, num_computed_tokens, block_size=64) == limit


def test_externally_loaded_blocks_cut_lower_start_hits():
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

    bs = 16
    init_none_hash(sha256)
    spec = FullAttentionSpec(block_size=bs, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["l0"], spec)])
    manager = KVCacheManager(config, max_model_len=1024, scheduler_block_size=bs, hash_block_size=bs)
    loaded = _track_externally_loaded_blocks(manager, gid=0, block_size=bs)
    hasher = get_request_block_hasher(bs, sha256)

    def request(rid, tokens, start):
        params = SamplingParams(max_tokens=1, routed_experts_prompt_start=start)
        return Request(rid, tokens, params, None, block_hasher=hasher)

    # A multi-turn request loads its 64-token prefix externally (the clamp allows it: start=64)...
    prefix = list(range(64))
    a = request("a", prefix + list(range(500, 520)), start=64)
    manager.allocate_slots(a, 20, num_external_computed_tokens=64)
    a.num_computed_tokens = a.num_tokens
    manager.free(a)
    # ...so a first-turn request (start=0) hitting that prefix locally must cut the hit at block 0,
    # while a later turn (start=64) keeps it.
    blocks, num_local, _ = manager.get_computed_blocks(request("b", prefix + [7, 7], start=0))
    assert num_local == 64
    assert _first_loaded_block(blocks.get_block_ids()[0], loaded, 0, bs) == 0
    assert _first_loaded_block(blocks.get_block_ids()[0], loaded, 64, bs) is None
    # Reallocating the blocks to a new owner clears them.
    assert manager.allocate_slots(request("c", list(range(1000, 1176)), start=0), 176) is not None
    assert not loaded
