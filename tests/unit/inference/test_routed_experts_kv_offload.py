from types import SimpleNamespace

import pytest

from prime_rl.inference.patches import _first_loaded_block, _routed_experts_load_limit, _track_externally_loaded_blocks


def _request(prompt_start, kv_transfer_params=None, num_output_tokens=0):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(routed_experts_prompt_start=prompt_start),
        kv_transfer_params=kv_transfer_params,
        num_output_tokens=num_output_tokens,
    )


@pytest.mark.parametrize(
    ("prompt_start", "num_computed_tokens", "kv_transfer_params", "num_output_tokens", "limit"),
    [
        (0, 0, None, 0, 0),  # first turn: no external load
        (100, 0, None, 0, 64),  # multi-turn: load up to the block before prompt_start
        (100, 64, None, 0, 0),  # local prefix hit already covers it
        (200, 64, None, 0, 128),
        (200, 0, {"do_remote_decode": True, "remote_engine_id": None}, 0, 192),  # P/D prefill
        (0, 64, {"do_remote_prefill": False, "remote_engine_id": "p0"}, 0, None),  # P/D decode
        (0, 0, None, 5, None),  # resumed after preemption
    ],
)
def test_routed_experts_load_limit(prompt_start, num_computed_tokens, kv_transfer_params, num_output_tokens, limit):
    request = _request(prompt_start, kv_transfer_params, num_output_tokens)
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


def test_cut_patch_takes_upstream_max_cache_hit_length():
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

    from prime_rl.inference.patches import monkey_patch_routed_experts_cut_loaded_prefix_hits

    monkey_patch_routed_experts_cut_loaded_prefix_hits()
    bs = 16
    init_none_hash(sha256)
    spec = FullAttentionSpec(block_size=bs, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["l0"], spec)])
    manager = KVCacheManager(config, max_model_len=1024, scheduler_block_size=bs, hash_block_size=bs)
    hasher = get_request_block_hasher(bs, sha256)
    a = Request("a", list(range(70)), SamplingParams(max_tokens=1), None, block_hasher=hasher)
    manager.allocate_slots(a, 70)
    a.num_computed_tokens = a.num_tokens
    manager.free(a)

    # The scheduler passes the AuxOutput connector's cap positionally.
    scheduler = SimpleNamespace(connector=None, kv_cache_manager=manager, _prime_rl_loaded_blocks=None)
    b = Request("b", list(range(70)), SamplingParams(max_tokens=1), None, block_hasher=hasher)
    _, num_local, _, _ = Scheduler._get_local_prefix_cache_hit(scheduler, b, 32)
    assert num_local == 32
    _, num_local, _, _ = Scheduler._get_local_prefix_cache_hit(scheduler, b)
    assert num_local == 64


def test_offload_loads_below_prompt_start_need_no_retained_rows():
    import torch
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    from prime_rl.inference.patches import monkey_patch_clamp_kv_offload_loads_for_routed_experts

    monkey_patch_clamp_kv_offload_loads_for_routed_experts()
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["l0"], spec)])
    manager = KVCacheManager(config, max_model_len=1024, scheduler_block_size=16, hash_block_size=16)
    connector = AuxOutputSchedulerConnector(manager, kv_connector=object())
    request = SimpleNamespace(
        request_id="r",
        block_hashes=[],
        num_tokens=100,
        num_output_tokens=0,
        kv_transfer_params=None,
        sampling_params=SimpleNamespace(routed_experts_prompt_start=70),
    )
    # The next turn needs rows from token 70 on, so a load of the first 64 tokens needs none: admitted
    # although this instance retains no rows for those blocks.
    assert connector.cap_external_hit(request, 0, 64) == 64
    assert connector.cap_external_hit(request, 16, 48) == 48
    # Past the prompt-start block, upstream's retained-rows check still decides (no rows -> no hit).
    assert connector.cap_external_hit(request, 0, 80) == 0
