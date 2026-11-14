from types import SimpleNamespace

import numpy as np

from prime_rl.inference.patches import (
    _emit_floors,
    monkey_patch_routed_experts_with_pd_connectors,
    skip_remote_prefill_partial_block,
)


def _kv_cache_manager(block_size=4):
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["l0"], spec)])
    return KVCacheManager(config, max_model_len=1024, scheduler_block_size=block_size, hash_block_size=block_size)


def test_remote_prefill_rows_start_at_first_forward():
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector

    monkey_patch_routed_experts_with_pd_connectors()
    connector = AuxOutputSchedulerConnector(_kv_cache_manager())
    request = SimpleNamespace(
        request_id="r",
        block_hashes=[b"a" * 32],
        num_tokens=6,
        num_prompt_tokens=6,
        num_output_tokens=0,
        num_computed_tokens=5,  # tokens [0, 5) were loaded from the prefill instance
        sampling_params=SimpleNamespace(routed_experts_prompt_start=0),
        kv_transfer_params={"do_remote_prefill": False, "remote_engine_id": "p0"},
        is_finished=lambda: False,
    )

    def emit_start():
        step = SimpleNamespace(num_scheduled_tokens={"r": request.num_tokens - request.num_computed_tokens})
        return connector.build_connector_meta(step, {"r": request}).requests["r"]

    assert emit_start() == 5
    # A preempted request recomputes from scratch but still emits from its first forward.
    connector.request_finished(request)
    request.num_computed_tokens = 0
    assert emit_start() == 5
    request.is_finished = lambda: True
    connector.request_finished(request)
    assert not _emit_floors(connector)


def test_pd_decode_requests_skip_retained_row_caps():
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector

    monkey_patch_routed_experts_with_pd_connectors()
    connector = AuxOutputSchedulerConnector(_kv_cache_manager(), kv_connector=object())

    def request(kv_transfer_params):
        return SimpleNamespace(
            request_id="r",
            block_hashes=[],
            num_tokens=10,
            num_output_tokens=0,
            kv_transfer_params=kv_transfer_params,
            # Rows from token 0 on: no load is free of the retained-rows check.
            sampling_params=SimpleNamespace(routed_experts_prompt_start=0),
        )

    # A decode instance has no rows for a remotely prefilled prompt: the NIXL load must not be zeroed.
    decode = request({"do_remote_prefill": True, "remote_engine_id": "p0"})
    assert connector.cap_external_hit(decode, 0, 9) == 9
    assert connector.max_cache_hit(decode) is None
    # Other requests keep upstream's caps (no retained rows -> no hit).
    local = request(None)
    assert connector.cap_external_hit(local, 0, 8) == 0
    assert connector.max_cache_hit(local) == 0


def test_remote_prefill_partial_first_block_is_not_buffered():
    from vllm.distributed.aux_output_connector.routed_experts import RoutedExpertsBuffer
    from vllm.distributed.aux_output_connector.worker import _WorkerRequestState

    rows = np.arange(12 * 3 * 2, dtype=np.uint8).reshape(12, 3, 2)
    buffer = RoutedExpertsBuffer(np.dtype("uint8"), (3, 2), 4, 1, 8, 2)
    state = _WorkerRequestState(emit_cursor=6)
    worker = SimpleNamespace(_buffer=buffer, _requests={"r": state})
    skip_remote_prefill_partial_block(worker)

    # The first forward starts mid-block at token 6; without the skip the buffer asserts.
    assert buffer.capture("r", 6, rows[6:7]) == []
    state.capture_cursor = 7
    assert buffer.capture("r", 7, rows[7:9]) == []
    state.capture_cursor = 9
    [(start, block)] = buffer.capture("r", 9, rows[9:12])
    assert start == 8
    np.testing.assert_array_equal(block, rows[8:12])
