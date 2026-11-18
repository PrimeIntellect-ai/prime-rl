from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import NixlBaseConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowMLASpec, SlidingWindowSpec

from prime_rl.inference.patches import (
    monkey_patch_nixl_bidirectional_single_pull,
    monkey_patch_nixl_sliding_window_backoff,
)


def _turn2_request(**params):
    remote = {
        "do_remote_decode": True,
        "remote_block_ids": [[1, 2, 3]],
        "remote_engine_id": "decode-0",
        "remote_request_id": "turn-1",
        "remote_host": "decode-host",
        "remote_port": 5600,
        "remote_num_tokens": 768,
    }
    return SimpleNamespace(request_id="turn-2", kv_transfer_params={**remote, **params}, num_prompt_tokens=1024)


def test_bidirectional_prefill_pulls_decode_blocks_once(monkeypatch):
    monkeypatch.setattr(
        NixlPullConnectorScheduler, "get_num_new_matched_tokens", NixlPullConnectorScheduler.get_num_new_matched_tokens
    )
    monkey_patch_nixl_bidirectional_single_pull()
    patched = NixlPullConnectorScheduler.get_num_new_matched_tokens
    monkey_patch_nixl_bidirectional_single_pull()
    assert NixlPullConnectorScheduler.get_num_new_matched_tokens is patched

    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler.kv_recompute_threshold = 0
    # First lookup: decode's blocks are offered for an async D->P read.
    assert scheduler.get_num_new_matched_tokens(_turn2_request(), 0) == (768, True)
    # Back through the lookup after the pull (failed read under `recompute`, or preempted and evicted): prefill locally.
    assert scheduler.get_num_new_matched_tokens(_turn2_request(_remote_blocks_processed=True), 0) == (0, False)


def _backoff_scheduler(*specs):
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler._has_mamba = False
    scheduler._bounded_replay = False
    scheduler.vllm_config = SimpleNamespace(num_prefill_lookahead_tokens=1)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in specs])
    return scheduler


def test_sliding_window_models_leave_last_prompt_token_to_decode(monkeypatch):
    monkeypatch.setattr(NixlBaseConnectorScheduler, "_prefill_backoff", NixlBaseConnectorScheduler._prefill_backoff)
    full, window = object.__new__(FullAttentionSpec), object.__new__(SlidingWindowSpec)
    assert _backoff_scheduler(full, window)._prefill_backoff() == 0

    monkey_patch_nixl_sliding_window_backoff()
    patched = NixlBaseConnectorScheduler._prefill_backoff
    monkey_patch_nixl_sliding_window_backoff()
    assert NixlBaseConnectorScheduler._prefill_backoff is patched

    assert _backoff_scheduler(full, window)._prefill_backoff() == 1
    # DeepSeek V4's compressor caches. The scheduler sees per-layer specs: vLLM unwraps UniformTypeKVCacheSpecs
    # (generate_scheduler_kv_cache_config), and SlidingWindowMLASpec subclasses SlidingWindowSpec.
    assert _backoff_scheduler(full, object.__new__(SlidingWindowMLASpec))._prefill_backoff() == 1
    assert _backoff_scheduler(full, window)._get_remote_prefill_token_count(1024) == 1023
    assert _backoff_scheduler(full)._prefill_backoff() == 0
