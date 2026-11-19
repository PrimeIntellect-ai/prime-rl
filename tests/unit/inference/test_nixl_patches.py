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


def _spec(cls, block_size):
    spec = object.__new__(cls)
    object.__setattr__(spec, "block_size", block_size)
    return SimpleNamespace(kv_cache_spec=spec)


def test_trim_block_ids_to_tokens():
    from vllm.v1.kv_cache_interface import MambaSpec

    from prime_rl.inference.patches import trim_block_ids_to_tokens

    groups = [_spec(FullAttentionSpec, 256), _spec(SlidingWindowSpec, 256), _spec(MambaSpec, 256)]
    # 513 positions span 3 blocks; the drafted positions past them added a 4th.
    block_ids = ([1, 2, 3, 4], [0, 0, 7, 8], [9])
    assert trim_block_ids_to_tokens(block_ids, groups, 513) == ([1, 2, 3], [0, 0, 7], [9])
    # Same block count: nothing to drop.
    assert trim_block_ids_to_tokens(([1, 2],), groups[:1], 300) == ([1, 2],)


class _FakeFinish:
    def __init__(self, status, num_tokens, num_computed_tokens, params):
        self.status = status
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed_tokens
        self.kv_transfer_params = params


def test_decode_handle_covers_only_valid_tokens_under_speculative_decoding(monkeypatch):
    from vllm.v1.request import RequestStatus

    from prime_rl.inference.patches import monkey_patch_nixl_bidirectional_spec_decode_coverage

    seen = []

    def original(self, request, block_ids):
        seen.append((request.num_computed_tokens, block_ids))
        return True, {"remote_num_tokens": request.num_computed_tokens}

    monkeypatch.setattr(NixlPullConnectorScheduler, "request_finished", original)
    monkey_patch_nixl_bidirectional_spec_decode_coverage()
    patched = NixlPullConnectorScheduler.request_finished
    monkey_patch_nixl_bidirectional_spec_decode_coverage()
    assert NixlPullConnectorScheduler.request_finished is patched

    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler.is_bidirectional_kv_xfer_enabled = True
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[_spec(FullAttentionSpec, 16)])
    decode = {"do_remote_decode": False, "do_remote_prefill": False}

    # 23 prompt + 16 output tokens; the last step also computed 3 drafted positions (42), crossing into a 3rd block.
    request = _FakeFinish(RequestStatus.FINISHED_LENGTH_CAPPED, 39, 42, dict(decode))
    _, params = scheduler.request_finished(request, ([5, 6, 7],))
    assert params["remote_num_tokens"] == 38 and seen[-1] == (38, ([5, 6, 7],))
    request = _FakeFinish(RequestStatus.FINISHED_LENGTH_CAPPED, 32, 35, dict(decode))
    _, params = scheduler.request_finished(request, ([5, 6, 7],))
    assert params["remote_num_tokens"] == 31 and seen[-1] == (31, ([5, 6],))
    assert request.num_computed_tokens == 35  # the scheduler's own count is left as it was

    # Unchanged: no speculative decoding, the prefill side, and requests that did not finish normally.
    for request in (
        _FakeFinish(RequestStatus.FINISHED_STOPPED, 39, 38, dict(decode)),
        _FakeFinish(RequestStatus.FINISHED_LENGTH_CAPPED, 39, 42, {"do_remote_decode": True}),
        _FakeFinish(RequestStatus.FINISHED_ABORTED, 39, 42, dict(decode)),
    ):
        _, params = scheduler.request_finished(request, ([5, 6, 7],))
        assert params["remote_num_tokens"] == request.num_computed_tokens and seen[-1][1] == ([5, 6, 7],)


def test_send_leases_do_not_expire_while_the_engine_is_stopped(monkeypatch):
    """A lease keeps the time it had left when the engine stopped stepping (a pause or a weight update); leases
    still expire on their own schedule while the engine steps."""
    import time

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker

    from prime_rl.inference.patches import monkey_patch_nixl_lease_clock_follows_engine

    monkeypatch.setattr(
        NixlBaseConnectorWorker, "_reap_expired_send_leases", NixlBaseConnectorWorker._reap_expired_send_leases
    )
    monkey_patch_nixl_lease_clock_follows_engine()
    clock = [1000.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    worker = SimpleNamespace(
        _reqs_to_send={"held": 1020.0, "short": 1001.5},
        _reqs_to_process={"held", "short"},
        consumer_notification_counts_by_req={},
        expected_consumer_notifications_by_req={},
        xfer_stats=SimpleNamespace(record_kv_expired_req=lambda: None),
    )

    def reap(at: float) -> set[str]:
        clock[0] = at
        done: set[str] = set()
        NixlBaseConnectorWorker._reap_expired_send_leases(worker, done)
        return done

    assert reap(1000.0) == set()
    assert reap(1001.0) == set()
    assert reap(1002.0) == {"short"}, "a lease still expires while the engine steps"
    # Paused for 5 minutes: "held" had 18 s left and keeps them.
    assert reap(1302.0) == set()
    assert worker._reqs_to_send["held"] == 1320.0
    # A lease granted after the pause is stamped fresh and not shifted.
    worker._reqs_to_send["new"] = 1310.0
    worker._reqs_to_process.add("new")
    assert reap(1303.0) == set() and worker._reqs_to_send["new"] == 1310.0
    # The engine steps again (checks less than 5 s apart): both leases run down and expire.
    assert reap(1306.0) == set() and reap(1309.0) == set()
    assert reap(1311.0) == {"new"}
    assert reap(1315.0) == set() and reap(1319.0) == set()
    assert reap(1320.0) == {"held"}
