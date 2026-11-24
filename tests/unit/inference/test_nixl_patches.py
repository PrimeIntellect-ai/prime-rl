from types import SimpleNamespace

import pytest
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


def _lease_worker(monkeypatch, leases: dict[str, float]):
    """A NIXL worker's lease state on a fake clock, reaped by vLLM's own reaper."""
    import time

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker

    clock = [1000.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    worker = SimpleNamespace(
        _reqs_to_send=dict(leases),
        _reqs_to_process=set(leases),
        consumer_notification_counts_by_req={},
        expected_consumer_notifications_by_req={},
        xfer_stats=SimpleNamespace(record_kv_expired_req=lambda: None),
    )

    def reap(at: float) -> set[str]:
        clock[0] = at
        done: set[str] = set()
        NixlBaseConnectorWorker._reap_expired_send_leases(worker, done)
        return done

    return worker, clock, reap


def test_send_lease_expires_during_slow_steps(monkeypatch):
    """Without a pause, a 30 s lease runs down on wall-clock time however long the steps are."""
    from prime_rl.inference.patches import monkey_patch_freeze_nixl_leases_during_keep_pause

    monkey_patch_freeze_nixl_leases_during_keep_pause()
    worker, _, reap = _lease_worker(monkeypatch, {"held": 1030.0})
    for step in range(1, 5):
        assert reap(1000.0 + 6 * step) == set()
    assert reap(1030.0) == {"held"}


def test_send_lease_keeps_its_remaining_time_across_a_keep_pause(monkeypatch):
    """A lease with 4 s left still has 4 s left after a two-minute pause, then expires. A heartbeat's later deadline
    is kept, an already expired lease is not revived, and a lease granted during the pause is left alone."""
    from prime_rl.inference.patches import freeze_send_leases, thaw_send_leases

    worker, clock, reap = _lease_worker(monkeypatch, {"held": 1004.0, "renewed": 1002.0, "expired": 999.0})
    freeze_send_leases(worker)
    clock[0] = 1120.0
    worker._reqs_to_send["renewed"] = 1150.0  # a heartbeat applied after the pause
    worker._reqs_to_send["new"] = 1125.0
    worker._reqs_to_process.add("new")
    thaw_send_leases(worker)
    assert worker._reqs_to_send == {"held": 1124.0, "renewed": 1150.0, "expired": 999.0, "new": 1125.0}

    assert reap(1120.0) == {"expired"}
    assert reap(1123.9) == set()
    assert reap(1124.0) == {"held"}
    assert reap(1125.0) == {"new"}


@pytest.mark.parametrize("engine_cls", ["EngineCore", "EngineCoreProc"])
def test_keep_pause_freezes_the_nixl_leases_of_the_engines_workers(monkeypatch, engine_cls):
    """The patched engine pause and resume reach the NIXL worker inside a MultiConnector through collective_rpc, also
    through EngineCoreProc's own pause and resume (the engine class that serves); an ``abort`` pause leaves leases on
    wall-clock time."""
    import time

    import vllm.distributed.kv_transfer as kv_transfer
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    from vllm.v1.engine import core

    from prime_rl.inference.patches import monkey_patch_freeze_nixl_leases_during_keep_pause

    monkey_patch_freeze_nixl_leases_during_keep_pause()
    clock = [1000.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    nixl_worker = object.__new__(NixlBaseConnectorWorker)
    nixl_worker._reqs_to_send = {"held": 1004.0}
    # vLLM's own MultiConnector, so the test follows how it exposes its children.
    group = object.__new__(MultiConnector)
    group._connectors = [SimpleNamespace(), SimpleNamespace(connector_worker=nixl_worker)]
    monkeypatch.setattr(kv_transfer, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: group)

    engine = object.__new__(getattr(core, engine_cls))
    engine.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace())
    engine.scheduler = SimpleNamespace(set_pause_state=lambda state: None, finish_requests=lambda ids, status: [])
    engine._finish_pause = lambda clear_cache, reset_connector: None
    engine._set_kv_cache_reclaimable = lambda reclaimable: None
    engine._checkpoint_state = SimpleNamespace(is_restored=True)
    engine.collective_rpc = lambda method: [method(None)]
    engine._pause_complete = lambda: True
    engine._send_abort_outputs = lambda requests: None

    engine.pause_scheduler(mode="keep", clear_cache=False)
    clock[0] = 1120.0
    engine.resume_scheduler()
    assert nixl_worker._reqs_to_send == {"held": 1124.0}

    engine.pause_scheduler(mode="abort", clear_cache=False)
    clock[0] = 1300.0
    engine.resume_scheduler()
    assert nixl_worker._reqs_to_send == {"held": 1124.0}


def test_request_whose_kv_load_failed_keeps_no_reservation():
    """A load that failed outright freed its blocks; reserving the request's whole sequence afterwards would, with
    enough such requests, exceed the pool and stop every later load from being admitted."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from prime_rl.inference.patches import monkey_patch_drop_reservation_of_failed_kv_load

    monkey_patch_drop_reservation_of_failed_kv_load()
    scheduler = object.__new__(Scheduler)
    scheduler.kv_cache_config = SimpleNamespace(hisparse_host_num_blocks=None)
    scheduler._parked_reservations, scheduler._parked_reserved_blocks = {}, 0
    scheduler._request_remaining_blocks = lambda request: 400

    class _Request:  # hashable, unlike SimpleNamespace: the scheduler keys its reservations by request
        num_computed_tokens = 98_304

    loading = _Request()
    scheduler._reserve_parked(loading)
    assert scheduler._parked_reserved_blocks == 400

    loading.num_computed_tokens = 0  # the load failed and its blocks were freed
    scheduler._reserve_parked(loading)
    assert scheduler._parked_reserved_blocks == 0 and not scheduler._parked_reservations


def test_store_lookup_is_skipped_once_another_connector_matched():
    """With NIXL ahead of the Mooncake store in a MultiConnector, a request NIXL supplies does not ask the store, even
    while the store's lookup would still be pending; a request NIXL cannot supply is looked up as before."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

    from prime_rl.inference.patches import monkey_patch_skip_store_lookup_after_match

    monkey_patch_skip_store_lookup_after_match()
    asked: list[str] = []

    def connector(name, answer, cls=None):
        stub = object.__new__(cls) if cls else SimpleNamespace()
        stub.get_num_new_matched_tokens = lambda request, computed: asked.append(name) or answer[0]
        return stub

    nixl_answer, store_answer = [(4096, True)], [(None, False)]
    multi = object.__new__(MultiConnector)
    multi._requests_to_connector = {}
    multi._connectors = [
        connector("nixl", nixl_answer),
        connector("store", store_answer, MooncakeStoreConnector),
        connector("other", [(0, False)]),
    ]
    request = SimpleNamespace(request_id="turn-2")

    assert multi.get_num_new_matched_tokens(request, 0) == (4096, True)
    assert asked == ["nixl", "other"] and multi._requests_to_connector == {"turn-2": 0}

    # No decode handle: the store is asked, and a pending lookup defers the request.
    asked.clear()
    nixl_answer[0] = (0, False)
    assert multi.get_num_new_matched_tokens(request, 0) == (None, False)
    assert asked == ["nixl", "store"]
    asked.clear()
    store_answer[0] = (2048, True)
    assert multi.get_num_new_matched_tokens(request, 0) == (2048, True)
    assert asked == ["nixl", "store", "other"] and multi._requests_to_connector == {"turn-2": 1}


def test_store_lookup_prefetch_is_skipped_for_requests_with_a_decode_handle():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler

    from prime_rl.inference.patches import monkey_patch_skip_store_lookup_after_match

    monkey_patch_skip_store_lookup_after_match()
    prefetched: list[str] = []
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.enable_lookup = scheduler.lookup_async = True
    scheduler.enable_partial_hash_hits = False
    scheduler._block_size = scheduler._hash_block_size = 256
    scheduler.client = SimpleNamespace(prefetch=lambda request_id, num_tokens, hashes: prefetched.append(request_id))

    def request(request_id, params):
        return SimpleNamespace(
            request_id=request_id,
            kv_transfer_params=params,
            skip_reading_external_kv=False,
            num_tokens=4096,
            block_hashes=[b"h"] * 16,
        )

    scheduler.start_lookup_prefetch(request("turn-1", {"do_remote_decode": True}))
    scheduler.start_lookup_prefetch(request("turn-2", {"do_remote_decode": True, "remote_block_ids": [[1, 2]]}))
    scheduler.start_lookup_prefetch(request("plain", None))
    assert prefetched == ["turn-1", "plain"]


def test_failed_kv_load_gives_its_blocks_back_before_the_scheduler_reaches_it(monkeypatch):
    """The waiting loop stops at the first request it cannot allocate for, so a failed load behind it would keep
    its blocks forever."""
    from collections import Counter, deque

    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.metrics.stats import KV_FETCH_COMPLETED_WAITING
    from vllm.v1.request import RequestStatus

    from prime_rl.inference.patches import (
        monkey_patch_drop_reservation_of_failed_kv_load,
        monkey_patch_release_failed_kv_loads,
    )

    # Restored after the test: the patches are applied to stock vLLM here.
    monkeypatch.setattr(Scheduler, "_update_from_kv_xfer_finished", Scheduler._update_from_kv_xfer_finished)
    monkeypatch.setattr(Scheduler, "_reserve_parked", Scheduler._reserve_parked)
    monkey_patch_drop_reservation_of_failed_kv_load()
    monkey_patch_release_failed_kv_loads()
    # Nothing underneath tolerates freed requests here, and the wrapper must not claim otherwise.
    assert Scheduler._update_from_kv_xfer_finished._prime_rl_tolerates_freed is False

    class _Request:
        def __init__(self, request_id, num_computed_tokens=0):
            self.request_id, self.num_computed_tokens, self.num_tokens = request_id, num_computed_tokens, 100_000
            self.status, self.num_preemptions = RequestStatus.WAITING_FOR_REMOTE_KVS, 0

    loading, failed_late, loaded, failed_early = (
        _Request("loading"),
        _Request("failed-late"),
        _Request("loaded", 98_304),
        _Request("failed-early"),
    )
    requests = [loading, failed_early, loaded, failed_late]
    freed = []
    scheduler = object.__new__(Scheduler)
    scheduler.connector = SimpleNamespace(update_connector_output=lambda output: None)
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.kv_holding_waiting = FCFSRequestQueue(requests)
    scheduler.waiting = FCFSRequestQueue(deque())
    scheduler.deferred_waiting = set(requests)
    scheduler.finished_recving_kv_req_ids = set()
    # `_handle_failed_recving` has run: both failures are recorded and have no computed tokens.
    scheduler.failed_recving_kv_req_ids = {"failed-early", "failed-late"}
    scheduler._kv_fetch_stages, scheduler._kv_fetch_counts = {}, Counter()
    scheduler.kv_cache_manager = SimpleNamespace(free=lambda request: freed.append(request.request_id))
    scheduler._mark_prefix_replay = lambda request, num_hit_tokens, window_loaded: 0
    scheduler._parked_reservations = {failed_early: 400, failed_late: 380, loaded: 30}
    scheduler._parked_reserved_blocks = 810

    scheduler._update_from_kv_xfer_finished(
        SimpleNamespace(finished_recving=["failed-late", "loaded", "failed-early"], finished_sending=None)
    )

    # Only the failed loads give their blocks and reservations back.
    assert sorted(freed) == ["failed-early", "failed-late"]
    assert scheduler._parked_reservations == {loaded: 30} and scheduler._parked_reserved_blocks == 30
    assert failed_early.status == failed_late.status == RequestStatus.WAITING
    # Block holders keep their order; the failed requests wait at the front of the ordinary queue, in theirs.
    assert list(scheduler.kv_holding_waiting) == [loading, loaded]
    assert list(scheduler.waiting) == [failed_early, failed_late]
    assert scheduler.deferred_waiting == {loading, loaded}
    # A finished load stays pending for the waiting loop; a failed one no longer counts as a load in flight.
    assert scheduler.finished_recving_kv_req_ids == {"loaded"} and not scheduler.failed_recving_kv_req_ids
    assert scheduler._kv_fetch_stages == {loaded: KV_FETCH_COMPLETED_WAITING}
