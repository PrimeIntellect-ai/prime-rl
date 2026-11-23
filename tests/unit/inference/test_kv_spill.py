from concurrent.futures import Future
from types import SimpleNamespace

from prime_rl.inference import kv_spill


def _worker(ratio=1):
    return SimpleNamespace(
        engine_id="e",
        tp_rank=0,
        kv_caches_base_addr={"e": {0: [1000, 50_000]}},
        block_stride_per_layer=[100, 10],
        block_len_per_layer=[90, 8],
        _group_region_indices=[[0], [1]],
        _physical_blocks_per_logical_kv_block=ratio,
        _reqs_to_send={},
    )


def _done(result) -> Future:
    future: Future = Future()
    future.set_result(result)
    return future


class _Pool:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn.__name__, args))
        return Future()


def _state():
    state = kv_spill._State(store=None, replicate_config=None, lead_s=10.0, wait_s=0.0)
    state.pool.shutdown()
    state.pool = _Pool()
    return state


def test_block_buffers_cover_every_kernel_block_of_the_group_regions():
    assert kv_spill.block_buffers(_worker(), 0, 3) == ([1300], [90])
    assert kv_spill.block_buffers(_worker(ratio=2), 1, 3) == ([50_060, 50_070], [8, 8])


def _scheduler(sliding=(0, 4)):
    groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size)) for size in (256, 64)]
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(hash_block_size=64, transfer_groups=groups), blocks_per_sw=list(sliding)
    )


def test_full_blocks_of_a_whole_group_are_keyed_by_content_and_the_rest_by_request():
    """600 tokens: two full 256-token blocks and a partial one in the full-attention group; the windowed group
    exports only its last blocks, so their positions say nothing about their content."""
    hashes = [bytes([i]) for i in range(9)]  # one per 64 tokens
    keys = kv_spill.handle_keys(_scheduler(), "turn-1", hashes, 600, [3, 4])
    assert keys[0] == ["spill-kv@g0@03", "spill-kv@g0@07", "spill-req@turn-1@g0@2"]
    assert keys[1] == [f"spill-req@turn-1@g1@{i}" for i in range(4)]


def test_group_whose_list_does_not_start_at_the_first_token_gets_no_content_keys():
    keys = kv_spill.handle_keys(_scheduler(), "turn-1", [bytes([i]) for i in range(9)], 600, [2, 4])
    assert keys[0] == ["spill-req@turn-1@g0@0", "spill-req@turn-1@g0@1"]


def test_spill_writes_every_exported_block_under_its_key():
    held = ([[4, 5, 6], [9]], (["a", "b", "c"], ["d"]))
    keys, addrs, sizes = kv_spill.spill_plan(_worker(), held)
    assert keys == ["a", "b", "c", "d"]
    assert addrs == [[1400], [1500], [1600], [50_090]] and sizes == [[90], [90], [90], [8]]


def test_fetch_pairs_local_blocks_with_the_end_of_decodes_list():
    """Prefill holds the first block already: its two missing blocks are decode's last two."""
    meta = SimpleNamespace(local_block_ids=[[20, 21], [30]], remote=SimpleNamespace(block_ids=[[4, 5, 6], [9]]))
    keys, addrs, _ = kv_spill.fetch_plan(_worker(), meta, (["a", "b", "c"], ["d"]))
    assert keys == ["b", "c", "d"] and addrs == [[3000], [3100], [50_300]]


def test_fetch_is_refused_when_prefill_needs_more_blocks_than_decode_exported():
    meta = SimpleNamespace(local_block_ids=[[20, 21]], remote=SimpleNamespace(block_ids=[[4]]))
    assert kv_spill.fetch_plan(_worker(), meta, (["a"],)) is None


class _Store:
    def __init__(self, existing):
        self.existing, self.put = set(existing), []

    def batch_is_exist(self, keys):
        return [1 if key in self.existing else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        self.put.append((keys, addrs, sizes))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        return [sum(size) for size in sizes]

    def batch_remove(self, keys, force):
        self.existing.difference_update(keys)
        return [0] * len(keys)


def test_put_writes_only_the_blocks_the_store_does_not_hold():
    state = _state()
    state.store = _Store({"a", "b"})
    assert kv_spill._put(state, "turn-2", (["a", "b", "c"], [[1], [2], [3]], [[9], [9], [9]]))
    assert state.store.put == [(["c"], [[3]], [[9]])]


def test_lease_about_to_end_is_spilled_and_held_until_the_put_is_done():
    worker, state = _worker(), _state()
    worker._reqs_to_send = {"near": 100.0, "far": 500.0}
    state.held = {"near": ([[4], [9]], (["a"], ["b"])), "far": ([[5], [10]], (["c"], ["d"]))}

    kv_spill.start_due_spills(worker, state, now=89.0)
    assert not state.pool.submitted  # 11 s before the deadline: not due yet
    kv_spill.start_due_spills(worker, state, now=90.0)
    assert [name for name, _ in state.pool.submitted] == ["_put"]
    assert worker._reqs_to_send == {"near": float("inf"), "far": 500.0}

    state.putting["near"] = (_done(True), 100.0)
    kv_spill.start_due_spills(worker, state, now=95.0)
    # The lease is back on its deadline, and the turn is not spilled twice.
    assert worker._reqs_to_send["near"] == 100.0 and not state.putting
    assert [name for name, _ in state.pool.submitted] == ["_put"]


def test_turn_read_back_during_its_put_has_its_objects_removed():
    worker, state = _worker(), _state()
    state.held = {"turn-1": ([[4], [9]], (["spill-kv@g0@00"], ["spill-req@turn-1@g1@0"]))}
    state.putting["turn-1"] = (_done(True), 100.0)
    kv_spill.start_due_spills(worker, state, now=95.0)
    ((name, args),) = state.pool.submitted
    # Only objects keyed by this request can be stale; content-keyed blocks are filtered out by the removal.
    assert name == "_remove" and args[1] == ["spill-kv@g0@00", "spill-req@turn-1@g1@0"]


def test_fetch_deletes_the_objects_keyed_by_the_request_and_keeps_shared_blocks():
    state = _state()
    state.store = _Store({"spill-kv@g0@00", "spill-req@turn-1@g0@1", "spill-req@turn-1@g1@0"})
    plan = (["spill-kv@g0@00", "spill-req@turn-1@g0@1", "spill-req@turn-1@g1@0"], [[1], [2], [3]], [[9], [9], [9]])
    assert kv_spill._fetch(state, "turn-2", plan)
    assert state.store.existing == {"spill-kv@g0@00"}


def test_fetch_whose_objects_are_not_in_the_store_fails_so_the_turn_is_recomputed():
    state = _state()
    state.store = _Store({"spill-kv@g0@00"})
    plan = (["spill-kv@g0@00", "spill-req@turn-1@g0@1"], [[1], [2]], [[9], [9]])
    assert not kv_spill._fetch(state, "turn-2", plan)
    assert state.store.existing == {"spill-kv@g0@00"}


def _connector(extra):
    config = SimpleNamespace(get_from_extra_config=lambda key, default: extra.get(key, default))
    return SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=config))


def test_spill_is_off_unless_the_nixl_connector_config_asks_for_it():
    assert kv_spill.spill_settings(_connector({"bidirectional_kv_xfer": True})) is None
    settings = kv_spill.spill_settings(_connector({"prime_rl_spill_lead_s": 10, "prime_rl_spill_fetch_wait_s": 3}))
    assert settings == (10.0, 3.0)
