import base64
import hashlib
import inspect
import pickle
import re
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.distributed
from vllm.config import CUDAGraphMode
from vllm.distributed.aux_output_connector import worker as aux_worker
from vllm.distributed.aux_output_connector.connector import AuxOutputConnectorMetadata, PackedBlockHashes
from vllm.distributed.aux_output_connector.routed_experts import RoutedExpertsBuffer
from vllm.distributed.aux_output_connector.store import BackgroundBlockObjectStore, BlockObject, BlockObjectStore
from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector, PendingAuxOutput
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import prime_rl.inference.vllm.forced_routing as fr
from prime_rl.inference.patches import monkey_patch_aux_output_store

B, L, K, E = 4, 3, 2, 300  # hash block size, captured layers (layer 0 dense), top-k, experts
NB = B * L * K * 2  # bytes of one stored uint16 block
MAX_TOKENS = 64
CPU = torch.device("cpu")
PIECEWISE = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 64, None)
DECODE_GRAPH = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)


@pytest.fixture(autouse=True)
def module_state(monkeypatch):
    """Fresh module globals per test; class patches below only touch per-test subclasses."""
    monkeypatch.setattr(fr, "_STATE", None)
    monkeypatch.setattr(fr, "_WEIGHT_VERSION", fr.UNKNOWN)
    monkeypatch.setattr(fr, "_PUBLISH_META", {})
    monkeypatch.setattr(fr, "_STORE_COUNTERS", dict.fromkeys(fr._STORE_COUNTERS, 0))
    monkeypatch.setattr(fr, "_CAPTURING_DECODE_ONLY_GRAPH", False)
    monkeypatch.delenv("PRIME_RL_FORCED_ROUTING", raising=False)


def _rows(start, n, base=0):
    """Rows that depend only on the absolute position, so two sources agree where they overlap."""
    return np.array(
        [[[(base + p + 5 * layer) % E, (p * 7 + layer) % E] for layer in range(L)] for p in range(start, start + n)],
        dtype=np.uint16,
    )


def _payload(start, rows, **extra):
    rows = np.asarray(rows)
    return {
        "start": start,
        "data": base64.b64encode(rows.tobytes()).decode(),
        "shape": list(rows.shape),
        "dtype": rows.dtype.name,
        **extra,
    }


def _natural(tokens, positions, pass_id):
    """Deterministic 'router output' of one forward: a function of token, position, layer and pass."""
    t = np.asarray(tokens, dtype=np.int64)[:, None, None]
    p = np.asarray(positions, dtype=np.int64)[:, None, None]
    layer = np.arange(L)[None, :, None]
    k = np.arange(K)[None, None, :]
    return ((t * 31 + p * 7 + layer * 13 + pass_id * 101 + k * 37) % E).astype(np.uint16)


def _block_hashes(tokens, salt):
    out, parent = [], salt.encode()
    for j in range(len(tokens) // B):
        parent = hashlib.sha256(parent + np.asarray(tokens[j * B : (j + 1) * B], np.int64).tobytes()).digest()[:8]
        out.append(parent)
    return out


class Sim:
    """vLLM 0.32's real routed-experts connector (store, writer thread, tail buffer, publish and teardown paths) on
    TP rank 0, driven step by step the way the V2 model runner drives it, with the forced-routing patches applied to
    per-test subclasses."""

    def __init__(self, stash_bytes=1 << 20, store_bytes=64 * NB, drift=False):
        store_cls = type("Store", (BlockObjectStore,), {})
        fr._patch_store(store_cls)
        conn_cls = type("Conn", (AuxOutputWorkerConnector,), {})
        fr._patch_connector(conn_cls)
        conn = conn_cls.__new__(conn_cls)
        conn._capturer = SimpleNamespace(device_buffer=torch.zeros(MAX_TOKENS, L, K), shape_per_token=(L, K))
        conn._store = BackgroundBlockObjectStore(
            store_cls(max_bytes=store_bytes, object_nbytes=NB), max_pending_batches=16
        )
        conn._buffer = RoutedExpertsBuffer(np.dtype(np.uint16), (L, K), B, 16, MAX_TOKENS, 2)
        conn._requests, conn._generation, conn._step_metadata = {}, 0, None
        conn._pending_outputs, conn._lock, conn._max_concurrent_batches = [], Lock(), 2
        self.conn = conn
        stash = fr.TailStash(B, stash_bytes, max_live=64) if stash_bytes else None
        self.state = fr.ForcedRoutingState(
            MAX_TOKENS, L, K, E, range(1, L), CPU, connector=conn, drift=drift, stash=stash
        )
        fr._STATE = self.state
        self.reqs: dict[str, dict] = {}
        self.finished: list[str] = []
        self.passes = 0
        self.captured: dict[str, dict[int, np.ndarray]] = {}  # req -> position -> captured row

    def close(self):
        self.conn._store.close()

    def add(self, req_id, tokens, salt="s", prompt_start=0, num_computed=0, prompt_len=None, payload=None):
        self.reqs[req_id] = dict(
            tokens=list(tokens),
            salt=salt,
            prompt_len=prompt_len or len(tokens),
            prompt_start=prompt_start,
            nc=num_computed,
            sent=0,
            new=True,
            payload=payload,
        )
        self.captured.setdefault(req_id, {})

    def finish(self, req_id):
        self.finished.append(req_id)

    def step(self, schedule, desc=PIECEWISE):
        """schedule: {req_id: number of tokens}. Returns the forced mask of the step's rows."""
        st, conn = self.state, self.conn
        finished, self.finished = self.finished, []
        hashes: dict[str, PackedBlockHashes] = {}
        for rid in finished:  # finish_requests, then the terminal event with the final hashes
            st.remove_request(rid)
            r = self.reqs.pop(rid)
            new = _block_hashes(r["tokens"], r["salt"])[r["sent"] :]
            if new:
                hashes[rid] = PackedBlockHashes(b"".join(new), 8)
        requests = {}
        for rid in schedule:  # add_requests
            r = self.reqs[rid]
            if r["new"]:
                r["new"] = False
                st.add_request(rid, r["prompt_len"], r["prompt_start"], r["tokens"], r["payload"])
            new = _block_hashes(r["tokens"], r["salt"])[r["sent"] :]
            if new:
                hashes[rid] = PackedBlockHashes(b"".join(new), 8)
                r["sent"] += len(new)
            outputs = len(r["tokens"]) - r["prompt_len"]
            requests[rid] = max(min(r["prompt_start"], r["prompt_len"]), 0 if outputs <= 0 else len(r["tokens"]) - 1)
        conn.begin_step(AuxOutputConnectorMetadata(0, requests, hashes, tuple(finished)))
        if not schedule:  # nothing to run: the runner returns before prepare_inputs
            return None
        req_ids = list(schedule)
        qsl = np.concatenate(([0], np.cumsum([schedule[r] for r in req_ids]))).astype(np.int32)
        nc = np.array([self.reqs[r]["nc"] for r in req_ids], dtype=np.int32)
        pl = np.array([len(self.reqs[r]["tokens"]) for r in req_ids], dtype=np.int32)
        st.fill(conn, req_ids, qsl, nc, pl, desc)
        # forward: the router's own pick, overridden where the step forces
        self.passes += 1
        n = int(qsl[-1])
        positions = np.concatenate([np.arange(nc[i], nc[i] + schedule[r]) for i, r in enumerate(req_ids)])
        tokens = np.concatenate([self.reqs[r]["tokens"][nc[i] : nc[i] + schedule[r]] for i, r in enumerate(req_ids)])
        mask = st.mask[:n].numpy().copy() if st.dirty else np.zeros(n, dtype=bool)
        rows = np.where(
            mask[:, None, None], st.ids[:n].numpy().astype(np.uint16), _natural(tokens, positions, self.passes)
        )
        for i, r in enumerate(req_ids):
            for p, row in zip(positions[qsl[i] : qsl[i + 1]], rows[qsl[i] : qsl[i + 1]]):
                self.captured[r][int(p)] = row
        # commit (prepare_output, then process_output on the output thread) and sample
        sampled = []
        for i, r in enumerate(req_ids):
            req = self.reqs[r]
            req["nc"] += schedule[r]
            if req["nc"] >= len(req["tokens"]):
                token = 1000 + len(req["tokens"])
                req["tokens"].append(token)
                sampled.append([token])
            else:
                sampled.append([])
        pending = PendingAuxOutput(conn, req_ids, nc, qsl, routed_experts_gpu=None)
        pending.routed_experts = rows
        pending.num_sampled = np.array([len(s) for s in sampled])
        pending.num_rejected = np.zeros(len(req_ids), dtype=np.int64)
        conn._pending_outputs.append(pending)
        for r in req_ids:
            conn._requests[r].pending_outputs += 1
        conn.process_output(pending)
        if st.stash is not None:
            st.stash.record(req_ids, sampled)
        return mask

    def decode(self, req_id, steps):
        for _ in range(steps):
            self.step({req_id: 1})


@pytest.fixture
def sim():
    s = Sim()
    yield s
    s.close()


def _first_turn(sim, rid="a", prompt=10, path=15, salt="s"):
    """A first turn: prompt of `prompt` tokens decoded to a path of `path` tokens (rows of [0, path - 1) captured)."""
    sim.add(rid, range(1, prompt + 1), salt=salt)
    sim.step({rid: prompt})
    sim.decode(rid, path - prompt - 1)
    assert len(sim.reqs[rid]["tokens"]) == path
    tokens = list(sim.reqs[rid]["tokens"])
    sim.finish(rid)
    return tokens


# ----------------------------------------------------------------------------- store access and provenance
def test_slot_table_on_vllm_store_on_mains_subclass_and_after_62392(monkeypatch):
    monkeypatch.setattr(BlockObjectStore, "_evict_to_fit", BlockObjectStore._evict_to_fit)
    monkeypatch.setattr(aux_worker, "BlockObjectStore", aux_worker.BlockObjectStore)
    monkey_patch_aux_output_store()  # main's 2x subclass, as the connector builds it
    for cls in (BlockObjectStore, aux_worker.BlockObjectStore):
        store = cls(max_bytes=8, object_nbytes=4)
        store.put([BlockObject("k", b"abcd")], retain_keys=["k"])
        assert fr.slot_of(fr.slot_table(store), "k") == 0 and fr.slot_of(fr.slot_table(store), "x") is None
        store.close()
    # vllm#62392 keeps slots in _slots and only unreferenced keys in _lru
    post = SimpleNamespace(_slots={"k": 3, "u": BlockObjectStore._UNALLOCATED_SLOT}, _lru={"k": None})
    assert fr.slot_of(fr.slot_table(post), "k") == 3 and fr.slot_of(fr.slot_table(post), "u") is None
    assert fr.resident_slots(post) == 1


def test_store_stamps_first_publication_and_counts_evictions(monkeypatch):
    store_cls = type("Store", (BlockObjectStore,), {})
    fr._patch_store(store_cls)
    store = store_cls(max_bytes=2 * NB, object_nbytes=NB)
    block = _rows(0, B).tobytes()
    monkeypatch.setattr(fr, "_WEIGHT_VERSION", 5)
    store.put([BlockObject("a", block)])
    monkeypatch.setattr(fr, "_WEIGHT_VERSION", 7)
    store.put([BlockObject("a", block), BlockObject("b", block)])  # "a" stays resident: first write wins
    slot_a, slot_b = store._lru["a"], store._lru["b"]
    assert store._prime_slot_version[[slot_a, slot_b]].tolist() == [5, 7]
    store.put([BlockObject("c", block)])  # evicts "a"
    store.put([BlockObject("a", block)])  # republished after eviction: a new stamp
    assert store._prime_slot_version[store._lru["a"]] == 7
    assert fr._STORE_COUNTERS["store_published"] == 4
    assert fr._STORE_COUNTERS["store_published"] - fr.resident_slots(store) == 2  # evicted "b" and the first "a"
    fr._PUBLISH_META["d"] = (3, True)  # provenance recorded by the connector before the put
    store.put([BlockObject("d", block)])
    assert store._prime_slot_version[store._lru["d"]] == 3 and store._prime_slot_taint[store._lru["d"]]
    assert "d" not in fr._PUBLISH_META and fr._STORE_COUNTERS["store_published_tainted"] == 1
    store.close()


def test_store_taints_a_key_published_with_other_rows():
    store_cls = type("Store", (BlockObjectStore,), {})
    fr._patch_store(store_cls)
    store = store_cls(max_bytes=4 * NB, object_nbytes=NB)
    block, other = _rows(0, B).tobytes(), _rows(0, B, base=1).tobytes()
    store.put([BlockObject("a", block)])
    store.put([BlockObject("a", block)])  # a forced recompute republishes the same rows
    assert not store._prime_slot_taint[store._lru["a"]] and fr._STORE_COUNTERS["store_conflicts"] == 0
    store.put([BlockObject("a", other)])  # a second first pass: the stored rows are not this publisher's
    store.put([BlockObject("a", other)])  # already tainted: counted once
    store.put([BlockObject("b", block), BlockObject("b", other)])  # two publishers in one put
    assert store._prime_slot_taint[[store._lru["a"], store._lru["b"]]].all()
    store.put([BlockObject("c", block)])
    fr._PUBLISH_META["c"] = (fr.UNKNOWN, True)  # a recompute with unforced known positions: not a first pass
    store.put([BlockObject("c", other)])
    assert not store._prime_slot_taint[store._lru["c"]]
    assert fr._STORE_COUNTERS["store_conflicts"] == 2 and fr._STORE_COUNTERS["store_published"] == 3
    store.close()


def test_concurrent_traces_of_one_episode_are_not_forced_to_each_others_rows(monkeypatch):
    """Best-of-n style envs run several trainable traces in one episode, so under one cache salt. Two of them compute
    the same blocks; each next turn recomputes from position 0 and must not be forced to its sibling's rows."""
    rng = np.random.default_rng(0)  # two passes over the same tokens route differently
    monkeypatch.setitem(globals(), "_natural", lambda tokens, *_: rng.integers(0, E, (len(tokens), L, K), np.uint16))
    sim = Sim()
    try:
        for rid in "ab":
            sim.add(rid, range(1, 11), salt="episode")
        sim.step({"a": 10, "b": 10})  # admitted together: both compute blocks 0 and 1
        for rid in "ab":
            sim.decode(rid, 3)  # identical sampled tokens: both paths are 1..10, 1010..1013 (rows of [0, 13))
        first = {rid: dict(sim.captured[rid]) for rid in "ab"}
        paths = {rid: list(sim.reqs[rid]["tokens"]) for rid in "ab"}
        for rid in "ab":
            sim.finish(rid)
        sim.step({})
        for rid in "ab":
            sim.add(rid + "2", paths[rid] + [500], salt="episode", prompt_start=13)  # prefix KV evicted
            mask = sim.step({rid + "2": 15})
            forced = [p for p in range(13) if mask[p]]
            assert all(np.array_equal(sim.captured[rid + "2"][p], first[rid][p]) for p in forced)
        st = sim.state.stats
        assert st["forced"] == 0 and st["known_unforced"] == 2 * 13
        assert st["store_tainted"] == 2 * 3 and fr._STORE_COUNTERS["store_conflicts"] == 3
        assert st["stash_conflicts"] == 2  # both tails [12, 13): same salt and tokens, other rows
    finally:
        sim.close()


def test_block_completed_before_its_key_keeps_the_version_it_was_computed_at(sim, monkeypatch):
    conn = sim.conn
    state = aux_worker._WorkerRequestState(aux_output_keys=["k0"])
    conn._requests["r"] = state
    monkeypatch.setattr(fr, "_WEIGHT_VERSION", 1)
    conn._publish_blocks([(state, [(0, _rows(0, B)), (4, _rows(4, B))])])  # block 1 has no key yet: pending
    assert len(state.pending_blocks) == 1
    monkeypatch.setattr(fr, "_WEIGHT_VERSION", 2)  # a weight update before the hash arrives
    state.aux_output_keys.append("k1")
    conn._publish_blocks([(state, [])])
    conn._store._queue.join()
    inner = conn._store._store
    assert inner._prime_slot_version[[inner._lru["k0"], inner._lru["k1"]]].tolist() == [1, 1]


# ----------------------------------------------------------------------------- lookup
@pytest.fixture
def connector():
    """A store holding blocks 0..3 (positions 0..15) of a previous request, keys k0..k5 (stamped version 3)."""
    store_cls = type("Store", (BlockObjectStore,), {})
    fr._patch_store(store_cls)
    store = BackgroundBlockObjectStore(store_cls(max_bytes=1 << 20, object_nbytes=NB), max_pending_batches=4)
    keys = [f"k{j}" for j in range(6)]
    fr._WEIGHT_VERSION = 3
    store.put([BlockObject(keys[j], _rows(j * B, B).tobytes()) for j in range(4)])
    store._queue.join()  # stamped on the writer thread
    conn = SimpleNamespace(
        _store=store,
        _buffer=SimpleNamespace(block_size=B, dtype=np.dtype(np.uint16), shape_per_token=(L, K)),
        _lock=Lock(),
        _requests={"a": SimpleNamespace(aux_output_keys=keys)},
    )
    yield conn
    store.close()


def test_lookup_returns_stored_full_blocks_from_the_cache_hit(connector):
    # cache hit at 4: blocks 1..3 are stored, block 4 (positions 16..19) is a full prompt block that is not
    plan, st = fr.lookup_rows(connector, None, [("a", 4, 21, 0)])
    ((start, rows, src, version),) = plan["a"]
    assert (start, rows.shape[0], src, version) == (4, 12, fr.SRC_STORE, 3)
    assert np.array_equal(rows, _rows(4, 12))
    # the partial last block is never looked up; unknown requests give nothing
    assert fr.lookup_rows(connector, None, [("a", 12, 16, 0)])[0]["a"][0][0] == 12
    assert fr.lookup_rows(connector, None, [("a", 12, 14, 0)])[0] == {}
    assert fr.lookup_rows(connector, None, [("zz", 0, 20, 0)])[0] == {}
    # hits and misses are counted only inside the known range; keys_short when hashes are missing
    _, st = fr.lookup_rows(connector, None, [("a", 0, 24, 20)])
    assert (st["store_hits"], st["store_misses"], st["keys_short"]) == (4, 1, 0)
    _, st = fr.lookup_rows(connector, None, [("a", 0, 32, 0)])
    assert (st["store_hits"], st["store_misses"], st["keys_short"]) == (0, 0, 2)


def test_lookup_splits_runs_at_version_changes_and_skips_tainted_blocks(connector):
    fr._WEIGHT_VERSION = 9
    fr._PUBLISH_META["k5"] = (9, True)
    connector._store.put([BlockObject("k4", _rows(16, B).tobytes()), BlockObject("k5", _rows(20, B).tobytes())])
    plan, st = fr.lookup_rows(connector, None, [("a", 8, 24, 24)])
    assert [(s, r.shape[0], v) for s, r, _, v in plan["a"]] == [(8, 8, 3), (16, 4, 9)]
    assert st["store_tainted"] == 1 and st["store_hits"] == 4


# ----------------------------------------------------------------------------- merging
def test_merge_prefers_later_sources_and_counts_disagreement_and_gaps():
    store_segs = [(4, _rows(4, 8), fr.SRC_STORE, 3)]  # positions 4..11
    client = (8, _rows(8, 6), 5)  # positions 8..13, agrees with the store on 8..11
    rows, segs, st, gaps = fr.merge_rows(store_segs, client, num_computed=4, known_end=14)
    assert segs == [(4, 8, fr.SRC_STORE, 3), (8, 14, fr.SRC_CLIENT, 5)]
    assert np.array_equal(rows, _rows(4, 10))
    assert {k: st[k] for k in ("store", "client", "disagree", "uncovered", "known_unforced")} == {
        "store": 4,
        "client": 6,
        "disagree": 0,
        "uncovered": 0,
        "known_unforced": 0,
    }
    assert gaps == []
    # where they disagree, the later source wins
    rows, segs, st, _ = fr.merge_rows(store_segs, (8, _rows(8, 6, base=100), 5), 4, 14)
    assert st["disagree"] == 4
    assert rows[4:, 0, 0].tolist() == [(100 + p) % E for p in range(8, 14)] and rows[:4, 0, 0].tolist() == [4, 5, 6, 7]
    # rows starting above the cache hit leave a gap of known positions without rows
    _, segs, st, gaps = fr.merge_rows([], (10, _rows(10, 3), -1), 4, 13)
    assert segs == [(10, 13, fr.SRC_CLIENT, -1)] and st["uncovered"] == 6 and gaps == [(4, 10)]
    assert st["known_unforced"] == 6
    # the stash fills the known partial block after the stored blocks
    stash = (12, _rows(12, 2), fr.SRC_STASH, 4)
    _, segs, st, gaps = fr.merge_rows([store_segs[0], stash], None, 4, 16)
    assert segs == [(4, 12, fr.SRC_STORE, 3), (12, 14, fr.SRC_STASH, 4)]
    assert (st["store"], st["stash"], st["known_unforced"], gaps) == (8, 2, 2, [(14, 16)])


@pytest.mark.parametrize(
    ("prompt_len", "prompt_start", "prefill_len", "expected"),
    [
        (20, 0, 20, (0, False)),  # first turn
        (20, 14, 20, (14, False)),  # bridged turn
        (20, 20, 20, (0, False)),  # chat request: start = prompt length
        (20, 0, 26, (25, True)),  # resumed after preemption
    ],
)
def test_known_end(prompt_len, prompt_start, prefill_len, expected):
    assert fr._known_end(prompt_len, prompt_start, prefill_len) == expected


# ----------------------------------------------------------------------------- client payload
@pytest.fixture
def state():
    s = fr.ForcedRoutingState(MAX_TOKENS, L, K, E, range(1, L), CPU)
    fr._STATE = s
    return s


@pytest.mark.parametrize(
    "payload",
    [
        _payload(0, np.zeros((3, 5, K), dtype=np.uint16)),  # wrong shape
        _payload(0, np.full((3, L, K), E, dtype=np.uint16)),  # expert id out of range
        {**_payload(0, np.zeros((3, L, K), dtype=np.uint16)), "dtype": "int32"},  # unsupported dtype
        {**_payload(0, np.zeros((3, L, K), dtype=np.uint16)), "data": "AAAA"},  # truncated data
        _payload(0, np.zeros((3, L, K), dtype=np.uint16), version=-2),  # invalid version
    ],
)
def test_bad_payload_is_ignored_not_raised(state, payload):
    state.add_request("a", 20, 0, list(range(20)), payload)
    assert "a" not in state.client and state.stats["rejected_requests"] == 1


def test_client_rows_past_the_prompt_are_rejected(state):
    state.add_request("a", 12, 0, list(range(12)), _payload(8, _rows(8, 6)))  # reaches position 13 of 12
    assert "a" not in state.client and state.stats["rejected_requests"] == 1
    state.add_request("b", 12, 0, list(range(12)), _payload(8, _rows(8, 4), version=6))
    start, rows, version = state.client["b"]
    assert (start, version) == (8, 6) and np.array_equal(rows, _rows(8, 4))


def test_rejected_payload_still_gets_stored_rows_and_ids_above_255_survive(state, connector):
    state.add_request("a", 21, 0, list(range(21)), {**_payload(16, _rows(16, 4)), "dtype": "int8"})
    state.fill(connector, ["a"], np.array([0, 17]), np.array([4]), np.array([21]), PIECEWISE)
    assert state.stats["rejected_requests"] == 1 and state.stats["store"] == 12
    assert state.ids[:12, 0, 0].tolist() == list(range(4, 16))
    wide = np.full((2, L, K), 299, dtype=np.uint16)
    state.add_request("b", 30, 0, list(range(30)), _payload(20, wide))
    connector._requests["b"] = SimpleNamespace(aux_output_keys=[])
    state.fill(connector, ["b"], np.array([0, 10]), np.array([20]), np.array([30]), PIECEWISE)
    assert state.ids[:2].unique().tolist() == [299]


# ----------------------------------------------------------------------------- fill
def test_fill_forces_stored_and_client_rows(state, connector):
    # request a recomputes 4..20 (cache hit at 4, prompt 21): stored blocks 1..3 + client rows for 16..19
    state.add_request("a", 21, 20, list(range(21)), _payload(16, _rows(16, 4)))
    state.fill(connector, ["a"], np.array([0, 17]), np.array([4]), np.array([21]), PIECEWISE)
    assert state.mask[:16].all() and not state.mask[16:].any()
    assert state.ids[:16, 0, 0].tolist() == list(range(4, 20))
    assert {k: state.stats[k] for k in ("forced", "store", "client", "known_unforced", "store_hits")} == {
        "forced": 16,
        "store": 12,
        "client": 4,
        "known_unforced": 0,
        "store_hits": 3,
    }
    assert state.rows == {}  # every segment is used: the pinned rows are released
    state.fill(connector, ["a"], np.array([0, 1]), np.array([21]), np.array([21]), PIECEWISE)  # decode
    assert not state.mask.any() and not state.dirty


def test_fill_buckets_by_source_and_age_and_trims_segments_across_chunks(connector, monkeypatch):
    s = fr.ForcedRoutingState(MAX_TOKENS, L, K, E, range(1, L), CPU, drift=True)
    fr._STATE = s
    monkeypatch.setattr(fr, "_WEIGHT_VERSION", 5)  # stored rows carry version 3
    s.add_request("a", 21, 20, list(range(21)), _payload(16, _rows(16, 4), version=5))
    chunks = [(4, 6), (10, 6), (16, 5)]  # a chunked prefill of [4, 21)
    for nc, n in chunks:
        s.fill(connector, ["a"], np.array([0, n]), np.array([nc]), np.array([21]), PIECEWISE)
        forced = min(nc + n, 20) - nc
        assert s.mask[:forced].all() and not s.mask[forced:].any()
        positions = range(nc, nc + forced)
        expected = [
            fr.age_bucket(fr.SRC_STORE, 3, 5) if p < 16 else fr.age_bucket(fr.SRC_CLIENT, 5, 5) for p in positions
        ]
        assert s.bucket[:forced].tolist() == expected
        assert s.pending_drift == (0, forced)
        assert s.ids[:forced, 0, 0].tolist() == list(positions)
    assert fr.bucket_label(expected[0]) == "client/age0" and fr.bucket_label(fr.age_bucket(0, 3, 5)) == "store/age2"
    assert s.stats["forced"] == 16 and s.rows == {}


def test_decode_only_graph_steps_are_not_marked_and_taint_their_blocks(sim):
    tokens = _first_turn(sim)
    sim.step({})
    # a one-token prompt extension over cached context, with client rows for that position
    sim.add("b", tokens[:13], num_computed=12, payload=_payload(12, _rows(12, 1)))
    mask = sim.step({"b": 1}, desc=DECODE_GRAPH)
    st = sim.state.stats
    assert not mask.any() and st["in_decode_graphs"] == 1 and st["forced"] == 0
    assert sim.conn._requests["b"]._prime_taint == {3}
    assert sim.state.rows == {}


def test_router_hook_is_a_no_op_on_steps_without_forced_tokens(state):
    state.mask[0] = True  # stale buffer contents must not matter when the step forces nothing
    calls = []
    router = SimpleNamespace(_compute_routing=lambda h, lg, it, input_ids=None: calls.append(1) or ("w", "ids"))
    fr._wrap_router(router, fr._RouterHook(0, "sigmoid", True, 1.0))
    assert router._compute_routing(None, None, None) == ("w", "ids") and calls == [1]


def test_router_hook_forces_ids_and_reweights(state):
    state.mask[1] = True
    state.dirty = True
    state.ids[1, 1] = torch.tensor([3, 0])
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.5, 0.0, 0.0, -1.0]])
    topk_ids = torch.tensor([[3, 2], [0, 1]], dtype=torch.int32)
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    hook = fr._RouterHook(layer_id=1, scoring_func="sigmoid", renormalize=True, routed_scaling_factor=2.5)
    w, ids = hook.apply(state, logits, topk_w, topk_ids)
    assert ids.tolist() == [[3, 2], [3, 0]] and ids.dtype == torch.int32
    assert torch.equal(w[0], topk_w[0])
    s = logits[1].sigmoid()[[3, 0]]
    assert torch.allclose(w[1], s / s.sum() * 2.5)


# ----------------------------------------------------------------------------- the real router
def _grouped_topk_router(monkeypatch, scoring_func, renormalize, scale):
    from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)  # the GPU factory branch, as on the server
    grouped = scoring_func == "softmax"  # without grouping, softmax top-k is a CUDA-only fused kernel
    return create_fused_moe_router(
        top_k=8,
        global_num_experts=288,
        renormalize=renormalize,
        use_grouped_topk=True,
        num_expert_group=4 if grouped else 1,
        topk_group=2 if grouped else 1,
        scoring_func=scoring_func,
        routed_scaling_factor=scale,
        e_score_correction_bias=torch.randn(288) * 0.05 if scoring_func == "sigmoid" else None,
    )


@pytest.mark.parametrize(("scoring_func", "renormalize", "scale"), [("sigmoid", True, 2.5), ("softmax", False, 1.0)])
def test_router_wrapper_on_vllms_grouped_topk_router(monkeypatch, scoring_func, renormalize, scale):
    torch.manual_seed(0)
    router = _grouped_topk_router(monkeypatch, scoring_func, renormalize, scale)
    assert type(router)._select_experts is BaseRouter._select_experts
    n, layer = 12, 2
    s = fr.ForcedRoutingState(n, 4, 8, 288, [1, 2, 3], CPU, drift=True)
    fr._STATE = s
    captured = {}
    router.set_capture_fn(lambda ids: captured.__setitem__("ids", ids.clone()))
    fr._wrap_router(router, fr._RouterHook(layer, scoring_func, renormalize, scale))
    logits, hidden = torch.randn(n, 288), torch.empty(n, 16)
    natural_w, natural_ids = router.select_experts(hidden, logits)  # not dirty: untouched
    assert s.natural.abs().sum() == 0
    forced = torch.stack([torch.randperm(288)[:8] for _ in range(n)])
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 2, 5]] = True
    s.ids[:n, layer], s.mask[:n], s.dirty = forced, mask, True
    w, ids = router.select_experts(hidden, logits)
    assert torch.equal(ids[mask].long(), forced[mask]) and torch.equal(captured["ids"][mask].long(), forced[mask])
    assert torch.equal(ids[~mask], natural_ids[~mask]) and torch.equal(w[~mask], natural_w[~mask])
    assert torch.equal(s.natural[:n, layer].long(), natural_ids.long())
    # forcing the router's own ids reproduces its own weights
    s.ids[:n, layer], s.mask[:n] = natural_ids.long(), True
    w, ids = router.select_experts(hidden, logits)
    assert torch.equal(ids, natural_ids) and torch.allclose(w, natural_w, rtol=1e-6, atol=1e-7)


# ----------------------------------------------------------------------------- drift
def _drift_reference(ids, natural, mask, bucket, moe_layers, num_buckets):
    ref = np.zeros((num_buckets, ids.shape[1], 3), dtype=np.int64)
    for t in np.flatnonzero(mask):
        for layer in np.flatnonzero(moe_layers):
            miss = len(set(ids[t, layer].tolist()) - set(natural[t, layer].tolist()))
            ref[bucket[t], layer] += [1, int(miss > 0), miss]
    return ref


def test_reduce_drift_matches_the_per_pair_reference_across_chunks():
    n, layers, top_k = 2600, 4, 3
    s = fr.ForcedRoutingState(n, layers, top_k, 400, [1, 3], CPU, drift=True)
    g = torch.Generator().manual_seed(1)
    natural = torch.stack([torch.randperm(400, generator=g)[:top_k] for _ in range(n * layers)]).view(n, layers, top_k)
    ids = natural.clone().long()
    flip = torch.rand(n, layers, generator=g) < 0.3
    ids[flip, 0] = 256 + torch.randint(0, 144, (int(flip.sum()),), generator=g)  # ids >= 256 survive the int16 copy
    mask = torch.rand(n, generator=g) < 0.6
    bucket = torch.randint(0, fr.NUM_BUCKETS, (n,), generator=g)
    s.ids.copy_(ids), s.natural.copy_(natural), s.mask.copy_(mask), s.bucket.copy_(bucket)
    s.pending_drift = (5, n)  # rows [0, 5) are outside the step's marked range
    s.reduce_drift()
    ref = _drift_reference(
        ids.numpy(),
        natural.numpy(),
        mask.numpy() & (np.arange(n) >= 5),
        bucket.numpy(),
        s.moe_layers.numpy(),
        fr.NUM_BUCKETS,
    )
    assert np.array_equal(s.drift.numpy(), ref) and ref[:, [0, 2], 0].sum() == 0  # dense layers masked


@pytest.mark.parametrize("log_due", [False, True])
def test_drift_counter_flags_a_one_row_shift(monkeypatch, log_due):
    """Forced rows one batch row off their tokens disagree with the router almost always: the alarm fires. A log
    line due on the forcing step must not reduce its drift before its forward has run."""
    n, layers, top_k = 2600, 5, 4  # 4 MoE layers x 2600 tokens > DRIFT_ALARM_MIN_PAIRS
    monkeypatch.setenv("PRIME_RL_FORCED_ROUTING_LOG_INTERVAL", "0" if log_due else "1e9")
    g = torch.Generator().manual_seed(0)
    natural = torch.stack([torch.randperm(64, generator=g)[:top_k] for _ in range(n * layers)]).view(n, layers, top_k)
    first_pass = natural.clone()
    flip = torch.rand(n, layers, generator=g) < 0.2  # two passes differ in ~20% of sets
    first_pass[flip, 0] = (first_pass[flip, 0] + 1) % 64
    rows = first_pass.numpy().astype(np.uint16)
    ratios = {}
    for shift in (0, 1):
        s = fr.ForcedRoutingState(n, layers, top_k, 64, range(1, layers), CPU, drift=True)
        fr._STATE = s
        s.rows["r"] = (0, fr._host_rows(rows.copy()), [(0, n, fr.SRC_STORE, fr.UNKNOWN)])
        s.first_chunk_seen.add("r")
        s.fill(None, ["r"], np.array([0, n]), np.array([0]), np.array([n]), PIECEWISE)
        if shift:
            s.ids[:n] = torch.roll(s.ids[:n], shift, dims=0)  # the bug the alarm must catch
        for layer in range(layers):
            fr._RouterHook(layer, "sigmoid", True, 1.0).apply(
                s, torch.zeros(n, 64), torch.zeros(n, top_k), natural[:, layer].int()
            )
        s.fill(None, [], np.array([0]), np.array([], np.int32), np.array([], np.int32), PIECEWISE)  # flushes it
        drift = s.drift.numpy()
        pairs, differ, _ = drift.sum(axis=(0, 1))
        ratios[shift] = differ / pairs
        assert pairs == n * (layers - 1)
        assert (fr.drift_alarm(drift) is not None) == bool(shift)
    assert 0.15 < ratios[0] < 0.25 and ratios[1] > 0.9


# ----------------------------------------------------------------------------- multi-turn episodes on vLLM's connector
def test_bridged_turn_forces_stored_blocks_and_the_stashed_tail(sim):
    tokens = _first_turn(sim)  # path 15: rows of [0, 14) captured, the partial block [12, 14) is stashed at teardown
    turn1 = sim.captured["a"]
    sim.add("b", tokens + [100, 101, 102], prompt_start=14, num_computed=8)  # known end 14, cache hit 8
    mask = sim.step({"b": 10})
    st = sim.state.stats
    assert mask[:6].all() and not mask[6:].any()  # [8, 12) from the store, [12, 14) from the stash
    assert {k: st[k] for k in ("store", "stash", "known_unforced", "store_hits", "stash_hits")} == {
        "store": 4,
        "stash": 2,
        "known_unforced": 0,
        "store_hits": 1,
        "stash_hits": 1,
    }
    for p in range(8, 14):
        assert np.array_equal(sim.captured["b"][p], turn1[p])
    assert sim.state.stash.snapshot()["stash_saved"] == 1
    assert fr._STORE_COUNTERS["store_published_tainted"] == 0


def test_a_stashed_tail_needs_the_same_tokens_and_salt(sim):
    tokens = _first_turn(sim)
    sim.step({})
    edited = tokens[:13] + [7] + tokens[14:]
    sim.add("b", edited + [100], prompt_start=14, num_computed=8)
    sim.add("c", tokens + [100], salt="other", prompt_start=14, num_computed=8)
    sim.step({"b": 7, "c": 7})
    st = sim.state.stats
    assert st["stash_misses"] == 2 and st["stash_hits"] == 0  # "b": other tokens; "c": other salt, other keys
    assert st["known_unforced"] == 2 + 6  # b: [12, 14); c: [8, 14) (its blocks were never stored under its salt)


def test_unforced_known_positions_taint_the_block_and_it_is_never_forced_later():
    sim = Sim(stash_bytes=0)
    try:
        tokens = _first_turn(sim)
        sim.add("b", tokens + [100, 101, 102], prompt_start=14, num_computed=8)
        sim.step({"b": 10})
        assert sim.state.stats["known_unforced"] == 2  # no stash: [12, 14) routes naturally
        sim.decode("b", 2)
        path = list(sim.reqs["b"]["tokens"])
        sim.finish("b")
        sim.step({})
        sim.conn._store._queue.join()
        assert fr._STORE_COUNTERS["store_published_tainted"] == 1  # block 3 ([12, 16)) of "b"
        assert len(path) == 21
        sim.add("c", path + [200], prompt_start=20, num_computed=8)
        mask = sim.step({"c": 14})
        st = sim.state.stats
        assert st["store_tainted"] == 1 and st["store_hits"] == 1 + 3  # b: block 2; c: blocks 2 to 4
        assert mask[:4].all() and not mask[4:8].any() and mask[8:12].all()  # block 3 routes naturally
        assert st["known_unforced"] == 2 + 4
    finally:
        sim.close()


def test_preempted_request_resumes_with_its_stored_blocks_and_stashed_tail(sim):
    sim.add("a", range(1, 11))
    sim.step({"a": 10})
    sim.decode("a", 4)  # path 15, rows of [0, 14)
    tokens = list(sim.reqs["a"]["tokens"])
    first = dict(sim.captured["a"])
    sim.finish("a")  # preempted: the scheduler sends the same terminal event
    sim.step({})
    sim.add("a", tokens, num_computed=8, prompt_len=10)  # resumed: prefill = prompt + outputs
    mask = sim.step({"a": len(tokens) - 8})
    st = sim.state.stats
    assert st["readmitted"] == 1 and st["resumed_unforced"] == 0 and st["known_unforced"] == 0
    assert mask[:6].all() and not mask[6:].any()
    for p in range(8, 14):
        assert np.array_equal(sim.captured["a"][p], first[p])


def test_stash_is_bounded_by_bytes_and_skips_block_zero_tails():
    entry_bytes = 2 * L * K * 2 + 2 * 4  # a 2-row tail and its tokens
    sim = Sim(stash_bytes=2 * entry_bytes)
    try:
        for i, rid in enumerate("abc"):
            _first_turn(sim, rid, salt=f"s{i}")
            sim.step({})
        stash = sim.state.stash.snapshot()
        assert (stash["stash_saved"], stash["stash_evictions"], stash["stash_entries"]) == (3, 1, 2)
        assert stash["stash_bytes"] == 2 * entry_bytes
        _first_turn(sim, "z", prompt=1, path=3)  # tail in block 0: no parent key
        sim.step({})
        assert sim.state.stash.snapshot()["stash_unaddressable"] == 1
        assert sim.state.stash.tokens == {}  # finished requests leave no token tails behind
    finally:
        sim.close()


def test_stash_token_tail_keeps_the_last_block():
    tail = fr._TokenTail(list(range(10)), keep=4, version=1)
    assert tail.offset == 6 and tail.get(6, 10).tolist() == [6, 7, 8, 9] and tail.get(5, 7) is None
    tail.extend([10, 11, 12, 13, 14], keep=4, version=2)
    assert tail.offset == 11 and tail.get(11, 15).tolist() == [11, 12, 13, 14] and tail.version == 2


# ----------------------------------------------------------------------------- tensor parallelism
def test_tp_ranks_end_up_with_identical_buffers(monkeypatch, connector):
    shared = {}

    class Group:
        def broadcast_object(self, obj, src=0):
            if obj is not None:
                shared["plan"] = pickle.dumps(obj)
                return obj
            return pickle.loads(shared.pop("plan"))

    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: Group())
    rank1_conn = SimpleNamespace(_store=None, _buffer=None, _requests={})
    ranks = [
        fr.ForcedRoutingState(MAX_TOKENS, L, K, E, range(1, L), CPU, tp_rank=r, tp_size=2, drift=True) for r in range(2)
    ]
    steps = [
        (["a"], np.array([0, 9]), np.array([4]), np.array([21])),
        (["a"], np.array([0, 8]), np.array([13]), np.array([21])),
    ]
    for s, conn in zip(ranks, (connector, rank1_conn)):
        fr._STATE = s
        s.add_request("a", 21, 20, list(range(21)), _payload(16, _rows(16, 4)))
        for req_ids, qsl, nc, pl in steps:
            s.fill(conn, req_ids, qsl, nc, pl, PIECEWISE)
    r0, r1 = ranks
    assert torch.equal(r0.ids, r1.ids) and torch.equal(r0.mask, r1.mask) and torch.equal(r0.bucket, r1.bucket)
    assert {k: r0.stats[k] for k in fr._SHARED_COUNTERS} == {k: r1.stats[k] for k in fr._SHARED_COUNTERS}
    assert r0.stats["forced"] == 16 and r1.stats["store_hits"] == 0 and r0.stats["store_hits"] == 3


# ----------------------------------------------------------------------------- bind
class _FakeRunner(torch.nn.Module):
    def __init__(self, router, layer_id, monolithic=False, sink=None):
        super().__init__()
        self.router, self.layer_id = router, layer_id
        self.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=monolithic), routing_sink=sink)

    @property
    def _quant_method(self):
        return self.routed_experts.quant_method


def _vllm_config(**parallel):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, enable_expert_parallel=False, **parallel),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )


@pytest.fixture
def bind_env(monkeypatch, sim):
    import vllm.model_executor.layers.fused_moe.layer as moe_layer

    monkeypatch.setattr(moe_layer, "MoERunner", _FakeRunner)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    fr._STATE = None

    def model(*runners):
        m = torch.nn.Module()
        m.runners = torch.nn.ModuleList(runners)
        m.weight = torch.nn.Parameter(torch.zeros(1))
        return m

    def router(**overrides):
        r = _grouped_topk_router(monkeypatch, "sigmoid", True, 2.5)
        r.top_k = K
        for k, v in overrides.items():
            setattr(r, k, v)
        return r

    return SimpleNamespace(conn=sim.conn, model=model, router=router)


def test_bind_wraps_every_router_and_reuses_buffers_on_a_rebind(bind_env):
    routers = [bind_env.router(), bind_env.router()]
    model = bind_env.model(_FakeRunner(routers[0], 1), _FakeRunner(routers[1], 2))
    assert fr.bind_forced_routing(bind_env.conn, model, _vllm_config()) == 2
    state = fr._STATE
    assert state.layout == (MAX_TOKENS, L, K, 288, (1, 2), CPU) and state.stash is not None
    assert all("_compute_routing" in vars(r) and r._prime_forced.layer_id == i + 1 for i, r in enumerate(routers))
    wrapped = routers[0]._compute_routing
    assert fr.bind_forced_routing(bind_env.conn, model, _vllm_config()) == 2
    assert fr._STATE is state and routers[0]._compute_routing is wrapped
    other = bind_env.model(_FakeRunner(bind_env.router(), 1))
    with pytest.raises(RuntimeError, match="cannot rebind"):
        fr.bind_forced_routing(bind_env.conn, other, _vllm_config())


def _captures_first(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
    self.capture_fn(router_logits)
    return self._compute_routing(hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("dp", "data parallelism"),
        ("ep", "expert parallelism"),
        ("spec", "speculative decoding"),
        ("order", "no longer routes before it captures"),
        ("shared", "fused shared experts"),
        ("layer", "outside the 3 captured layers"),
        ("experts", "disagree on the expert count"),
        ("monolithic", "monolithic MoE kernels"),
        ("sink", "routing sink"),
        ("source", "captures routing outside a BaseRouter"),
        ("none", "no MoE routers"),
    ],
)
def test_bind_refuses_unsupported_setups_loudly(bind_env, monkeypatch, case, match):
    config = _vllm_config()
    runners = [_FakeRunner(bind_env.router(), 1)]
    if case == "dp":
        config.parallel_config.data_parallel_size = 2
    elif case == "ep":
        config.parallel_config.enable_expert_parallel = True
    elif case == "spec":
        config.speculative_config = object()
    elif case == "order":
        monkeypatch.setattr(BaseRouter, "_select_experts", _captures_first)
    elif case == "shared":
        runners = [_FakeRunner(bind_env.router(num_fused_shared_experts=1), 1)]
    elif case == "layer":
        runners = [_FakeRunner(bind_env.router(), L)]
    elif case == "experts":
        runners.append(_FakeRunner(bind_env.router(global_num_experts=64), 2))
    elif case == "monolithic":
        runners = [_FakeRunner(bind_env.router(), 1, monolithic=True)]
    elif case == "sink":
        runners = [_FakeRunner(bind_env.router(), 1, sink=object())]
    elif case == "source":
        source = torch.nn.Module()
        source.layer_id, source.capture_fn = 1, None
        runners.append(source)
    elif case == "none":
        runners = []
    with pytest.raises(RuntimeError, match=match):
        fr.bind_forced_routing(bind_env.conn, bind_env.model(*runners), config)
    assert fr._STATE is None


def test_select_experts_still_routes_before_it_captures():
    source = inspect.getsource(BaseRouter._select_experts)
    assert 0 <= source.find("self._compute_routing(") < source.find("self.capture_fn(")


# ----------------------------------------------------------------------------- vLLM glue
def test_capture_leaves_the_ops_out_of_decode_only_full_graphs(monkeypatch, state):
    descs = [
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 512, None),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 64, 64, uniform_token_count=1),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 64, 64, max_query_len=1),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 512, 64),
    ]
    seen = []

    class Manager:
        def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
            for desc in descs:  # warmup forward, then the capture forward (as CudaGraphManager.capture does)
                create_forward_fn(desc, warmup=True)(CUDAGraphMode.NONE)
                create_forward_fn(desc, warmup=False)(desc.cg_mode)

    def forward_factory(desc, warmup):
        return lambda mode: seen.append((desc.cg_mode, warmup, fr._CAPTURING_DECODE_ONLY_GRAPH))

    messages = []
    monkeypatch.setattr(fr, "_logger", lambda: SimpleNamespace(info=messages.append, warning=messages.append))
    fr._patch_capture(Manager)
    Manager().capture(forward_factory)
    assert [flag for _, _, flag in seen] == [False, False, True, True, True, True, False, False]
    assert not fr._CAPTURING_DECODE_ONLY_GRAPH
    assert messages == ["Forced routing: 2 decode-only CUDA graphs captured without forcing ops (3 FULL graphs in all)"]


def test_runner_hooks_pass_vllms_request_and_batch_fields(monkeypatch, state):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    calls = []

    class Runner:
        aux_output_connector = "conn"

        def add_requests(self, scheduler_output):
            pass

        def _remove_request(self, req_id):
            return True

        def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
            return SimpleNamespace(
                req_ids=["a"], query_start_loc_np="qsl", num_computed_tokens_np="nc", prefill_len_np="pl"
            )

    fr._patch_runner(Runner)
    monkeypatch.setattr(state, "fill", lambda *args: calls.append(args))
    params = SamplingParams(max_tokens=4, extra_args={fr.EXTRA_ARGS_KEY: _payload(8, _rows(8, 2))})
    params.routed_experts_prompt_start = 10
    new = NewRequestData("a", list(range(12)), [], params, None, ([],), 0, None, prefill_token_ids=list(range(12)))
    runner = Runner()
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=[new]))
    assert state.info["a"] == (10, False) and state.client["a"][0] == 8
    runner.prepare_inputs("so", "brs", PIECEWISE, 0)
    assert calls == [("conn", ["a"], "qsl", "nc", "pl", PIECEWISE)]
    assert runner._remove_request("a") is True and "a" not in state.info


def test_sampled_tokens_reach_the_stash(sim):
    class Output:
        def get_output(self):
            return SimpleNamespace(req_ids=["a", "b"], sampled_token_ids=[[7], []])

    fr._patch_async_output(Output)
    sim.state.stash.add_request("a", [1, 2, 3], 0)
    Output().get_output()
    assert sim.state.stash.tokens["a"].get(0, 4).tolist() == [1, 2, 3, 7]


def test_engine_core_sends_each_weight_version_to_the_workers():
    calls = []

    class Core:
        def set_weight_version(self, weight_version):
            self.version = weight_version

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            calls.append((method, args))
            return [fr._worker_rpc(None, *args)]

    fr._patch_engine_core(Core)
    core = Core()
    core.set_weight_version("12")
    assert core.version == "12" and calls == [(fr.RPC_METHOD, ("set_weight_version", 12))] and fr._WEIGHT_VERSION == 12
    core.set_weight_version("default")
    assert fr._WEIGHT_VERSION == fr.UNKNOWN


def test_worker_refuses_to_start_where_bind_would_never_run():
    class Worker:
        def __init__(self, vllm_config, local_rank=0):
            self.vllm_config = vllm_config

    fr._patch_worker(Worker)

    def config(capture=True, v2=True):
        aux = SimpleNamespace(enable_return_routed_experts=capture)
        return SimpleNamespace(aux_output_config=aux, use_v2_model_runner=v2)

    assert Worker(vllm_config=config(), local_rank=0).vllm_config.use_v2_model_runner
    with pytest.raises(RuntimeError, match="routed-experts capture is off"):
        Worker(vllm_config=config(capture=False))
    with pytest.raises(RuntimeError, match="V1 model runner"):
        Worker(config(v2=False))
    assert Worker.prime_rl_forced_routing is fr._worker_rpc


def test_worker_rpc_resolves_through_vllms_run_method(state):
    from vllm.v1.serial_utils import run_method

    worker = type("Worker", (), {fr.RPC_METHOD: fr._worker_rpc})()
    run_method(worker, fr.RPC_METHOD, ("set_weight_version", 4), {})
    assert fr._WEIGHT_VERSION == 4 and state.stats["weight_versions_set"] == 1
    assert run_method(worker, fr.RPC_METHOD, ("stats",), {})["weight_version"] == 4
    with pytest.raises(ValueError, match="unknown forced-routing RPC"):
        run_method(worker, fr.RPC_METHOD, ("nope",), {})


def test_stats_and_log_lines(monkeypatch):
    s = Sim(drift=True)
    try:
        tokens = _first_turn(s)
        s.add("b", tokens + [100], prompt_start=14, num_computed=8)
        s.step({"b": 8})
        s.step({"b": 1})  # flushes the drift of the forcing step
        out = fr.forced_routing_stats()
        assert out["bound"] and out["tp_rank"] == 0 and out["forced"] == 6 and out["store_slots"] == 64
        assert out["store_published"] == 4 and out["store_evicted"] == 0 and out["stash_hits"] == 1
        assert (
            set(out["drift"]) == {"store/ageunknown", "stash/ageunknown"} and out["drift"]["store/ageunknown"][0] == 8
        )
        assert len(out["drift_by_layer"]) == L
        rank1 = {"tp_rank": 1, "weight_version": -1, "admit_ms": 99.0, "admit_ms_max": 50.0, "admissions_over_4mib": 0}
        combined = fr.combine_stats([out, rank1])
        assert combined["admit_ms_max"] == 50.0 and combined["weight_version_by_rank"] == [-1, -1]
        lines = fr.format_log_lines(fr._STATE.counters(), fr._STATE.drift_snapshot()[0])
        assert re.match(r"Forced routing: [0-9]+ prompt tokens forced", lines[0])  # parsed by validation drivers
        assert lines[1].startswith("Forced routing drift") and "store/ageunknown" in lines[1]
        messages = []
        monkeypatch.setattr(fr, "_logger", lambda: SimpleNamespace(info=messages.append, warning=messages.append))
        monkeypatch.setenv("PRIME_RL_FORCED_ROUTING_LOG_INTERVAL", "0")
        s.step({"b": 1})
        assert len(messages) == 2 and messages[0].startswith("Forced routing: 6 prompt tokens forced")
    finally:
        s.close()


def test_off_by_default_leaves_vllm_untouched():
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    targets = [
        (BlockObjectStore, "put"),
        (AuxOutputWorkerConnector, "__init__"),
        (AuxOutputWorkerConnector, "_publish_blocks"),
        (AuxOutputWorkerConnector, "_teardown"),
        (GPUModelRunner, "add_requests"),
        (GPUModelRunner, "_remove_request"),
        (GPUModelRunner, "prepare_inputs"),
        (AsyncOutput, "get_output"),
        (CudaGraphManager, "capture"),
        (EngineCore, "set_weight_version"),
        (Worker, "__init__"),
    ]
    before = [vars(cls).get(name) for cls, name in targets]
    fr.monkey_patch_forced_routing()
    assert [vars(cls).get(name) for cls, name in targets] == before
    assert not hasattr(Worker, fr.RPC_METHOD)
