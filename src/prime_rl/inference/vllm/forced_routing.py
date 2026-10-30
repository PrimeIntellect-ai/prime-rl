"""Forced MoE routing for recomputed prompt positions (router replay on prefill), vLLM 0.32 V2 model runner.

Opt-in: ``PRIME_RL_FORCED_ROUTING=1`` (written by ``setup_vllm_env`` from ``inference.forced_routing.enabled``). When
it is off, no vLLM class is touched.

With prefix caching, every turn after the first makes vLLM recompute part of the old context (the KDA state of
hybrid models is only kept at prompt boundaries, so a long previous completion is recomputed in full). Routing those
positions again disagrees with the first pass in about a third of (token, layer) top-k sets, while the trainer
replays the first pass. This module makes vLLM route recomputed positions as they were first routed:

- Store rows: vLLM's routed-experts store keys every full hash block by its hash (tokens + cache salt) and keeps a
  key's first published rows. On a request's first scheduled chunk, TP rank 0 looks up every stored full block from
  the cache hit on.
- Stash rows: the store never holds a request's partial last block. When a request finishes (or is preempted), TP
  rank 0 keeps that block's rows in a bounded LRU stash keyed by the parent full block's key, with the tokens they
  were computed for. A bridged next turn of the same episode (``routed_experts_prompt_start = known_end``) finds its
  known partial block ``[known_end // B * B, known_end)`` there, verified token by token.
- Client rows (``sampling_params.extra_args["routed_experts_replay"]``, ``{start, data (b64), shape [n, L, K],
  dtype, version?}``), still accepted; prime-rl does not send them.

Rank 0 broadcasts its plan to every TP rank. Later sources win where they overlap (store < stash < client;
disagreements are counted). The router uses the known ids; the combine weights come from this forward's gate scores
at those ids. Known positions without rows route normally and are counted (``known_unforced``,
``resumed_unforced``). A stored block whose known positions were not all forced when it was computed is tainted: it
is never forced later (``store_tainted``). So is a block two requests published with different rows (concurrent traces
of one episode share its cache salt; ``store_conflicts``), and a stashed tail with such a twin (``stash_conflicts``).

The ops are static-shape on persistent device buffers, so they stay valid inside CUDA graphs; decode-only FULL graphs
are captured without them. Drift: for every forced (token, MoE layer) pair, whether the forced top-k set differs
from the top-k this forward picks, by source and by age in weight updates.
"""

from __future__ import annotations

import base64
import bisect
import inspect
import os
import threading
import time
from array import array
from collections import OrderedDict

import numpy as np
import torch

EXTRA_ARGS_KEY = "routed_experts_replay"
RPC_METHOD = "prime_rl_forced_routing"
UNKNOWN = -1
SOURCES = ("store", "stash", "client")
SRC_STORE, SRC_STASH, SRC_CLIENT = range(3)
AGE_LABELS = ("unknown", "0", "1", "2", "3", "4-7", "8-15", "16+")
_AGE_EDGES = (1, 2, 3, 4, 8, 16)
NUM_BUCKETS = len(SOURCES) * len(AGE_LABELS)
_REDUCE_CHUNK = 2048
# Plans above this size spill from the TP group's shared-memory broadcast queue to ZMQ.
_BROADCAST_SLOT_BYTES = 4 << 20
# Drift ratios above this at age unknown/0 mean the natural and forced buffers are misaligned. The low side has no
# alarm: two prefill passes at identical weights already disagree in about 20% of top-k sets.
DRIFT_ALARM_RATIO = 0.6
DRIFT_ALARM_MIN_PAIRS = 10_000

# Counters every TP rank computes identically.
_SHARED_COUNTERS = (
    "forced",
    "store",
    "stash",
    "client",
    "disagree",
    "uncovered",
    "known_unforced",
    "resumed_unforced",
    "in_decode_graphs",
    "admissions",
    "readmitted",
    "requests_with_store_rows",
    "requests_with_stash_rows",
    "requests_with_client_rows",
    "rejected_requests",
    "weight_versions_set",
)
# Counters only TP rank 0 (the store owner) has.
_RANK0_COUNTERS = (
    "store_hits",
    "store_misses",
    "store_tainted",
    "stash_hits",
    "stash_misses",
    "stash_tainted",
    "stash_conflicts",
    "keys_short",
    "plan_bytes",
    "lookup_failures",
)
# Counters each rank measures for itself.
_RANK_COUNTERS = ("admit_ms", "admit_ms_max", "admissions_over_4mib")


def enabled() -> bool:
    return os.environ.get("PRIME_RL_FORCED_ROUTING", "0") == "1"


def _drift_enabled() -> bool:
    return os.environ.get("PRIME_RL_FORCED_ROUTING_DRIFT", "1") == "1"


def _log_interval() -> float:
    return float(os.environ.get("PRIME_RL_FORCED_ROUTING_LOG_INTERVAL", "60"))


def _stash_max_bytes() -> int:
    return int(os.environ.get("PRIME_RL_FORCED_ROUTING_STASH_MIB", "2048")) << 20


def _logger():
    from vllm.logger import init_logger

    return init_logger("vllm.prime_rl.forced_routing")


_STATE: ForcedRoutingState | None = None
_CAPTURING_DECODE_ONLY_GRAPH = False
# Policy version of the loaded weights, identical on every worker (set by the EngineCore.set_weight_version wrapper).
_WEIGHT_VERSION = UNKNOWN
# key -> (version, tainted) of blocks about to be published; filled before the put is queued, popped by the writer.
# If two requests publish the same key before the writer gets to the first put (only with a shared cache salt, e.g.
# concurrent traces of one episode), the slot gets the later version and either taint.
_PUBLISH_META: dict[str, tuple[int, bool]] = {}
# TP rank 0, written on the store's writer thread.
_STORE_COUNTERS = {"store_published": 0, "store_published_tainted": 0, "store_conflicts": 0}


def age_bucket(src: int, version: int, current: int) -> int:
    if version < 0 or current < 0 or current < version:
        age = 0
    else:
        age = 1 + bisect.bisect_right(_AGE_EDGES, current - version)
    return src * len(AGE_LABELS) + age


def bucket_label(bucket: int) -> str:
    return f"{SOURCES[bucket // len(AGE_LABELS)]}/age{AGE_LABELS[bucket % len(AGE_LABELS)]}"


def parse_weight_version(version) -> int:
    text = str(version).strip()
    return int(text) if text.lstrip("-").isdigit() else UNKNOWN


# ----------------------------------------------------------------------------- store (TP rank 0)
def slot_table(inner) -> dict:
    """key -> slot. vLLM 0.32.0 keeps it in ``_lru`` (OrderedDict[str, int]); vllm#62392 moves it to ``_slots``."""
    table = getattr(inner, "_slots", None)
    return table if isinstance(table, dict) else inner._lru


def slot_of(table, key) -> int | None:
    slot = table.get(key)
    return slot if isinstance(slot, int) and slot >= 0 else None


def resident_slots(inner) -> int:
    table = slot_table(inner)
    if table is getattr(inner, "_lru", None):
        return len(table)
    return sum(1 for slot in list(table.values()) if isinstance(slot, int) and slot >= 0)


def _patch_store(store_cls) -> None:
    """Stamp each newly allocated slot with its block's provenance (weight version, taint) and count publications.

    A key published untainted with rows that differ from the rows it already holds (or twice in one put with different
    rows) had two first passes, from two requests sharing a cache salt; its slot is tainted (``store_conflicts``). A
    forced recompute republishes identical rows.
    """
    original_put = store_cls.put

    def put(self, objects, *, retain_keys=(), release_keys=()):
        table = slot_table(self)
        payloads: dict[str, bytes] = {}
        conflicts = set()
        for obj in objects:
            if payloads.setdefault(obj.key, obj.payload) != obj.payload:
                conflicts.add(obj.key)
        meta = {key: _PUBLISH_META.pop(key, None) for key in payloads}
        resident = {key for key in payloads if slot_of(table, key) is not None}
        original_put(self, objects, retain_keys=retain_keys, release_keys=release_keys)
        if not payloads:
            return
        versions = self.__dict__.get("_prime_slot_version")
        if versions is None:
            versions = self._prime_slot_version = np.full(self.num_slots, UNKNOWN, dtype=np.int64)
            self._prime_slot_taint = np.zeros(self.num_slots, dtype=bool)
        table, size = slot_table(self), self.object_nbytes
        for key, payload in payloads.items():
            slot = slot_of(table, key)
            if slot is None:
                continue
            version, tainted = meta[key] or (_WEIGHT_VERSION, False)
            if key in resident:  # the first publication's rows stay
                # Tainted rows are not a first pass, so they say nothing about the stored ones.
                if not tainted and self._arena[slot * size : (slot + 1) * size] != payload:
                    conflicts.add(key)
            else:
                versions[slot] = version
                self._prime_slot_taint[slot] = tainted
                _STORE_COUNTERS["store_published"] += 1
                _STORE_COUNTERS["store_published_tainted"] += int(tainted)
            if key in conflicts and not self._prime_slot_taint[slot]:
                self._prime_slot_taint[slot] = True
                _STORE_COUNTERS["store_conflicts"] += 1

    store_cls.put = put


def label_published_blocks(connector, batches) -> None:
    """Record, before ``_publish_blocks`` queues them, the provenance of the blocks it will publish.

    A block's version is the weight version of the step whose capture completed it (a block completed before its
    hash arrived keeps that version); it is tainted when known positions in it were not forced.
    """
    block = connector._buffer.block_size
    current = _WEIGHT_VERSION
    for state, completed in batches:
        pending = state.pending_blocks
        if not completed and not pending:
            continue
        extra = state.__dict__
        taint = extra.get("_prime_taint") or ()
        prov = extra.setdefault("_prime_prov", {})
        keys = state.aux_output_keys
        keyed_end = len(keys) * block
        for start, _ in completed:
            prov[start] = current
        for start, _ in [*pending, *completed]:
            if start < keyed_end:
                key = keys[start // block]
                earlier = _PUBLISH_META.get(key)  # another request's label the writer has not consumed yet
                tainted = start // block in taint or (earlier is not None and earlier[1])
                _PUBLISH_META[key] = (prov.pop(start, current), tainted)


def lookup_rows(connector, stash, items) -> tuple[dict, dict]:
    """Rows this server holds for each newly admitted request (TP rank 0).

    items: [(req_id, num_computed, prefill_len, known_end)]. Returns ({req_id: [(start, rows, src, version)]}, stats).
    Store: runs of stored full prefill blocks from ceil(num_computed / B), split where the stamped version changes;
    tainted blocks are skipped. Stash: the known partial block [known_end // B * B, known_end).
    """
    store, buffer = connector._store, connector._buffer
    block = buffer.block_size
    plan: dict[str, list] = {}
    stats = dict.fromkeys(_RANK0_COUNTERS, 0)
    with connector._lock:
        store._queue.join()  # puts queued by earlier steps are applied
        inner = store._store
        table = slot_table(inner)
        versions = inner.__dict__.get("_prime_slot_version")
        taint = inner.__dict__.get("_prime_slot_taint")
        for req_id, num_computed, prefill_len, known_end in items:
            state = connector._requests.get(req_id)
            if state is None:
                continue
            keys = state.aux_output_keys
            first, want = -(-num_computed // block), prefill_len // block
            last = min(want, len(keys))
            stats["keys_short"] += want - last  # 0 on 0.32.0, which sends every hash at the first schedule
            known_last = min(known_end // block, last)
            runs: list[tuple[list[int], int]] = []
            for j in range(first, last):
                slot = slot_of(table, keys[j])
                if j < known_last:
                    stats["store_hits" if slot is not None else "store_misses"] += 1
                if slot is None:
                    continue
                if taint is not None and taint[slot]:
                    stats["store_tainted"] += 1
                    continue
                version = int(versions[slot]) if versions is not None else UNKNOWN
                if runs and runs[-1][0][-1] == j - 1 and runs[-1][1] == version:
                    runs[-1][0].append(j)
                else:
                    runs.append(([j], version))
            segs = []
            for idx, version in runs:
                payload = inner.get_concatenated([keys[j] for j in idx])
                rows = np.frombuffer(payload, dtype=buffer.dtype).reshape((-1, *buffer.shape_per_token)).copy()
                segs.append((idx[0] * block, rows, SRC_STORE, version))
            if stash is not None:
                found = stash.find(req_id, keys, num_computed, stats)
                if found is not None:
                    segs.append(found)
            for _, rows, _, _ in segs:
                stats["plan_bytes"] += rows.nbytes
            if segs:
                plan[req_id] = segs
    return plan, stats


# ----------------------------------------------------------------------------- tail stash (TP rank 0)
class _TokenTail:
    """The last tokens of a running request (prefill tokens, then sampled tokens as outputs arrive)."""

    __slots__ = ("offset", "ids", "version")

    def __init__(self, token_ids, keep: int, version: int):
        cut = max(0, len(token_ids) - keep)
        self.offset = cut
        self.ids = array("i", token_ids[cut:])
        self.version = version

    def extend(self, token_ids, keep: int, version: int) -> None:
        self.ids.extend(token_ids)
        self.version = version
        if len(self.ids) > 2 * keep:
            drop = len(self.ids) - keep
            del self.ids[:drop]
            self.offset += drop

    def get(self, start: int, end: int) -> np.ndarray | None:
        if start < self.offset or end > self.offset + len(self.ids):
            return None
        return np.frombuffer(self.ids, dtype=np.int32)[start - self.offset : end - self.offset].copy()


class _TailEntry:
    __slots__ = ("start", "tokens", "rows", "version", "tainted", "parent")

    def __init__(self, start, tokens, rows, version, tainted, parent):
        self.start, self.tokens, self.rows = start, tokens, rows
        self.version, self.tainted, self.parent = version, tainted, parent

    @property
    def nbytes(self) -> int:
        return self.rows.nbytes + self.tokens.nbytes


class TailStash:
    """Partial last blocks of finished or preempted requests, for the next turn of the same episode.

    Keyed by the key of the block before the tail, which hashes every earlier token and the cache salt, and verified
    against the tail's own tokens. LRU-bounded by bytes. Thread-safe: requests are added on the main thread, outputs
    arrive on the output thread, and teardown runs on either.
    """

    def __init__(self, block_size: int, max_bytes: int, max_live: int):
        self.block_size = block_size
        self.max_bytes = max_bytes
        self.max_live = max_live
        self.keep = block_size + 16
        self.lock = threading.Lock()
        self.tokens: dict[str, _TokenTail] = {}
        self.live: set[str] = set()
        self.want: dict[str, tuple[int, np.ndarray]] = {}
        self.entries: OrderedDict[int, _TailEntry] = OrderedDict()
        self.by_parent: dict[str, list[_TailEntry]] = {}
        self.bytes = 0
        self.stats = {"stash_saved": 0, "stash_unaddressable": 0, "stash_evictions": 0}

    def add_request(self, req_id: str, token_ids, known_end: int) -> None:
        j, n = divmod(known_end, self.block_size)
        with self.lock:
            self.live.add(req_id)
            self.tokens[req_id] = _TokenTail(token_ids, self.keep, _WEIGHT_VERSION)
            if j and n:
                self.want[req_id] = (
                    j * self.block_size,
                    np.asarray(token_ids[j * self.block_size : known_end], np.int32),
                )
            if len(self.tokens) > self.max_live:
                for stale in [r for r in self.tokens if r not in self.live]:
                    del self.tokens[stale]

    def remove_request(self, req_id: str, connector_has_request: bool) -> None:
        with self.lock:
            self.live.discard(req_id)
            self.want.pop(req_id, None)
            if not connector_has_request:  # teardown already ran, or never will
                self.tokens.pop(req_id, None)

    def record(self, req_ids, sampled_token_ids) -> None:
        version = _WEIGHT_VERSION
        with self.lock:
            for req_id, token_ids in zip(req_ids, sampled_token_ids):
                tail = self.tokens.get(req_id)
                if tail is not None and token_ids:
                    tail.extend(token_ids, self.keep, version)

    def save(self, connector, req_ids) -> None:
        """Stash the partial last block of requests about to be torn down (under ``connector._lock``)."""
        buffer = connector._buffer
        for req_id in req_ids:
            tail = buffer._requests.get(req_id)
            state = connector._requests.get(req_id)
            if tail is None or tail.length == 0 or state is None:
                continue
            j = tail.block_start // self.block_size
            with self.lock:
                token_tail = self.tokens.get(req_id)
                tokens = token_tail.get(tail.block_start, tail.block_start + tail.length) if token_tail else None
            if j == 0 or len(state.aux_output_keys) < j or tokens is None:
                self.stats["stash_unaddressable"] += 1
                continue
            entry = _TailEntry(
                tail.block_start,
                tokens,
                buffer._rows[tail.slot, : tail.length].copy(),
                token_tail.version,
                j in (state.__dict__.get("_prime_taint") or ()),
                state.aux_output_keys[j - 1],
            )
            self._put(entry)

    def forget(self, req_ids) -> None:
        with self.lock:
            for req_id in req_ids:
                if req_id not in self.live:
                    self.tokens.pop(req_id, None)

    def _put(self, entry: _TailEntry) -> None:
        if entry.nbytes > self.max_bytes:
            return
        with self.lock:
            self.entries[id(entry)] = entry
            self.by_parent.setdefault(entry.parent, []).append(entry)
            self.bytes += entry.nbytes
            self.stats["stash_saved"] += 1
            while self.bytes > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self._unlink(old)
                self.stats["stash_evictions"] += 1

    def _unlink(self, entry: _TailEntry) -> None:
        siblings = self.by_parent[entry.parent]
        siblings.remove(entry)
        if not siblings:
            del self.by_parent[entry.parent]
        self.bytes -= entry.nbytes

    def find(self, req_id: str, keys, num_computed: int, stats: dict):
        """The stashed rows of this request's known partial block, as a plan segment, or None."""
        with self.lock:
            want = self.want.pop(req_id, None)
            if want is None:
                return None
            start, tokens = want
            n, parent = len(tokens), start // self.block_size - 1
            if start + n <= num_computed:  # all of it is a prefix-cache hit
                return None
            if parent >= len(keys):
                stats["stash_misses"] += 1
                return None
            matches = [
                entry
                for entry in self.by_parent.get(keys[parent], ())
                if entry.start == start and len(entry.tokens) >= n and np.array_equal(entry.tokens[:n], tokens)
            ]
            if not matches:
                stats["stash_misses"] += 1
                return None
            entry = matches[-1]
            self.entries.move_to_end(id(entry))
            if entry.tainted:
                stats["stash_tainted"] += 1
                return None
            # Untainted tails with the same salt and tokens but other rows: concurrent traces of one episode.
            if any(not other.tainted and not np.array_equal(other.rows[:n], entry.rows[:n]) for other in matches[:-1]):
                stats["stash_conflicts"] += 1
                return None
            stats["stash_hits"] += 1
            return start, entry.rows[:n].copy(), SRC_STASH, entry.version

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.stats, "stash_entries": len(self.entries), "stash_bytes": self.bytes}


# ----------------------------------------------------------------------------- merging (all ranks, deterministic)
def merge_rows(server_segs, client, num_computed: int, known_end: int):
    """Merge the rank-0 plan and the client rows of one request; later sources win (store < stash < client).

    server_segs: [(start, rows, src, version)]; client: (start, rows, version) or None. Returns (rows, segments,
    stats, gaps): rows (uint16) covers [num_computed, hi); segments are maximal runs [start, end) of one (src, version);
    gaps are the known positions [num_computed, known_end) without rows. stats: positions per source, disagree
    (positions a later source overwrote with other ids), uncovered (gaps below the last position with rows) and
    known_unforced.
    """
    lo = num_computed
    parts = list(server_segs)
    if client is not None:
        parts.append((client[0], client[1], SRC_CLIENT, client[2]))
    parts.sort(key=lambda part: part[2])
    hi = max((start + rows.shape[0] for start, rows, _, _ in parts), default=lo)
    stats = dict.fromkeys((*SOURCES, "disagree", "uncovered", "known_unforced"), 0)
    if hi <= lo:
        gaps = [(lo, known_end)] if known_end > lo else []
        stats["known_unforced"] = max(0, known_end - lo)
        return None, [], stats, gaps
    rows = np.zeros((hi - lo, *parts[0][1].shape[1:]), dtype=np.uint16)
    src = np.full(hi - lo, -1, dtype=np.int8)
    ver = np.full(hi - lo, UNKNOWN, dtype=np.int64)
    for start, part, part_src, version in parts:
        a, b = max(start, lo), start + part.shape[0]
        if b <= a:
            continue
        new = part[a - start : b - start]
        seen = src[a - lo : b - lo] >= 0
        if seen.any():
            differs = (rows[a - lo : b - lo] != new).reshape(b - a, -1).any(axis=1)
            stats["disagree"] += int((seen & differs).sum())
        rows[a - lo : b - lo], src[a - lo : b - lo], ver[a - lo : b - lo] = new, part_src, version
    have = src >= 0
    for i, name in enumerate(SOURCES):
        stats[name] = int((src == i).sum())
    last = int(np.flatnonzero(have)[-1]) + 1 if have.any() else 0
    stats["uncovered"] = last - int(have[:last].sum())
    k = max(0, min(known_end, hi) - lo)
    edges = np.flatnonzero(np.diff(np.concatenate(([1], have[:k].view(np.int8), [1]))))
    gaps = [(lo + int(a), lo + int(b)) for a, b in zip(edges[::2], edges[1::2])]
    if known_end > hi:
        gaps.append((hi, known_end))
    stats["known_unforced"] = sum(b - a for a, b in gaps)
    cuts = np.flatnonzero((src[1:] != src[:-1]) | (ver[1:] != ver[:-1])) + 1
    bounds = np.concatenate(([0], cuts, [hi - lo]))
    segments = [
        (lo + int(a), lo + int(b), int(src[a]), int(ver[a])) for a, b in zip(bounds[:-1], bounds[1:]) if src[a] >= 0
    ]
    return rows, segments, stats, gaps


def _host_rows(rows: np.ndarray) -> torch.Tensor:
    """uint16 bits viewed as int16 (exact for ids < 32768), pinned once per request; widened on the device copy."""
    tensor = torch.from_numpy(rows.view(np.int16))
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


def _blocks(intervals, block: int) -> set[int]:
    return {j for a, b in intervals if b > a for j in range(a // block, (b - 1) // block + 1)}


def _decode_only(desc) -> bool:
    from vllm.config import CUDAGraphMode

    return (
        desc is not None
        and desc.cg_mode == CUDAGraphMode.FULL
        and (desc.max_query_len or desc.uniform_token_count) == 1
    )


def _known_end(prompt_len: int, prompt_start: int, prefill_len: int) -> tuple[int, bool]:
    """End of the positions whose first-pass routing exists, and whether the request resumed after preemption."""
    if prefill_len > prompt_len:  # resumed: rows of [0, prefill_len - 1) were computed before
        return prefill_len - 1, True
    if 0 < prompt_start < prompt_len:  # bridged train turn: [0, prompt_start) were computed by earlier turns
        return prompt_start, False
    return 0, False  # first turn, or a chat request (start = prompt length)


# ----------------------------------------------------------------------------- per-worker state
class _DriftSnapshot:
    """Non-blocking copies of the drift counters into pinned memory; reads return the newest finished copy."""

    def __init__(self, source: torch.Tensor):
        self.source = source
        pin = source.is_cuda
        self.host = [torch.zeros(source.shape, dtype=source.dtype, pin_memory=pin) for _ in range(2)]
        self.event = torch.cuda.Event() if source.is_cuda else None
        self.pending: int | None = None
        self.ready: int | None = None
        self.ready_at = 0.0
        self.issued_at = 0.0

    def poll(self) -> tuple[np.ndarray | None, float]:
        """Start a copy if none is in flight; return (newest finished copy, its age in seconds)."""
        if self.pending is not None and (self.event is None or self.event.query()):
            self.ready, self.ready_at, self.pending = self.pending, self.issued_at, None
        if self.pending is None:
            index = 1 if self.ready == 0 else 0
            self.host[index].copy_(self.source, non_blocking=self.event is not None)
            self.issued_at = time.monotonic()
            if self.event is not None:
                self.event.record()
                self.pending = index
            else:
                self.ready, self.ready_at = index, self.issued_at
        if self.ready is None:
            return None, 0.0
        return self.host[self.ready].numpy().copy(), time.monotonic() - self.ready_at


class ForcedRoutingState:
    """Per-worker buffers shared by every MoE router, filled once per step."""

    def __init__(
        self,
        max_tokens: int,
        num_layers: int,
        top_k: int,
        num_experts: int,
        moe_layers,
        device,
        tp_rank: int = 0,
        tp_size: int = 1,
        connector=None,
        drift: bool = True,
        stash: TailStash | None = None,
    ):
        def zeros(*shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.max_tokens = max_tokens
        self.ids = zeros(max_tokens, num_layers, top_k, dtype=torch.int64)  # gather index; unwritten rows hold expert 0
        self.mask = zeros(max_tokens, dtype=torch.bool)
        self.moe_layers = zeros(num_layers, dtype=torch.bool)
        self.moe_layers[list(moe_layers)] = True
        self.natural = self.bucket = self.drift = None
        if drift:
            chunk = min(_REDUCE_CHUNK, max_tokens)
            self.natural = zeros(max_tokens, num_layers, top_k, dtype=torch.int16)  # this forward's own pick
            self.bucket = zeros(max_tokens, dtype=torch.int64)  # (source, age) bucket per batch row
            self.drift = zeros(NUM_BUCKETS, num_layers, 3, dtype=torch.int64)  # pairs, sets differ, ids differ
            self._eq = zeros(chunk, num_layers, top_k, top_k, dtype=torch.bool)
            self._found = zeros(chunk, num_layers, top_k, dtype=torch.bool)
            self._miss = zeros(chunk, num_layers, dtype=torch.int64)
            self._pair = zeros(chunk, num_layers, dtype=torch.bool)
            self._differ = zeros(chunk, num_layers, dtype=torch.bool)
            self._values = zeros(chunk, num_layers, 3, dtype=torch.int64)
            self._snapshot = _DriftSnapshot(self.drift)
        self.num_layers, self.top_k, self.num_experts = num_layers, top_k, num_experts
        self.tp_rank, self.tp_size, self.connector, self.stash = tp_rank, tp_size, connector, stash
        self.info: dict[str, tuple[int, bool]] = {}  # req -> (known_end, resumed)
        self.client: dict[str, tuple[int, np.ndarray, int]] = {}
        # req -> (first position, pinned rows from there, segments [(start, end, src, version)])
        self.rows: dict[str, tuple[int, torch.Tensor, list[tuple[int, int, int, int]]]] = {}
        self.first_chunk_seen: set[str] = set()
        self.dirty = False
        self.pending_drift: tuple[int, int] | None = None
        self.stats = dict.fromkeys((*_SHARED_COUNTERS, *_RANK0_COUNTERS, *_RANK_COUNTERS), 0)
        self.layout = (max_tokens, num_layers, top_k, num_experts, tuple(moe_layers), self.ids.device)
        self._last_log = time.monotonic()

    def reset(self, connector, stash: TailStash | None) -> None:
        """Rebind to a new connector (a new KV cache), keeping the device buffers and counters."""
        self.connector, self.stash = connector, stash
        self.info.clear()
        self.client.clear()
        self.rows.clear()
        self.first_chunk_seen.clear()
        self.mask.zero_()
        self.dirty, self.pending_drift = False, None

    # -- requests (every rank sees the same SchedulerOutput)
    def _decode(self, payload: dict) -> tuple[int, np.ndarray, int]:
        shape = [int(x) for x in payload["shape"]]
        if len(shape) != 3 or shape[1:] != [self.num_layers, self.top_k] or shape[0] < 1:
            raise ValueError(f"shape {shape} is not [n >= 1, {self.num_layers}, {self.top_k}]")
        start = int(payload.get("start", 0))
        if start < 0:
            raise ValueError(f"start {start} < 0")
        version = int(payload.get("version", UNKNOWN))
        if version < UNKNOWN:
            raise ValueError(f"version {version} is neither >= 0 nor {UNKNOWN} (unknown)")
        dtype = np.dtype(payload.get("dtype", "uint16"))
        if dtype not in (np.uint8, np.uint16):
            raise ValueError(f"dtype {dtype} is not uint8 or uint16")
        raw = base64.b64decode(payload["data"])
        if len(raw) != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"{len(raw)} bytes do not match shape {shape} x {dtype}")
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        if int(arr.max()) >= self.num_experts:
            raise ValueError(f"expert id {int(arr.max())} >= {self.num_experts}")
        return start, arr.copy(), version

    def add_request(self, req_id: str, prompt_len: int, prompt_start: int, token_ids, payload) -> None:
        known_end, resumed = _known_end(prompt_len, prompt_start, len(token_ids))
        self.info[req_id] = (known_end, resumed)
        if self.stash is not None:
            self.stash.add_request(req_id, token_ids, known_end)
        if payload is None:
            return
        try:
            start, rows, version = self._decode(payload)
            if start + rows.shape[0] > prompt_len:
                raise ValueError(f"rows cover [{start}, {start + rows.shape[0]}) beyond the prompt ({prompt_len})")
            self.client[req_id] = (start, rows, version)
            self.stats["requests_with_client_rows"] += 1
        except Exception as e:  # noqa: BLE001 - a bad payload must not take down the engine
            self.stats["rejected_requests"] += 1
            _logger().warning(f"Forced routing: ignoring routed_experts_replay of request {req_id}: {e}")

    def remove_request(self, req_id: str) -> None:
        for table in (self.client, self.rows, self.info):
            table.pop(req_id, None)
        self.first_chunk_seen.discard(req_id)
        if self.stash is not None:
            self.stash.remove_request(req_id, req_id in self.connector._requests)

    def _admit(self, connector, new) -> None:
        """First chunk of each new request: rank 0's plan (store + stash), broadcast, merged with client rows."""
        t0 = time.perf_counter()
        items = [(r, nc, pl, *self.info.get(r, (0, False))) for r, nc, pl in new]
        plan = None
        if self.tp_rank == 0:
            try:
                plan, st = lookup_rows(connector, self.stash, [(r, nc, pl, ke) for r, nc, pl, ke, _ in items])
                for k, v in st.items():
                    self.stats[k] += v
            except Exception as e:  # noqa: BLE001 - a failed lookup must not desynchronize the TP ranks
                self.stats["lookup_failures"] += 1
                _logger().warning(f"Forced routing: row lookup failed, using client rows only: {e}")
                plan = {}
        if self.tp_size > 1:
            from vllm.distributed import get_tp_group

            plan = get_tp_group().broadcast_object(plan, src=0)
        plan_bytes = 0
        for r, nc, _, known_end, resumed in items:
            server_segs = plan.get(r, [])
            plan_bytes += sum(rows.nbytes for _, rows, _, _ in server_segs)
            self.stats["admissions"] += 1
            self.stats["readmitted"] += int(resumed)
            for src, key in ((SRC_STORE, "requests_with_store_rows"), (SRC_STASH, "requests_with_stash_rows")):
                self.stats[key] += int(any(seg[2] == src for seg in server_segs))
            rows, segments, st, gaps = merge_rows(server_segs, self.client.pop(r, None), nc, known_end)
            for k in (*SOURCES, "disagree", "uncovered"):
                self.stats[k] += st[k]
            self.stats["resumed_unforced" if resumed else "known_unforced"] += st["known_unforced"]
            if segments:
                self.rows[r] = (nc, _host_rows(rows), segments)
            if gaps and self.tp_rank == 0:
                self._taint(connector, r, gaps)
        elapsed = (time.perf_counter() - t0) * 1e3
        self.stats["admit_ms"] += elapsed
        self.stats["admit_ms_max"] = max(self.stats["admit_ms_max"], elapsed)
        self.stats["admissions_over_4mib"] += int(plan_bytes > _BROADCAST_SLOT_BYTES)

    def _taint(self, connector, req_id: str, intervals) -> None:
        """Mark the blocks holding known positions computed unforced; they are published tainted (TP rank 0)."""
        state = connector._requests.get(req_id)
        if state is not None:
            blocks = _blocks(intervals, connector._buffer.block_size)
            state.__dict__.setdefault("_prime_taint", set()).update(blocks)

    # -- steps
    def fill(self, connector, req_ids, qsl, nc_np, pl_np, batch_desc) -> None:
        """Mark this step's tokens whose absolute position (num_computed + j) has rows."""
        if self.pending_drift is not None:
            self.reduce_drift()
        if self.tp_rank == 0:  # before this step is marked: its forward has not run yet
            self._maybe_log()
        if self.dirty:
            self.mask.zero_()
            self.dirty = False
        new = [(r, int(nc_np[i]), int(pl_np[i])) for i, r in enumerate(req_ids) if r not in self.first_chunk_seen]
        if new:  # identical on every TP rank, so all of them take part in the broadcast
            self.first_chunk_seen.update(r for r, _, _ in new)
            self._admit(connector, new)
        if self.rows:
            decode_graph = _decode_only(batch_desc)  # those graphs have no forcing ops
            current = _WEIGHT_VERSION
            n_forced = n_skipped = 0
            lo_mark, hi_mark = self.max_tokens, 0
            for i, r in enumerate(req_ids):
                entry = self.rows.get(r)
                if entry is None:
                    continue
                first, rows, segments = entry
                nc, t_lo, t_hi = int(nc_np[i]), int(qsl[i]), int(qsl[i + 1])
                end = nc + (t_hi - t_lo)
                for start, stop, src, version in segments:
                    p_lo, p_hi = max(nc, start), min(end, stop)
                    if p_hi <= p_lo:
                        continue
                    if decode_graph:
                        n_skipped += p_hi - p_lo
                        if self.tp_rank == 0:
                            self._taint(connector, r, [(p_lo, p_hi)])
                        continue
                    a, b = t_lo + (p_lo - nc), t_lo + (p_hi - nc)
                    self.ids[a:b].copy_(rows[p_lo - first : p_hi - first], non_blocking=True)
                    self.mask[a:b] = True
                    if self.bucket is not None:
                        self.bucket[a:b] = age_bucket(src, version, current)
                    n_forced += b - a
                    lo_mark, hi_mark = min(lo_mark, a), max(hi_mark, b)
                rest = [seg for seg in segments if seg[1] > end]
                if rest:
                    self.rows[r] = (first, rows, rest)
                else:
                    del self.rows[r]
            self.stats["in_decode_graphs"] += n_skipped
            if n_forced:
                self.dirty = True
                self.stats["forced"] += n_forced
                if self.drift is not None:
                    self.pending_drift = (lo_mark, hi_mark)

    def reduce_drift(self) -> None:
        """Accumulate the last forcing step's drift; eager, on the stream that ran the forward, no host sync."""
        lo, hi = self.pending_drift
        self.pending_drift = None
        for a in range(lo, hi, self._eq.shape[0]):
            b = min(hi, a + self._eq.shape[0])
            m = b - a
            eq, found, miss = self._eq[:m], self._found[:m], self._miss[:m]
            pair, differ, values = self._pair[:m], self._differ[:m], self._values[:m]
            torch.eq(self.ids[a:b].unsqueeze(-1), self.natural[a:b].unsqueeze(-2), out=eq)
            torch.any(eq, dim=-1, out=found)
            torch.sum(found, dim=-1, out=miss)
            miss.neg_().add_(self.top_k)  # forced ids the router would not pick now
            torch.logical_and(self.mask[a:b, None], self.moe_layers[None, :], out=pair)
            torch.gt(miss, 0, out=differ)
            differ.logical_and_(pair)
            values[..., 0].copy_(pair)
            values[..., 1].copy_(differ)
            torch.mul(miss, pair, out=values[..., 2])
            self.drift.index_add_(0, self.bucket[a:b], values)

    def drift_snapshot(self) -> tuple[np.ndarray | None, float]:
        if self.drift is None:
            return None, 0.0
        if self.pending_drift is not None:
            self.reduce_drift()
        return self._snapshot.poll()

    def counters(self) -> dict:
        out = {
            "tp_rank": self.tp_rank,
            "weight_version": _WEIGHT_VERSION,
            **{k: self.stats[k] for k in _RANK_COUNTERS},
        }
        if self.tp_rank != 0:
            return out
        out |= {k: self.stats[k] for k in (*_SHARED_COUNTERS, *_RANK0_COUNTERS)}
        out["requests_with_pending_rows"] = len(self.rows)
        conn = self.connector
        if conn is not None and conn._store is not None:
            inner = conn._store._store
            resident = resident_slots(inner)
            out |= {
                **_STORE_COUNTERS,
                "store_slots_used": resident,
                "store_slots": inner.num_slots,
                "store_evicted": max(0, _STORE_COUNTERS["store_published"] - resident),
            }
        if self.stash is not None:
            out |= self.stash.snapshot()
        return out

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < _log_interval():
            return
        self._last_log = now
        log = _logger()
        drift, _ = self.drift_snapshot()
        for line in format_log_lines(self.counters(), drift):
            log.info(line)
        alarm = drift_alarm(drift)
        if alarm:
            log.warning(alarm)


def drift_table(drift: np.ndarray) -> dict:
    """{bucket label: [pairs, sets differ, ids differ]} for buckets with pairs."""
    totals = drift.sum(axis=1)
    return {bucket_label(b): totals[b].tolist() for b in range(drift.shape[0]) if totals[b, 0] > 0}


def drift_alarm(drift: np.ndarray | None) -> str | None:
    """A warning when forced and natural ids disagree far above the two-pass floor at age unknown/0 (misalignment)."""
    if drift is None:
        return None
    totals = drift.sum(axis=1)
    for src, name in enumerate(SOURCES):
        base = src * len(AGE_LABELS)
        pairs, differ = totals[base : base + 2, :2].sum(axis=0)
        if pairs >= DRIFT_ALARM_MIN_PAIRS and differ / pairs > DRIFT_ALARM_RATIO:
            return (
                f"Forced routing drift: {name} rows at age unknown/0 differ from this forward's routing in "
                f"{differ / pairs:.2f} of {pairs} (token, layer) pairs (> {DRIFT_ALARM_RATIO}); forced and natural "
                "ids look misaligned"
            )
    return None


def format_log_lines(st: dict, drift: np.ndarray | None) -> list[str]:
    admissions = max(st["admissions"], 1)
    lines = [
        f"Forced routing: {st['forced']} prompt tokens forced ({st['store']} from stored blocks, {st['stash']} from "
        f"stashed tails, {st['client']} from client rows), {st['disagree']} source disagreements, "
        f"{st['uncovered']} uncovered known positions, {st['in_decode_graphs']} in decode-only graphs; requests with "
        f"stored rows {st['requests_with_store_rows']}, with stashed rows {st['requests_with_stash_rows']}, with "
        f"client rows {st['requests_with_client_rows']}, rejected {st['rejected_requests']}; known unforced "
        f"{st['known_unforced']}, resumed unforced {st['resumed_unforced']} (readmitted {st['readmitted']}); store "
        f"lookups {st['store_hits']} hit / {st['store_misses']} miss / {st['store_tainted']} tainted, keys short "
        f"{st['keys_short']}; stash {st.get('stash_hits', 0)} hit / {st.get('stash_misses', 0)} miss / "
        f"{st.get('stash_tainted', 0)} tainted / {st['stash_conflicts']} conflicting, {st.get('stash_entries', 0)} "
        f"entries, {st.get('stash_evictions', 0)} evicted; store {st.get('store_slots_used', 0)}/"
        f"{st.get('store_slots', 0)} slots, {st.get('store_evicted', 0)} evicted, "
        f"{st.get('store_published_tainted', 0)} published tainted, {st.get('store_conflicts', 0)} conflicting; "
        f"admit {st['admit_ms'] / admissions:.1f} ms/admission (max {st['admit_ms_max']:.1f}); weight version "
        f"{st['weight_version']}"
    ]
    if drift is not None:
        parts = []
        for label, (pairs, differ, ids) in drift_table(drift).items():
            parts.append(f"{label} {differ / pairs:.2f} ({ids / max(differ, 1):.1f}) of {pairs}")
        per_layer = drift.sum(axis=0)
        ratios = [(per_layer[l, 1] / per_layer[l, 0], l) for l in range(per_layer.shape[0]) if per_layer[l, 0] > 0]
        worst = ", ".join(f"L{l} {r:.2f}" for r, l in sorted(ratios, reverse=True)[:3])
        lines.append(
            "Forced routing drift (sets differ / pairs, ids differ / sets differ, of pairs): "
            + ("; ".join(parts) or "no forced pairs yet")
            + (f"; worst layers {worst}" if worst else "")
        )
    return lines


def _hook_needed(state: ForcedRoutingState) -> bool:
    """Decode-only CUDA graphs are captured without the ops; other graphs with them (the mask decides at replay);
    an eager forward runs them only when this step forces a token."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return not _CAPTURING_DECODE_ONLY_GRAPH
    return state.dirty


class _RouterHook:
    def __init__(self, layer_id: int, scoring_func: str, renormalize: bool, routed_scaling_factor: float):
        self.layer_id = layer_id
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor

    def apply(self, state: ForcedRoutingState, router_logits, topk_weights, topk_ids):
        n = topk_ids.shape[0]
        if state.natural is not None:
            state.natural[:n, self.layer_id].copy_(topk_ids)  # this forward's own pick, before the override
        mask = state.mask[:n].unsqueeze(1)
        forced = state.ids[:n, self.layer_id]
        if self.scoring_func == "sigmoid":
            w = router_logits.gather(1, forced).float().sigmoid()  # elementwise: equal to sigmoid then gather
        else:
            w = router_logits.float().softmax(dim=-1).gather(1, forced)
        if self.renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            w = w * self.routed_scaling_factor
        ids = torch.where(mask, forced.to(topk_ids.dtype), topk_ids)
        weights = torch.where(mask, w.to(topk_weights.dtype), topk_weights)
        return weights, ids


def _wrap_router(router, hook: _RouterHook) -> None:
    """Force between routing and capture: BaseRouter._select_experts calls self._compute_routing, then capture_fn."""
    original = router._compute_routing

    def _compute_routing(hidden_states, router_logits, indices_type, *, input_ids=None):
        topk_weights, topk_ids = original(hidden_states, router_logits, indices_type, input_ids=input_ids)
        state = _STATE
        if state is None or not _hook_needed(state):
            return topk_weights, topk_ids
        return hook.apply(state, router_logits, topk_weights, topk_ids)

    router._compute_routing = _compute_routing
    router._prime_forced = hook


# ----------------------------------------------------------------------------- bind
def _unsupported_reason(vllm_config) -> str | None:
    pc = vllm_config.parallel_config
    kv = vllm_config.kv_transfer_config
    checks = (
        (pc.data_parallel_size > 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (getattr(pc, "use_sequence_parallel_moe", False), "sequence-parallel MoE"),
        (getattr(pc, "use_ubatching", False), "micro-batching (DBO)"),
        (vllm_config.speculative_config is not None, "speculative decoding"),
        (kv is not None and kv.has_connector("NixlConnector"), "P/D disaggregation (NIXL)"),
    )
    return next((why for bad, why in checks if bad), None)


def _environment_reason(connector, model) -> str | None:
    """Class-level facts, so every TP rank reaches the same decision."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCaptureSource
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    source = inspect.getsource(BaseRouter._select_experts)
    if not 0 <= source.find("self._compute_routing(") < source.find("self.capture_fn("):
        return "BaseRouter._select_experts no longer routes before it captures"
    if not all(hasattr(connector, a) for a in ("_store", "_buffer", "_lock", "_requests", "_capturer")):
        return "AuxOutputWorkerConnector lacks _store/_buffer/_lock/_requests/_capturer"
    for module in model.modules():
        if isinstance(module, RoutedExpertsCaptureSource):
            return f"{type(module).__name__} captures routing outside a BaseRouter"
        if isinstance(module, MoERunner) and getattr(module.routed_experts, "routing_sink", None) is not None:
            return "a monolithic MoE kernel captures routing through a routing sink"
    return None


def bind_forced_routing(connector, model, vllm_config) -> int:
    """Attach the forcing hook to every MoE router (where the routed-experts capturer binds). Raises if it cannot."""
    global _STATE
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    log = _logger()
    reason = _unsupported_reason(vllm_config)
    if reason is None:
        try:
            reason = _environment_reason(connector, model)
        except Exception as e:  # noqa: BLE001 - refuse with the reason instead of failing differently per rank
            reason = f"environment probe failed: {type(e).__name__}: {e}"
    hooks, experts = [], set()
    if reason is None:
        max_tokens = connector._capturer.device_buffer.shape[0]  # max_num_batched_tokens
        num_layers, top_k = connector._capturer.shape_per_token  # all decoder layers, dense ones included
        runners = [m for m in model.modules() if isinstance(m, MoERunner)]
        if any(m._quant_method.is_monolithic for m in runners):
            reason = "monolithic MoE kernels route inside the kernel"
        for m in runners if reason is None else ():
            router = m.router
            scoring = getattr(router, "scoring_func", None)
            checks = (
                (not isinstance(router, BaseRouter), f"unsupported router {type(router).__name__}"),
                (
                    type(router)._select_experts is not BaseRouter._select_experts,
                    f"{type(router).__name__} overrides _select_experts",
                ),
                (scoring not in ("sigmoid", "softmax"), f"unsupported scoring function {scoring}"),
                (getattr(router, "num_fused_shared_experts", 0) != 0, "fused shared experts widen the top-k"),
                (int(router.top_k) != top_k, f"router top_k {router.top_k} != captured {top_k}"),
                (not 0 <= m.layer_id < num_layers, f"layer_id {m.layer_id} outside the {num_layers} captured layers"),
            )
            reason = next((why for bad, why in checks if bad), None)
            if reason is not None:
                break
            experts.add(int(router.global_num_experts))
            hooks.append(
                (
                    router,
                    _RouterHook(
                        m.layer_id,
                        scoring,
                        bool(getattr(router, "renormalize", True)),
                        float(getattr(router, "routed_scaling_factor", 1.0)),
                    ),
                )
            )
        if reason is None and len(experts) > 1:
            reason = f"routers disagree on the expert count {sorted(experts)}"
        if reason is None and not hooks:
            reason = "no MoE routers"
    if reason is not None:
        raise RuntimeError(f"Forced routing is enabled (PRIME_RL_FORCED_ROUTING=1) but cannot bind: {reason}")
    for why, bad in (
        ("a KV transfer/offload connector", vllm_config.kv_transfer_config is not None),
        ("LoRA", vllm_config.lora_config is not None),
    ):
        if bad:
            log.warning(f"Forced routing with {why} is not validated")
    tp_rank = get_tensor_model_parallel_rank()
    device = next(model.parameters()).device
    moe_layers = [h.layer_id for _, h in hooks]
    num_experts = experts.pop()
    stash = None
    if connector._buffer is not None and _stash_max_bytes() > 0:
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        stash = TailStash(connector._buffer.block_size, _stash_max_bytes(), max_live=4 * max_seqs)
    if _STATE is not None:  # a rebind: CUDA graphs captured so far read the existing buffers
        if _STATE.layout != (max_tokens, num_layers, top_k, num_experts, tuple(moe_layers), device):
            raise RuntimeError("Forced routing cannot rebind: the model's routing layout or device changed")
        _STATE.reset(connector, stash)
    else:
        _STATE = ForcedRoutingState(
            max_tokens,
            num_layers,
            top_k,
            num_experts,
            moe_layers,
            device,
            tp_rank,
            get_tensor_model_parallel_world_size(),
            connector,
            drift=_drift_enabled(),
            stash=stash,
        )
    for router, hook in hooks:
        if getattr(router, "_prime_forced", None) is None:
            _wrap_router(router, hook)
    log.info(
        f"Forced routing bound to {len(hooks)} MoE routers ({num_layers} layers x top-{top_k}, {num_experts} experts)"
    )
    if tp_rank == 0:
        store = connector._store._store
        log.info(
            f"Forced routing: hash block size {connector._buffer.block_size}, store {store.num_slots} slots, "
            f"tail stash {_stash_max_bytes() >> 20} MiB, drift counter {'on' if _STATE.drift is not None else 'off'}"
        )
    return len(hooks)


# ----------------------------------------------------------------------------- weight version, stats, RPC
def set_weight_version(version: int) -> None:
    """Runs on every worker after each weight update (the EngineCore.set_weight_version wrapper)."""
    global _WEIGHT_VERSION
    _WEIGHT_VERSION = int(version)
    if _STATE is not None:
        _STATE.stats["weight_versions_set"] += 1


def forced_routing_stats() -> dict:
    """This worker's counters; TP rank 0 also returns the shared, store and drift counters."""
    state = _STATE
    if state is None:
        return {"enabled": enabled(), "bound": False, "weight_version": _WEIGHT_VERSION}
    out = {"enabled": True, "bound": True, **state.counters()}
    if state.tp_rank != 0:
        return out
    # The newest finished device-to-host copy: on a GPU it lags this call by the copy just started.
    drift, age = state.drift_snapshot()
    if drift is not None:
        out["drift"] = drift_table(drift)
        out["drift_by_layer"] = drift.sum(axis=0)[:, :2].tolist()  # [layer][pairs, sets differ]
        out["drift_snapshot_age_s"] = round(age, 3)
    return out


def combine_stats(per_rank: list[dict]) -> dict:
    """The /forced_routing/stats response: TP rank 0's counters, the max over ranks of the admission timings."""
    out = dict(per_rank[0])
    for key in _RANK_COUNTERS:
        values = [r[key] for r in per_rank if key in r]
        if values:
            out[key] = max(values)
    out["weight_version_by_rank"] = [r.get("weight_version") for r in per_rank]
    return out


def _worker_rpc(self, cmd: str, *args):
    if cmd == "set_weight_version":
        return set_weight_version(int(args[0]))
    if cmd == "stats":
        return forced_routing_stats()
    raise ValueError(f"unknown forced-routing RPC {cmd!r}")


# ----------------------------------------------------------------------------- install
def _patch_connector(connector_cls) -> None:
    original_init = connector_cls.__init__
    original_publish = connector_cls._publish_blocks
    original_teardown = connector_cls._teardown

    def __init__(self, *, vllm_config, model, kv_cache_config):
        original_init(self, vllm_config=vllm_config, model=model, kv_cache_config=kv_cache_config)
        bind_forced_routing(self, model, vllm_config)

    def _publish_blocks(self, batches, *args, **kwargs):
        if _STATE is not None and batches:
            label_published_blocks(self, batches)
        return original_publish(self, batches, *args, **kwargs)

    def _teardown(self, request_ids):
        state = _STATE
        stash = state.stash if state is not None else None
        if stash is None:
            return original_teardown(self, request_ids)
        request_ids = list(request_ids)
        stash.save(self, request_ids)
        original_teardown(self, request_ids)
        stash.forget(request_ids)

    connector_cls.__init__ = __init__
    connector_cls._publish_blocks = _publish_blocks
    connector_cls._teardown = _teardown


def _patch_runner(runner_cls) -> None:
    original_add = runner_cls.add_requests
    original_remove = runner_cls._remove_request
    original_prepare = runner_cls.prepare_inputs

    def add_requests(self, scheduler_output):
        original_add(self, scheduler_output)
        state = _STATE
        if state is None:
            return
        for new_req in scheduler_output.scheduled_new_reqs:
            sp = new_req.sampling_params
            if sp is None:
                continue
            extra = sp.extra_args
            state.add_request(
                new_req.req_id,
                new_req.prompt_len,
                sp.routed_experts_prompt_start,
                new_req.prefill_token_ids,
                extra.get(EXTRA_ARGS_KEY) if extra else None,
            )

    def _remove_request(self, req_id):
        if _STATE is not None:
            _STATE.remove_request(req_id)
        return original_remove(self, req_id)

    def prepare_inputs(self, *args, **kwargs):
        input_batch = original_prepare(self, *args, **kwargs)
        if _STATE is not None:
            batch_desc = kwargs.get("batch_desc", args[2] if len(args) > 2 else None)
            _STATE.fill(
                self.aux_output_connector,
                input_batch.req_ids,
                input_batch.query_start_loc_np,
                input_batch.num_computed_tokens_np,
                input_batch.prefill_len_np,
                batch_desc,
            )
        return input_batch

    prepare_inputs._prime_forced_routing = True
    runner_cls.add_requests = add_requests
    runner_cls._remove_request = _remove_request
    runner_cls.prepare_inputs = prepare_inputs


def _patch_async_output(output_cls) -> None:
    """Record sampled tokens on TP rank 0 so a stashed tail can be checked against the tokens it was computed for."""
    original_get_output = output_cls.get_output

    def get_output(self):
        output = original_get_output(self)
        state = _STATE
        if state is not None and state.stash is not None:
            state.stash.record(output.req_ids, output.sampled_token_ids)
        return output

    output_cls.get_output = get_output


def _patch_capture(manager_cls) -> None:
    """Capture decode-only FULL graphs without the forcing ops."""
    from vllm.config import CUDAGraphMode

    original_capture = manager_cls.capture

    def capture(self, create_forward_fn, *args, **kwargs):
        counts = {"decode_only": 0, "full": 0}

        def create(desc, warmup):
            forward_fn = create_forward_fn(desc, warmup)
            decode_only = _decode_only(desc)
            if not warmup:
                counts["decode_only"] += int(decode_only)
                counts["full"] += int(desc.cg_mode == CUDAGraphMode.FULL)

            def run(cg_mode):
                global _CAPTURING_DECODE_ONLY_GRAPH
                _CAPTURING_DECODE_ONLY_GRAPH = decode_only
                try:
                    return forward_fn(cg_mode)
                finally:
                    _CAPTURING_DECODE_ONLY_GRAPH = False

            return run

        out = original_capture(self, create, *args, **kwargs)
        if _STATE is not None and _STATE.tp_rank == 0:
            log = _logger()
            log.info(
                f"Forced routing: {counts['decode_only']} decode-only CUDA graphs captured without forcing ops "
                f"({counts['full']} FULL graphs in all)"
            )
            if counts["full"] and not counts["decode_only"]:
                log.warning(
                    "Forced routing: FULL CUDA graphs were captured but none is decode-only, so every decode step "
                    "replays the forcing ops"
                )
        return out

    manager_cls.capture = capture


def _patch_engine_core(engine_core_cls) -> None:
    """Send every weight-version update (all update paths end in EngineCore.set_weight_version) to the workers."""
    original_set_weight_version = engine_core_cls.set_weight_version

    def set_weight_version(self, weight_version):
        original_set_weight_version(self, weight_version)
        self.collective_rpc(RPC_METHOD, args=("set_weight_version", parse_weight_version(weight_version)))

    engine_core_cls.set_weight_version = set_weight_version


def monkey_patch_forced_routing() -> None:
    if not enabled():
        return  # stock vLLM: no class attribute is touched
    from vllm.distributed.aux_output_connector.store import BlockObjectStore
    from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(GPUModelRunner.prepare_inputs, "_prime_forced_routing", False):
        return
    _patch_store(BlockObjectStore)  # main's 2x store subclass inherits put
    _patch_connector(AuxOutputWorkerConnector)
    _patch_runner(GPUModelRunner)
    _patch_async_output(AsyncOutput)
    _patch_capture(CudaGraphManager)
    _patch_engine_core(EngineCore)
    Worker.prime_rl_forced_routing = _worker_rpc
    _logger().info("Forced routing patches installed")
