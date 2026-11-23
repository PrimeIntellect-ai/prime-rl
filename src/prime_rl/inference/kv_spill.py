"""Spill of expiring decode KV to a Mooncake store, for bidirectional P/D (NIXL pull mode).

A decode engine keeps a finished turn's KV blocks for a lease so the next turn's prefill can read them back over
NIXL. Idle sessions therefore pin decode's KV cache. With ``deployment.bidirectional_kv_transfer.spill_to_store``:

- decode: shortly before a lease runs out unread, the NIXL worker writes the turn's exported blocks to the store
  from a background thread. The lease is held until the put has finished, then expires as usual.
- prefill: a turn whose handle has expired fetches the blocks it lacks from the store into its own blocks, in place
  of the NIXL read, and reports the load like a finished readback. Anything missing fails the load, and the
  ``recompute`` load-failure policy applies as it does for a declined read.

The objects hold the same bytes a readback copies and nothing runs per engine step. A full block of a
full-attention group is keyed by its content hash, so a session's unchanged prefix is written once and later
spills add only the new blocks; partial last blocks and sliding-window blocks are keyed by request id and position,
and deleted by the fetch that reads them. The store handle comes from a ``MooncakeStoreConnector`` in capacity-only
mode next to the NIXL connector: vLLM's own store lookups, loads and saves stay off.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from vllm.logger import init_logger

logger = init_logger("vllm.prime_rl.kv_spill")  # under the vllm logger, so INFO lines reach the engine log

LEAD_KEY = "prime_rl_spill_lead_s"
WAIT_KEY = "prime_rl_spill_fetch_wait_s"


def spill_settings(connector) -> tuple[float, float] | None:
    """(seconds before lease expiry at which a put starts, seconds a fetch waits for a put in flight) of a NIXL
    scheduler or worker, or None when the spill is not configured for it."""
    config = connector.vllm_config.kv_transfer_config
    lead_s = config.get_from_extra_config(LEAD_KEY, None)
    if lead_s is None:
        return None
    return float(lead_s), float(config.get_from_extra_config(WAIT_KEY, 6.0))


class _State:
    def __init__(self, store, replicate_config, lead_s: float, wait_s: float):
        self.store = store
        self.lead_s = lead_s
        self.wait_s = wait_s
        self.replicate_config = replicate_config
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kv-spill")
        # decode: request id -> exported block ids per transfer group
        self.held: dict[str, Any] = {}
        # decode: request id -> (put, lease deadline)
        self.putting: dict[str, tuple[Future, float]] = {}
        self.spilled: set[str] = set()
        # prefill: request id -> fetch
        self.fetches: dict[str, Future] = {}
        self.calls = 0


def _state(worker) -> _State | None:
    """The spill state of a NIXL worker, created on first use; None when it cannot spill."""
    state = getattr(worker, "_prime_rl_spill", False)
    if state is not False:
        return state
    settings = spill_settings(worker)
    worker._prime_rl_spill = state = _make_state(worker, *settings) if settings is not None else None
    return state


def _make_state(worker, lead_s: float, wait_s: float) -> _State | None:
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import MooncakeStoreWorker

    if not has_kv_transfer_group():
        return None
    group = get_kv_transfer_group()
    store_workers = [
        connector.connector_worker
        for connector in getattr(group, "sub_connectors", None) or [group]
        if isinstance(getattr(connector, "connector_worker", None), MooncakeStoreWorker)
    ]
    if not store_workers:
        logger.warning("KV spill: no Mooncake store worker in this process, spill is off")
        return None
    if worker._has_mamba or "DRAM" in worker.region_mem_types or worker.use_host_buffer:
        logger.warning("KV spill: unsupported KV layout, spill is off")
        return None
    store_worker = store_workers[0]
    bases = worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank]
    for region, base in enumerate(bases):
        length = (worker.region_num_blocks[region] - 1) * worker.block_stride_per_layer[
            region
        ] + worker.block_len_per_layer[region]
        if base in store_worker._device_regions:
            continue
        ret = store_worker.store.register_buffer(base, length)
        if ret != 0:
            logger.warning("KV spill: register_buffer failed (%s) for region %d, spill is off", ret, region)
            return None
    logger.info("KV spill: ready on rank %s with %d regions", worker.tp_rank, len(bases))
    return _State(store_worker.store, store_worker.store_replicate_config, lead_s, wait_s)


def handle_keys(scheduler, request_id: str, block_hashes, num_tokens: int, group_lens) -> tuple[list[str], ...]:
    """Store keys for the exported blocks of a held turn, per transfer group. Decode (at finish) and prefill (for the
    handle it was given) compute the same keys from the same tokens."""
    config = scheduler.kv_cache_config
    hash_block_size = config.hash_block_size
    sliding = getattr(scheduler, "blocks_per_sw", None)
    keys = []
    for group, num_blocks in enumerate(group_lens):
        block_size = config.transfer_groups[group].kv_cache_spec.block_size
        # Content keys need the list to start at token 0: not clipped to a window.
        whole = (
            not (sliding and sliding[group])
            and block_size % hash_block_size == 0
            and num_blocks == -(-num_tokens // block_size)
        )
        group_keys = []
        for index in range(num_blocks):
            end = (index + 1) * block_size
            hash_index = end // hash_block_size - 1
            if whole and end <= num_tokens and hash_index < len(block_hashes):
                group_keys.append(f"spill-kv@g{group}@{bytes(block_hashes[hash_index]).hex()}")
            else:
                group_keys.append(f"spill-req@{request_id}@g{group}@{index}")
        keys.append(group_keys)
    return tuple(keys)


def block_buffers(worker, group: int, block_id: int) -> tuple[list[int], list[int]]:
    """Addresses and sizes of one logical block of a transfer group on this worker."""
    ratio = worker._physical_blocks_per_logical_kv_block
    bases = worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank]
    addrs, sizes = [], []
    for region in worker._group_region_indices[group] or ():
        for kernel_block in range(block_id * ratio, (block_id + 1) * ratio):
            addrs.append(bases[region] + kernel_block * worker.block_stride_per_layer[region])
            sizes.append(worker.block_len_per_layer[region])
    return addrs, sizes


def spill_plan(worker, held) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Decode: every exported block of a held turn. ``held`` is (block ids, keys) per transfer group."""
    block_ids, group_keys = held
    keys, addrs, sizes = [], [], []
    for group, blocks in enumerate(block_ids):
        for index, block_id in enumerate(blocks):
            block_addrs, block_sizes = block_buffers(worker, group, block_id)
            if not block_addrs:
                continue
            keys.append(group_keys[group][index])
            addrs.append(block_addrs)
            sizes.append(block_sizes)
    return keys, addrs, sizes


def fetch_plan(worker, meta, group_keys) -> tuple[list[str], list[list[int]], list[list[int]]] | None:
    """Prefill: the blocks a readback would copy. Local blocks are the uncomputed suffix of each group and pair with
    the end of decode's list, as in the NIXL read. None when the lists cannot be paired."""
    keys, addrs, sizes = [], [], []
    for group, (local_blocks, remote_blocks) in enumerate(
        zip(meta.local_block_ids, meta.remote.block_ids, strict=True)
    ):
        if len(local_blocks) > len(remote_blocks) or len(group_keys[group]) != len(remote_blocks):
            return None
        first = len(remote_blocks) - len(local_blocks)
        for offset, block_id in enumerate(local_blocks):
            block_addrs, block_sizes = block_buffers(worker, group, block_id)
            if not block_addrs:
                continue
            keys.append(group_keys[group][first + offset])
            addrs.append(block_addrs)
            sizes.append(block_sizes)
    return (keys, addrs, sizes) if keys else None


def _put(state: _State, request_id: str, plan) -> bool:
    keys, addrs, sizes = plan
    start = time.perf_counter()
    total = len(keys)
    try:
        # Blocks an earlier spill of this session (or a shared prefix) already wrote.
        missing = [i for i, exists in enumerate(state.store.batch_is_exist(keys)) if exists != 1]
        keys, addrs, sizes = ([column[i] for i in missing] for column in (keys, addrs, sizes))
        results = state.store.batch_put_from_multi_buffers(keys, addrs, sizes, state.replicate_config) if keys else []
    except Exception:
        logger.exception("KV spill: put failed for %s", request_id)
        return False
    failed = sum(1 for result in results if result < 0)
    logger.info(
        "KV spill: put req=%s blocks=%d written=%d failed=%d bytes=%d ms=%.0f",
        request_id,
        total,
        len(keys),
        failed,
        sum(map(sum, sizes)),
        (time.perf_counter() - start) * 1000,
    )
    return failed == 0


def _remove(state: _State, keys: list[str]) -> None:
    """Delete the objects keyed by one request: used once, or possibly torn. Content-keyed blocks stay."""
    keys = [key for key in keys if key.startswith("spill-req@")]
    if not keys:
        return
    try:
        # force: the fetch that just read them, or the put that wrote them, left a lease on them.
        failed = sum(1 for result in state.store.batch_remove(keys, True) if result != 0)
    except Exception:
        logger.exception("KV spill: remove failed")
        return
    if failed:
        logger.info("KV spill: remove keys=%d failed=%d", len(keys), failed)


def _fetch(state: _State, request_id: str, plan) -> bool:
    keys, addrs, sizes = plan
    start = time.perf_counter()
    try:
        while True:
            present = sum(1 for exists in state.store.batch_is_exist(keys) if exists == 1)
            if present == len(keys):
                break
            if time.perf_counter() - start > state.wait_s:
                logger.info("KV spill: fetch miss req=%s keys=%d present=%d", request_id, len(keys), present)
                return False
            time.sleep(0.2)
        waited = time.perf_counter() - start
        results = state.store.batch_get_into_multi_buffers(keys, addrs, sizes)
    except Exception:
        logger.exception("KV spill: fetch failed for %s", request_id)
        return False
    failed = sum(1 for result in results if result < 0)
    logger.info(
        "KV spill: fetch req=%s keys=%d failed=%d bytes=%d wait_ms=%.0f ms=%.0f",
        request_id,
        len(keys),
        failed,
        sum(map(sum, sizes)),
        waited * 1000,
        (time.perf_counter() - start) * 1000,
    )
    if failed == 0:
        # A handle is read at most once: its tail and window objects are not needed again.
        _remove(state, keys)
    return failed == 0


def start_due_spills(worker, state: _State, now: float) -> None:
    """Decode, once per worker step: start puts for leases about to end, and release the leases of finished puts."""
    leases = worker._reqs_to_send
    for request_id, (put, deadline) in list(state.putting.items()):
        if not put.done():
            continue
        del state.putting[request_id]
        if request_id in leases:
            # Unread so far: the lease ends at its own deadline.
            leases[request_id] = deadline
            state.spilled.add(request_id)
        else:
            # Read back while the put ran: its blocks may have been reused under the put.
            held = state.held.get(request_id)
            if held is not None:
                state.pool.submit(_remove, state, spill_plan(worker, held)[0])
    for request_id, deadline in list(leases.items()):
        if now < deadline - state.lead_s or request_id in state.putting or request_id in state.spilled:
            continue
        held = state.held.get(request_id)
        if held is None:
            continue
        plan = spill_plan(worker, held)
        if not plan[0]:
            state.spilled.add(request_id)
            continue
        state.putting[request_id] = (state.pool.submit(_put, state, request_id, plan), deadline)
        # Not reaped while the put reads its blocks.
        leases[request_id] = float("inf")
    state.calls += 1
    if state.calls % 256 == 0:
        for request_id in [r for r in state.held if r not in leases and r not in state.putting]:
            del state.held[request_id]
        state.spilled.intersection_update(leases)


def monkey_patch_nixl_spill_expiring_kv_to_store():
    """Install the spill on the NIXL pull connector. It stays inert on engines whose NIXL connector config carries
    no ``prime_rl_spill_lead_s`` (set from ``deployment.bidirectional_kv_transfer.spill_to_store``)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import NixlBaseConnectorScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker

    if getattr(NixlPullConnectorWorker.start_load_kv, "_prime_rl_spills_kv", False):
        return
    original_finished = NixlPullConnectorScheduler.request_finished
    original_build_meta = NixlBaseConnectorScheduler.build_connector_meta
    original_alloc = NixlPullConnectorScheduler.update_state_after_alloc
    original_start_load = NixlPullConnectorWorker.start_load_kv
    original_reap = NixlBaseConnectorWorker._reap_expired_send_leases
    original_read = NixlPullConnectorWorker._read_blocks_for_req
    original_results = NixlBaseConnectorWorker.get_transfer_results

    def request_finished(self, request, block_ids):
        delay_free, params = original_finished(self, request, block_ids)
        if (
            delay_free
            and params
            and params.get("remote_blocks_expiry_time") is not None
            and spill_settings(self) is not None
        ):
            block_ids = params["remote_block_ids"]
            keys = handle_keys(
                self,
                request.request_id,
                request.block_hashes,
                params["remote_num_tokens"],
                [len(group) for group in block_ids],
            )
            self.__dict__.setdefault("_prime_rl_spill_blocks", {})[request.request_id] = (block_ids, keys)
        return delay_free, params

    def build_connector_meta(self, scheduler_output):
        meta = original_build_meta(self, scheduler_output)
        held = self.__dict__.get("_prime_rl_spill_blocks")
        if held:
            meta.prime_rl_spill_blocks = held
            self._prime_rl_spill_blocks = {}
        fetch_keys = self.__dict__.get("_prime_rl_spill_fetch_keys")
        if fetch_keys:
            meta.prime_rl_spill_fetch_keys = fetch_keys
            self._prime_rl_spill_fetch_keys = {}
        return meta

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params or {}
        handle = (
            (params.get("remote_request_id"), params.get("remote_num_tokens"), params.get("remote_block_ids"))
            if params.get("do_remote_decode")
            and params.get("remote_blocks_expiry_time") is not None
            and spill_settings(self) is not None
            else None
        )
        original_alloc(self, request, blocks, num_external_tokens)
        if (
            handle is not None
            and all(part is not None for part in handle)
            and request.request_id in self._reqs_need_recv
        ):
            remote_request_id, num_tokens, block_ids = handle
            self.__dict__.setdefault("_prime_rl_spill_fetch_keys", {})[request.request_id] = handle_keys(
                self, remote_request_id, request.block_hashes, num_tokens, [len(group) for group in block_ids]
            )

    def start_load_kv(self, metadata):
        held = getattr(metadata, "prime_rl_spill_blocks", None)
        if held:
            state = _state(self)
            if state is not None:
                state.held.update(held)
        fetch_keys = getattr(metadata, "prime_rl_spill_fetch_keys", None)
        if fetch_keys:
            self.__dict__.setdefault("_prime_rl_spill_fetch_keys", {}).update(fetch_keys)
        return original_start_load(self, metadata)

    def _reap_expired_send_leases(self, done_sending):
        state = getattr(self, "_prime_rl_spill", None)
        if state:
            start_due_spills(self, state, time.perf_counter())
        return original_reap(self, done_sending)

    def _read_blocks_for_req(self, req_id, meta):
        remote = meta.remote
        if remote.engine_id in self._invalid_remote_engines or remote.engine_id not in self._remote_agents:
            # Fails, or comes back here once the handshake is done.
            return original_read(self, req_id, meta)
        group_keys = self.__dict__.get("_prime_rl_spill_fetch_keys", {}).pop(req_id, None)
        if self._bidirectional_kv_xfer_enabled and self._is_turn2_read_expired(meta):
            state = _state(self)
            if state is not None and req_id in state.fetches:
                return
            plan = fetch_plan(self, meta, group_keys) if state is not None and group_keys is not None else None
            if plan is not None:
                state.fetches[req_id] = state.pool.submit(_fetch, state, req_id, plan)
                return
        return original_read(self, req_id, meta)

    def get_transfer_results(self):
        state = getattr(self, "_prime_rl_spill", None)
        if state and state.fetches:
            for req_id, fetch in list(state.fetches.items()):
                if not fetch.done():
                    continue
                del state.fetches[req_id]
                if req_id not in self._recving_metadata:
                    continue
                if fetch.result():
                    # No NIXL transfer outstanding: reported as received by the original below.
                    self._recving_transfers[req_id] = []
                else:
                    self._handle_failed_transfer(req_id, None, self._recv_failures, record_failed_transfer=False)
        return original_results(self)

    start_load_kv._prime_rl_spills_kv = True
    NixlPullConnectorScheduler.request_finished = request_finished
    NixlBaseConnectorScheduler.build_connector_meta = build_connector_meta
    NixlPullConnectorScheduler.update_state_after_alloc = update_state_after_alloc
    NixlPullConnectorWorker.start_load_kv = start_load_kv
    NixlBaseConnectorWorker._reap_expired_send_leases = _reap_expired_send_leases
    NixlPullConnectorWorker._read_blocks_for_req = _read_blocks_for_req
    NixlBaseConnectorWorker.get_transfer_results = get_transfer_results
