import math
import os
from dataclasses import dataclass, replace

import numpy as np
import torch


def apply_shared_vllm_patches():
    """vLLM general plugin and the single place prime-rl applies its vLLM patches; vLLM runs it once in every process.

    vLLM swallows plugin load failures (``load_plugins_by_group`` logs and continues), so a broken
    entry-point target silently skips ALL of these patches.
    """
    from prime_rl.inference.vllm.gpt_oss_weight_loading import patch_gpt_oss_weight_loading
    from prime_rl.inference.vllm.qwen38_weight_loading import patch_qwen38_weight_loading

    patch_gpt_oss_weight_loading()
    patch_qwen38_weight_loading()
    _patch_lora_key_prefix()
    _patch_qwen35_moe_lora_format()
    monkey_patch_minimax_m2_think_end_passthrough()
    monkey_patch_routed_experts_with_pd_connectors()
    monkey_patch_clamp_kv_offload_loads_for_routed_experts()
    monkey_patch_routed_experts_cut_loaded_prefix_hits()
    monkey_patch_kv_xfer_finished_tolerate_freed()
    monkey_patch_strip_routed_experts_from_chat()
    monkey_patch_skip_prompt_routing_for_chat()
    monkey_patch_dp_coordinator_startup_timeout()
    monkey_patch_minimax_m2_for_lora()
    monkey_patch_packed_sampling_masks()
    monkey_patch_engine_gc()
    monkey_patch_packed_aux_outputs()
    monkey_patch_aux_output_store()
    from prime_rl.inference.vllm.request_replay import patch_request_replay

    patch_request_replay()
    monkey_patch_block_outermost_mla_stride()
    monkey_patch_fit_sliding_window_pages()
    monkey_patch_flashinfer_fp8_ds_mla_draft()
    monkey_patch_dspark_dp_compile_cache()
    monkey_patch_nixl_bidirectional_single_pull()
    monkey_patch_nixl_sliding_window_backoff()
    monkey_patch_nixl_bidirectional_spec_decode_coverage()
    monkey_patch_keep_pause_ignores_held_kv()
    monkey_patch_freeze_nixl_leases_during_keep_pause()
    monkey_patch_nixl_prefill_lapsed_remote_kv_locally()
    monkey_patch_drop_reservation_of_failed_kv_load()
    monkey_patch_skip_store_lookup_after_match()
    monkey_patch_release_failed_kv_loads()
    from prime_rl.inference.kv_spill import monkey_patch_nixl_spill_expiring_kv_to_store

    # After the spec-decode coverage patch: the spill stores the blocks that patch exports.
    monkey_patch_nixl_spill_expiring_kv_to_store()
    # Set by `server()` when the LoRA target modules include no expert layers.
    if os.environ.get("PRIME_NO_MOE_LORA") == "1":
        monkey_patch_no_moe_lora()


def monkey_patch_nixl_bidirectional_single_pull():
    """Bidirectional NIXL: a prefill request that comes back through the connector lookup after its D->P pull
    prefills locally instead of waiting for a second read.

    With ``bidirectional_kv_xfer``, the next turn's prefill request carries decode's block ids, and the first lookup
    registers the D->P read and marks them ``_remote_blocks_processed``. If the read fails under the ``recompute``
    load-failure policy (e.g. prefill declines it because decode's blocks are close to their TTL), or the request is
    preempted after the pull and its blocks are evicted, the request goes through the lookup again. That lookup
    still offered decode's tokens, so the request waited in ``WAITING_FOR_REMOTE_KVS`` for a read that is never
    issued, holding its blocks: under load the prefill instance stalls. Backport of vllm-project/vllm#64860.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_scheduler

    cls = pull_scheduler.NixlPullConnectorScheduler
    original = cls.get_num_new_matched_tokens
    if getattr(original, "_prime_rl_single_pull", False):
        return

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("_remote_blocks_processed")
            and not params.get("do_remote_prefill")
        ):
            return 0, False
        return original(self, request, num_computed_tokens)

    get_num_new_matched_tokens._prime_rl_single_pull = True
    cls.get_num_new_matched_tokens = get_num_new_matched_tokens


def monkey_patch_nixl_sliding_window_backoff():
    """P/D with sliding-window layers: leave the last prompt token to decode, as for Mamba.

    Decode recomputes the last prompt token to sample the first output token. If prefill transfers the whole prompt,
    the sliding-window group is laid out for the next position, which drops the oldest block of the window whenever
    ``N - W + 1`` is a multiple of the block size: the recomputed token then attends to the null block. Prefill
    therefore stops one token short and decode recomputes it with its window intact (DeepSeek V4's compressor
    caches are sliding-window specs). Backport of vllm-project/vllm#65584 (NIXL side).
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_scheduler
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    cls = base_scheduler.NixlBaseConnectorScheduler
    original = cls._prefill_backoff
    if getattr(original, "_prime_rl_sliding_window", False):
        return

    def _prefill_backoff(self) -> int:
        has_sliding_window = getattr(self, "_prime_rl_has_sliding_window", None)
        if has_sliding_window is None:
            has_sliding_window = any(
                isinstance(group.kv_cache_spec, SlidingWindowSpec) for group in self.kv_cache_config.kv_cache_groups
            )
            self._prime_rl_has_sliding_window = has_sliding_window
        return max(original(self), 1 if has_sliding_window else 0)

    _prefill_backoff._prime_rl_sliding_window = True
    cls._prefill_backoff = _prefill_backoff


def trim_block_ids_to_tokens(block_ids, kv_cache_groups, num_tokens: int):
    """Keep, per KV cache group, only the blocks that hold positions ``[0, num_tokens)``.

    Block lists are positional from token 0 for every attention group (sliding-window groups pad out-of-window
    positions with null blocks), so the blocks to keep are the first ``cdiv(num_tokens, block_size)``. SSM groups hold
    one running state, not positions, and stay as they are.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    trimmed = []
    for group, blocks in zip(kv_cache_groups, block_ids, strict=True):
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            trimmed.append(blocks)
            continue
        keep = -(-num_tokens // spec.block_size)
        trimmed.append(list(blocks)[:keep])
    return tuple(trimmed)


def monkey_patch_nixl_bidirectional_spec_decode_coverage():
    """Bidirectional NIXL with speculative decoding: decode's handle covers only tokens with valid KV.

    When a request finishes, decode exports its retained blocks for the next turn's prefill with
    ``remote_num_tokens = request.num_computed_tokens``. With speculative decoding, the last step also computed the
    drafted positions after the final token; the request ends before a rejection rolls them back, so the count
    includes positions holding rejected drafts' KV (e.g. 42 for 23 prompt + 16 output tokens with 3 drafts). Only
    ``num_tokens - 1`` positions hold KV of the actual sequence: every token but the last was an input. Prefill would
    load the stale draft KV, and a router that checks coverage against the tokens it has seen drops every such
    handle, so the readback never happens. The extra positions can also add a trailing block, and prefill pairs block
    lists from their ends, which would shift every block.

    Report ``num_tokens - 1`` and drop the blocks past it, per KV cache group, before vLLM clips sliding-window
    groups to their in-window tail (clipping a list with an extra trailing block would cut a block the window needs).
    Without speculative decoding, ``num_computed_tokens`` is already ``num_tokens - 1`` and nothing changes.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_scheduler
    from vllm.v1.request import RequestStatus

    cls = pull_scheduler.NixlPullConnectorScheduler
    original = cls.request_finished
    if getattr(original, "_prime_rl_spec_decode_coverage", False):
        return

    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params
        valid = request.num_tokens - 1
        if (
            getattr(self, "is_bidirectional_kv_xfer_enabled", False)
            and params
            and not params.get("do_remote_decode")
            and request.status in (RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_STOPPED)
            and 0 < valid < request.num_computed_tokens
        ):
            block_ids = trim_block_ids_to_tokens(block_ids, self.kv_cache_config.kv_cache_groups, valid)
            computed = request.num_computed_tokens
            request.num_computed_tokens = valid
            try:
                return original(self, request, block_ids)
            finally:
                request.num_computed_tokens = computed
        return original(self, request, block_ids)

    request_finished._prime_rl_spec_decode_coverage = True
    cls.request_finished = request_finished


def monkey_patch_keep_pause_ignores_held_kv():
    """A ``keep`` pause completes without waiting for KV blocks held for another instance.

    A paused engine reports itself idle, which completes a ``keep`` pause, once the scheduler has no requests. Under
    P/D, the scheduler also counts finished requests whose blocks it holds for a remote reader: prefill holds each
    finished prefill until decode pulls it, and decode's lease heartbeats keep that hold alive while the request waits
    in decode's queue; with bidirectional transfer, decode holds each finished turn for up to its TTL so the next
    turn's prefill can read it. Pausing for a weight update therefore waits for decode to admit its whole queue and
    for every held turn to be read or expire, which takes many minutes under load.

    While the scheduler is paused with ``PAUSED_ALL``, count only requests the next step must still report as
    finished. The held blocks stay allocated, a weight update does not touch them, remote reads of them proceed without
    this engine stepping, and the engine processes their completion or expiry once it resumes. The paused peer sends no
    lease heartbeats, so ``monkey_patch_freeze_nixl_leases_during_keep_pause`` keeps those leases from expiring meanwhile.
    """
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.has_finished_requests
    if getattr(original, "_prime_rl_keep_pause", False):
        return

    def has_finished_requests(self) -> bool:
        if self.pause_state == PauseState.PAUSED_ALL:
            return bool(self.finished_req_ids)
        return original(self)

    has_finished_requests._prime_rl_keep_pause = True
    Scheduler.has_finished_requests = has_finished_requests


def freeze_send_leases(nixl_worker) -> None:
    """Record the time each unexpired NIXL send lease has left; expired leases are not recorded, so not revived."""
    import time

    now = time.perf_counter()
    nixl_worker._prime_rl_frozen_leases = {
        req_id: expires - now for req_id, expires in nixl_worker._reqs_to_send.items() if expires > now
    }


def thaw_send_leases(nixl_worker) -> None:
    """Give each lease recorded by ``freeze_send_leases`` that is still outstanding its recorded remaining time from
    now, keeping a later deadline set by a heartbeat. Leases granted since the freeze stay as they are."""
    import time

    now = time.perf_counter()
    frozen = getattr(nixl_worker, "_prime_rl_frozen_leases", None) or {}
    for req_id, remaining in frozen.items():
        if req_id in nixl_worker._reqs_to_send:
            nixl_worker._reqs_to_send[req_id] = max(nixl_worker._reqs_to_send[req_id], now + remaining)
    nixl_worker._prime_rl_frozen_leases = None


def _nixl_workers() -> list:
    """The NIXL connector workers in this worker process, also inside a MultiConnector."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker

    if not has_kv_transfer_group():
        return []
    group = get_kv_transfer_group()
    connectors = getattr(group, "sub_connectors", None) or [group]
    workers = [getattr(connector, "connector_worker", None) for connector in connectors]
    return [worker for worker in workers if isinstance(worker, NixlBaseConnectorWorker)]


def _freeze_nixl_send_leases(_worker) -> None:
    for nixl_worker in _nixl_workers():
        freeze_send_leases(nixl_worker)


def _thaw_nixl_send_leases(_worker) -> None:
    for nixl_worker in _nixl_workers():
        thaw_send_leases(nixl_worker)


def monkey_patch_freeze_nixl_leases_during_keep_pause():
    """NIXL send leases keep their remaining time across a ``keep`` pause.

    A worker holds a finished request's blocks for a remote reader until the reader reports the read or the lease
    expires; the reader extends the lease with heartbeats it sends as it steps. While both engines are paused for a
    weight update, no heartbeats are sent or processed, and a lease that runs on wall-clock time would expire: on
    resume the holder would release and reuse blocks the reader still reads.

    A ``keep`` pause records each unexpired lease's remaining time in the engine's workers, and resume restores it from
    the resume time before the next step reaps expired leases. Outside of a ``keep`` pause, vLLM's wall-clock lease
    reaper is unchanged, so leases run down during slow steps as they do without this patch.
    """
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    for cls in (EngineCore, EngineCoreProc):
        original_pause = cls.__dict__["pause_scheduler"]
        if getattr(original_pause, "_prime_rl_freeze_leases", False):
            continue

        def pause_scheduler(self, mode="abort", clear_cache=True, *args, _original=original_pause, **kwargs):
            result = _original(self, mode, clear_cache, *args, **kwargs)
            if mode == "keep" and self.vllm_config.kv_transfer_config is not None:
                self.collective_rpc(_freeze_nixl_send_leases)
                self._prime_rl_leases_frozen = True
            return result

        pause_scheduler._prime_rl_freeze_leases = True
        cls.pause_scheduler = pause_scheduler

    original_resume = EngineCore.resume_scheduler
    if getattr(original_resume, "_prime_rl_freeze_leases", False):
        return

    def resume_scheduler(self) -> None:
        original_resume(self)
        if getattr(self, "_prime_rl_leases_frozen", False):
            self.collective_rpc(_thaw_nixl_send_leases)
            self._prime_rl_leases_frozen = False

    resume_scheduler._prime_rl_freeze_leases = True
    EngineCore.resume_scheduler = resume_scheduler


def monkey_patch_nixl_prefill_lapsed_remote_kv_locally():
    """A decode engine prefills a request itself when the blocks its prefill engine held for it may be gone.

    The prefill engine holds a finished prefill's blocks under a lease, which decode renews with heartbeats it sends as
    it steps: every ``kv_lease_duration / 6`` s, each extending the lease to ``2/3 * kv_lease_duration`` s from then. A
    decode engine that does not step renews nothing, e.g. one paused for a weight update while the prefill engines go on
    serving (engines that pull their weights switch per DP group). Once a lease runs out, prefill frees the blocks and
    reuses them, and a decode engine that reads them afterwards decodes from another request's KV.

    So when decode is about to start a request's read, and the lease was last renewed (by a heartbeat, or when the
    request arrived) more than half the lease duration ago, decode drops the remote prefill and prefills the prompt
    itself. The prefill engine releases the blocks when the lease runs out.
    """
    import time

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import NixlBaseConnectorScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
    from vllm.logger import init_logger

    original_matched = NixlPullConnectorScheduler.get_num_new_matched_tokens
    if getattr(original_matched, "_prime_rl_lapsed_lease", False):
        return
    original_new_request = NixlBaseConnectorScheduler.on_new_request
    logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler")

    def on_new_request(self, request) -> None:
        original_new_request(self, request)
        request._prime_rl_kv_lease_renewed = time.perf_counter()

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            renewed = max(getattr(request, "_prime_rl_kv_lease_renewed", 0.0), self._last_heartbeat_time)
            waited = time.perf_counter() - renewed
            if waited > self._kv_lease_duration / 2:
                logger.warning(
                    f"Prefilling {request.request_id} locally: its KV lease on the prefill engine was last renewed "
                    f"{waited:.0f}s ago, so the prefill engine may have released the blocks"
                )
                params["do_remote_prefill"] = False
                self._stop_heartbeat(request.request_id)
        return original_matched(self, request, num_computed_tokens)

    get_num_new_matched_tokens._prime_rl_lapsed_lease = True
    NixlBaseConnectorScheduler.on_new_request = on_new_request
    NixlPullConnectorScheduler.get_num_new_matched_tokens = get_num_new_matched_tokens


def monkey_patch_release_failed_kv_loads():
    """A request whose asynchronous KV load failed outright gives its blocks back as soon as the failure is known.

    The scheduler only acts on a finished load when its waiting loop reaches the request, and the loop stops at the
    first request it cannot allocate blocks for. A request whose load failed (a declined or expired read, a store
    miss) still holds the blocks it allocated for that load, and needs its whole sequence recomputed. With enough
    of them, the request at the head cannot allocate, the loop never reaches the ones behind it, and their blocks
    are never released: the engine idles with a full KV cache, nothing running and no error.

    When the failure arrives, release the request's blocks right away and move it from the queue of block-holding
    requests to the front of the ordinary waiting queue. Requests whose loads succeeded then stay first in line,
    and they only need the few blocks they reserved.
    """
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    original = Scheduler._update_from_kv_xfer_finished
    if getattr(original, "_prime_rl_releases_failed_loads", False):
        return

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        original(self, kv_connector_output)
        failed = self.failed_recving_kv_req_ids & self.finished_recving_kv_req_ids
        if not failed:
            return
        released = []
        for request in list(self.kv_holding_waiting):
            # A load that failed outright has no computed tokens: `_handle_failed_recving` zeroed them earlier in
            # this update. One that kept part of its prefix keeps those blocks, and is left to the waiting loop.
            if (
                request.request_id not in failed
                or request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
                or request.num_computed_tokens
            ):
                continue
            # Frees the blocks, drops the reservation, and makes the request WAITING (or PREEMPTED) again.
            self._handle_blocked_waiting_request(request)
            # The load is over: it no longer counts toward --max-num-inflight-kv-loads.
            self._set_kv_fetch_stage(request, None)
            released.append(request)
        if released:
            self.kv_holding_waiting.remove_requests(released)
            self.deferred_waiting.difference_update(released)
            # `prepend_requests` reverses its argument; this keeps the requests in their queue order.
            self.waiting.prepend_requests(reversed(released))

    _update_from_kv_xfer_finished._prime_rl_releases_failed_loads = True
    # Carry the marker of the patch underneath, if it is there, so that one is neither applied twice nor skipped.
    _update_from_kv_xfer_finished._prime_rl_tolerates_freed = getattr(original, "_prime_rl_tolerates_freed", False)
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished


def monkey_patch_drop_reservation_of_failed_kv_load():
    """Stop requests whose KV load failed from reserving the whole KV pool.

    The scheduler reserves, for every request parked on an asynchronous KV load, the blocks it still needs to hold
    its full sequence, and admits a new load only if it fits beside those reservations. The reservation is refreshed
    when the load finishes. A load that failed outright has freed its blocks by then, so the refresh reserves the
    request's whole sequence, and it is only released when the request runs. Enough such requests (for example
    turns whose decode blocks expired while they queued) reserve more than the pool holds: no load is admitted any
    more, nothing runs, and the engine idles with an empty KV cache and a full queue.

    A request that holds no blocks has nothing in flight to protect, so it keeps no reservation.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler._reserve_parked
    if getattr(original, "_prime_rl_drops_failed_load", False):
        return

    def _reserve_parked(self, request) -> None:
        # Both callers set the computed count first: the load's hit when it is parked, what survived when it ends.
        if request.num_computed_tokens == 0:
            self._unreserve_parked(request)
            return
        original(self, request)

    _reserve_parked._prime_rl_drops_failed_load = True
    Scheduler._reserve_parked = _reserve_parked


def has_remote_kv_handle(request) -> bool:
    """Whether the request carries another instance's KV blocks to read (a bidirectional decode handle)."""
    params = request.kv_transfer_params
    return bool(params and params.get("remote_block_ids"))


def monkey_patch_skip_store_lookup_after_match():
    """A request whose KV another connector already supplies does not also look it up in the Mooncake store.

    ``MultiConnector`` asks every connector how many tokens it can supply and uses the first that has any, but it
    keeps asking the rest. Under bidirectional P/D, a prefill request whose missing KV comes back from decode over
    NIXL still pays a store lookup (one key per KV block and group, so thousands for a long prompt) whose answer is
    discarded. A synchronous lookup blocks ``Scheduler.schedule()`` for its whole duration, and the data-parallel
    ranks of a node step together, so one rank waiting on the store stalls them all: under a backlog a prefill engine
    spent two thirds of its time there. An asynchronous lookup that is still pending makes ``MultiConnector`` report
    the request as not ready, which delays it by a step.

    After a connector has matched, the remaining store connectors are not asked, and a request that arrives with a
    remote KV handle does not start the store's lookup prefetch. Other connectors are still asked, since they may
    track requests through this call. A request without a match is looked up in the store as before.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1 import multi_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler

    cls = multi_connector.MultiConnector
    if getattr(cls.get_num_new_matched_tokens, "_prime_rl_skip_store_lookup", False):
        return

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        to_return = (0, False)
        for i, connector in enumerate(self._connectors):
            if to_return[0] > 0 and isinstance(connector, MooncakeStoreConnector):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return (None, False)
            if to_return[0] == 0 and tokens > 0:
                self._requests_to_connector[request.request_id] = i
                to_return = (tokens, load_async)
        return to_return

    get_num_new_matched_tokens._prime_rl_skip_store_lookup = True
    cls.get_num_new_matched_tokens = get_num_new_matched_tokens

    original_prefetch = MooncakeStoreScheduler.start_lookup_prefetch

    def start_lookup_prefetch(self, request) -> None:
        if has_remote_kv_handle(request):
            return
        original_prefetch(self, request)

    MooncakeStoreScheduler.start_lookup_prefetch = start_lookup_prefetch


def monkey_patch_block_outermost_mla_stride():
    """Align the shared block stride of packed KV caches to DeepSeek V4's MLA pages.

    Block-outermost groups share one block stride: the widest group's bytes per block, rounded to
    each layer's ``block_stride_alignment``. DeepSeek V4's MLA specs declare their 576 B FlashMLA
    alignment only as page padding (``alignment``), so a wider group of another kind, such as a
    draft's KV cache, leaves the stride unaligned and FlashMLA rejects it.
    """
    from vllm.utils.math_utils import round_up
    from vllm.v1.core import kv_cache_planning

    original = kv_cache_planning._get_kv_cache_bytes_per_block
    if getattr(original, "_prime_rl_aligns_mla_stride", False):
        return

    def _get_kv_cache_bytes_per_block(kv_cache_groups, layout):
        bytes_per_block = original(kv_cache_groups, layout)
        if not layout.is_block_outermost:
            return bytes_per_block
        alignments = {
            alignment
            for group in kv_cache_groups
            for name in group.layer_names
            if (alignment := getattr(kv_cache_planning._get_per_layer_spec(group, name), "alignment", None))
        }
        return round_up(bytes_per_block, math.lcm(*alignments)) if alignments else bytes_per_block

    _get_kv_cache_bytes_per_block._prime_rl_aligns_mla_stride = True
    kv_cache_planning._get_kv_cache_bytes_per_block = _get_kv_cache_bytes_per_block


def monkey_patch_fit_sliding_window_pages():
    """Keep a sliding-window KV group, such as a draft's, from widening block-outermost KV blocks.

    The packed layout sizes every physical block by the widest group's bytes per block, and every
    group draws blocks from that one pool. A sliding-window layer takes the largest kernel block its
    backend runs (64 tokens on FlashInfer) regardless of the packing, so a dense GQA draft next to
    DeepSeek V4's compressed MLA widens every block of the pool, although the draft only needs its
    window's blocks. Halving the group's block size until it fits under the widest other group
    restores the target's capacity at the cost of more window blocks per request.
    """
    from vllm.v1.core import kv_cache_planning
    from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs

    original = kv_cache_planning._get_packed_kv_cache_groups
    if getattr(original, "_prime_rl_fits_sliding_window_pages", False):
        return

    def layer_specs(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return dict(spec.kv_cache_specs)
        return {name: spec for name in group.layer_names}

    def group_bytes(group):
        return sum(spec.page_size_bytes for spec in layer_specs(group).values())

    def _get_packed_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = original(vllm_config, kv_cache_spec)
        if not groups or len(groups) == 1:
            return groups
        fitted = []
        for i, group in enumerate(groups):
            specs = layer_specs(group)
            widest_other = max(group_bytes(other) for j, other in enumerate(groups) if j != i)
            plain_swa = all(type(s) is SlidingWindowSpec and s.page_size_padded is None for s in specs.values())
            block_size = next(iter(specs.values())).block_size
            if plain_swa and group_bytes(group) > widest_other:
                # An unpadded sliding-window page is linear in the block size.
                bytes_per_token = group_bytes(group) // block_size
                while block_size > 16 and bytes_per_token * block_size > widest_other:
                    block_size //= 2
                specs = {name: replace(s, block_size=block_size) for name, s in specs.items()}
                group = replace(group, kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(specs))
            fitted.append(group)
        return fitted

    _get_packed_kv_cache_groups._prime_rl_fits_sliding_window_pages = True
    kv_cache_planning._get_packed_kv_cache_groups = _get_packed_kv_cache_groups


def monkey_patch_flashinfer_fp8_ds_mla_draft():
    """Let a draft's fp8 KV cache on FlashInfer coexist with DeepSeek V4's ``fp8_ds_mla`` cache.

    DeepSeek V4's attention writes its packed format onto the shared ``cache_config.cache_dtype``,
    and FlashInfer reads that global for a draft with its own fp8 KV cache
    (``speculative_config.kv_cache_dtype = "fp8"``). FlashInfer never serves DeepSeek V4's layers,
    so for it the format can only mean the draft's per-tensor fp8 e4m3 rows.
    """
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    original = FlashInferBackend.get_dtype_for_flashinfer
    if getattr(original, "_prime_rl_maps_fp8_ds_mla", False):
        return

    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        return original("fp8" if kv_cache_dtype == "fp8_ds_mla" else kv_cache_dtype)

    get_dtype_for_flashinfer._prime_rl_maps_fp8_ds_mla = True
    FlashInferBackend.get_dtype_for_flashinfer = staticmethod(get_dtype_for_flashinfer)


def monkey_patch_dspark_dp_compile_cache():
    """Keep independent DP replicas in distinct draft compilation caches."""
    from functools import wraps

    from vllm.v1.worker.gpu.spec_decode.dspark import utils

    original = utils._get_dspark_parallel_config
    if getattr(original, "_prime_rl_preserves_dp_cache_index", False):
        return

    @wraps(original)
    def _get_dspark_parallel_config(parallel_config, tensor_parallel_size, draft_model_config):
        draft = original(parallel_config, tensor_parallel_size, draft_model_config)
        # Config replacement reconstructs this init=False field from the local rank.
        draft.data_parallel_index = parallel_config.data_parallel_index
        return draft

    _get_dspark_parallel_config._prime_rl_preserves_dp_cache_index = True
    utils._get_dspark_parallel_config = _get_dspark_parallel_config


@dataclass
class PackedSamplingMask:
    """A finished request's sampling masks in CSR form: ``ids[sum(counts[:i]):][:counts[i]]``
    are the kept token ids of completion token ``i``."""

    ids: np.ndarray
    counts: np.ndarray


def monkey_patch_packed_sampling_masks():
    """Keep sampling masks as numpy arrays in the API server's output processor.

    vLLM keeps one ``SamplingMaskLists`` tuple per generated token while a request
    runs and turns a finished request's masks into ``list[list[int]]``: GC-tracked
    objects per token that every gen-2 pass walks. Store only the (untracked) id
    arrays, one per position, and hand ``PrimeRlServingTokens`` a ``PackedSamplingMask``
    instead. A speculative step appends one chunk covering all its accepted positions.
    Only non-streaming (``FINAL_ONLY``) outputs are packed, the form
    ``PrimeRlServingTokens.serve_tokens_full_generator`` reads. Replay requests, the only ones that keep their
    masks, are non-streaming tokens requests.
    """
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.output_processor import RequestState

    original_init = RequestState.__init__
    original_new_completion_output = RequestState._new_completion_output

    class _MaskIds(list):
        def append(self, chunk):
            if chunk.offsets is None:
                super().append(chunk.token_ids)
            else:
                self.extend(chunk.token_ids[a:b] for a, b in zip(chunk.offsets[:-1], chunk.offsets[1:]))

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.output_kind == RequestOutputKind.FINAL_ONLY:
            self.sampling_mask_chunks = _MaskIds()

    def _new_completion_output(self, token_ids, finish_reason, stop_reason):
        chunks = self.sampling_mask_chunks
        if not isinstance(chunks, _MaskIds) or finish_reason is None or not chunks:
            return original_new_completion_output(self, token_ids, finish_reason, stop_reason)
        self.sampling_mask_chunks = _MaskIds()
        output = original_new_completion_output(self, token_ids, finish_reason, stop_reason)
        output.sampling_mask = PackedSamplingMask(
            ids=np.concatenate(chunks),
            counts=np.fromiter(map(len, chunks), dtype=np.int32, count=len(chunks)),
        )
        return output

    RequestState.__init__ = _init
    RequestState._new_completion_output = _new_completion_output


ENGINE_GC_THRESHOLDS = (50_000, 20, 100)


def monkey_patch_engine_gc():
    """Freeze the startup heap and raise GC thresholds in EngineCore (vLLM only freezes) and the GPU workers."""
    import gc

    from vllm.v1.engine import core
    from vllm.v1.worker.gpu_worker import Worker

    original_freeze_gc_heap = core.freeze_gc_heap
    original_compile_or_warm_up_model = Worker.compile_or_warm_up_model

    def freeze_gc_heap():
        original_freeze_gc_heap()
        gc.set_threshold(*ENGINE_GC_THRESHOLDS)

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original_compile_or_warm_up_model(self, *args, **kwargs)
        gc.collect()
        gc.freeze()
        gc.set_threshold(*ENGINE_GC_THRESHOLDS)
        return result

    core.freeze_gc_heap = freeze_gc_heap
    Worker.compile_or_warm_up_model = compile_or_warm_up_model


def monkey_patch_packed_aux_outputs():
    """Send each step's routed experts from the output worker to EngineCore as one ``PackedAuxOutputs``."""
    from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector

    from prime_rl.inference.vllm.routed_experts import PackedAuxOutputs

    original_process_output = AuxOutputWorkerConnector.process_output

    def process_output(self, pending):
        outputs = original_process_output(self, pending)
        return PackedAuxOutputs.pack(outputs) if outputs else outputs

    AuxOutputWorkerConnector.process_output = process_output


AUX_OUTPUT_STORE_HEADROOM = 3


def monkey_patch_aux_output_store():
    """Size the routed-experts store above the KV cache.

    Referenced rows are never evicted. With a KV connector the scheduler retains rows for up to twice the KV
    cache's blocks (vllm-project/vllm#62591), and running requests reference up to one more KV cache of blocks
    they have not finished, so a smaller store grows by copying its whole arena on the engine path. The worker
    builds it at ``AUX_OUTPUT_STORE_HEADROOM`` times vLLM's size instead (anonymous mmap: unused slots cost no
    memory).
    """
    from vllm.distributed.aux_output_connector import worker as aux_worker
    from vllm.distributed.aux_output_connector.store import BlockObjectStore

    class _RoomyBlockObjectStore(BlockObjectStore):
        def __init__(self, *, max_bytes: int, object_nbytes: int):
            super().__init__(max_bytes=AUX_OUTPUT_STORE_HEADROOM * max_bytes, object_nbytes=object_nbytes)

    aux_worker.BlockObjectStore = _RoomyBlockObjectStore


def monkey_patch_kv_xfer_finished_tolerate_freed():
    """Tolerate KV-transfer finish notifications for already-freed requests.

    In disaggregated P/D (NIXL, optionally + a KV store connector) a request can
    be finished — most often ``FINISHED_ABORTED`` from a cancelled rollout, a
    client disconnect, or a request timeout — while it still has in-flight KV
    transfers. When such a request's ``finished_recving`` and ``finished_sending``
    both land in the same ``Scheduler.update_from_output`` step, the stock
    ``_update_from_kv_xfer_finished`` frees it in the recving branch
    (``_free_blocks`` -> ``del self.requests[req_id]``) and then the sending
    branch hits ``assert req_id in self.requests`` and kills the EngineCore. On a
    DP deployment that one death cascades to every rank via the gloo finish-state
    all-reduce, taking down the whole inference pool.

    The trigger is the abort itself, not weight-update pause/resume: it reproduces
    during normal stepping whenever an aborted request's recv and send complete in
    the same step (observed with zero orchestrator cancellations, driven only by
    incidental client-side aborts). Skip already-freed request ids instead of
    asserting — their blocks are freed either way, so dropping the stale
    notification is safe.

    Upstream issue: https://github.com/vllm-project/vllm/issues/46240
    """
    from vllm.logger import init_logger
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.metrics.stats import KV_FETCH_COMPLETED_WAITING
    from vllm.v1.request import RequestStatus

    logger = init_logger("vllm.v1.core.sched.scheduler")

    if getattr(Scheduler._update_from_kv_xfer_finished, "_prime_rl_tolerates_freed", False):
        return

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            # Stale notification for a request freed earlier this step (e.g. an
            # aborted request whose send completion freed it): only its load slot is left.
            if req_id not in self.requests:
                self._aborted_kv_loads.discard(req_id)
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
                self._set_kv_fetch_stage(req, KV_FETCH_COMPLETED_WAITING)
            else:
                assert RequestStatus.is_finished(req.status)
                self._aborted_kv_loads.discard(req_id)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            # See above: the recving branch may have already freed an aborted
            # request whose send also completed this step.
            if req_id not in self.requests:
                continue
            self._free_blocks(self.requests[req_id])

    _update_from_kv_xfer_finished._prime_rl_tolerates_freed = True
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished
    logger.warning("Patched Scheduler._update_from_kv_xfer_finished to tolerate freed (aborted) KV-transfer reqs.")


def monkey_patch_minimax_m2_think_end_passthrough():
    """Keep the literal ``</think>`` in MiniMax-M2 content on tool-calling turns.

    prime-rl serves MiniMax-M2 with ``reasoning=minimax_m2_append_think``, which
    returns content as ``<think>`` + the full completion so think tags round-trip
    through multi-turn re-serialization. vLLM 0.24's minimax_m2 parser engine
    added a ``(CONTENT, THINK_END) -> no-events`` transition that silently
    swallows the ``</think>`` (0.23's regex tool parser passed it through
    untouched), and it also ``.strip()``s content whenever tool calls are
    present. Drop the transition — the engine emits unmatched terminals as plain
    state content — and disable the content strip.
    """
    import dataclasses
    import functools

    from vllm.parser import minimax_m2
    from vllm.parser.engine.parser_engine_config import ParserState

    original_config = minimax_m2.minimax_m2_config

    @functools.cache
    def _patched_config():
        config = original_config()
        transitions = dict(config.transitions)
        del transitions[(ParserState.CONTENT, "THINK_END")]
        return dataclasses.replace(
            config,
            transitions=transitions,
            strip_content_whitespace_with_tools=False,
        )

    minimax_m2.minimax_m2_config = _patched_config


def monkey_patch_routed_experts_with_pd_connectors():
    """Let vLLM's routed-experts capture (AuxOutput connector) run on NIXL P/D instances.

    vLLM (0.34 included) rejects routed-experts capture with P/D connectors, because a decode instance never
    computes a remotely prefilled prompt, so it has no rows for it. This carries our upstream fix
    (AuxOutput patch 02): a request on a decode instance emits rows only from its first forward on
    this instance, and the worker does not buffer the
    partially known first block. The router takes the prompt rows from the prefill instance.

    Since vllm-project/vllm#62591 the scheduler also caps local and external prefix hits at the blocks whose
    rows this instance retains (``max_cache_hit`` / ``cap_external_hit``). A decode instance never has rows for
    a remotely prefilled prompt, so the cap would zero the NIXL load and recompute the whole prompt here; decode
    requests skip both caps. Delete once vLLM supports routed experts with P/D connectors.
    """
    from vllm.config.vllm import VllmConfig
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
    from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector

    original_verify = VllmConfig._verify_aux_output_compatibility
    if getattr(original_verify, "_prime_rl_allows_pd", False):
        return

    def _verify_aux_output_compatibility(config):
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.has_connector("NixlConnector"):
            return original_verify(config)
        # The P/D connector check is the last one; every other check still runs.
        config.kv_transfer_config = None
        try:
            return original_verify(config)
        finally:
            config.kv_transfer_config = kv_transfer_config

    original_build_connector_meta = AuxOutputSchedulerConnector.build_connector_meta

    def build_connector_meta(self, scheduler_output, requests):
        metadata = original_build_connector_meta(self, scheduler_output, requests)
        floors = _emit_floors(self)
        for request_id, emit_start in metadata.requests.items():
            request = requests[request_id]
            if not _is_pd_decode_request(request):
                continue
            # Fixed at the first forward (built before the step advances num_computed_tokens).
            # Normally prompt_len - 1; lower after a failed KV load, where this instance
            # recomputes those rows itself.
            floor = floors.setdefault(request_id, request.num_computed_tokens)
            metadata.requests[request_id] = max(emit_start, floor)
        return metadata

    original_request_finished = AuxOutputSchedulerConnector.request_finished

    def request_finished(self, request):
        # Preemption keeps the floor: a resumed request may hit prompt blocks loaded from the prefill
        # instance, which have no rows here.
        if request.is_finished():
            _emit_floors(self).pop(request.request_id, None)
        return original_request_finished(self, request)

    original_max_cache_hit = AuxOutputSchedulerConnector.max_cache_hit

    def max_cache_hit(self, request):
        # None is the scheduler's "no cap", as without an AuxOutput connector.
        return None if _is_pd_decode_request(request) else original_max_cache_hit(self, request)

    original_cap_external_hit = AuxOutputSchedulerConnector.cap_external_hit

    def cap_external_hit(self, request, num_local, num_external):
        if _is_pd_decode_request(request):
            return num_external
        return original_cap_external_hit(self, request, num_local, num_external)

    original_initialize_output_storage = AuxOutputWorkerConnector.initialize_output_storage

    def initialize_output_storage(self, kv_cache_config):
        # vLLM builds the buffer here once the KV cache layout is resolved, on the output rank only
        original_initialize_output_storage(self, kv_cache_config)
        if self._buffer is not None:
            skip_remote_prefill_partial_block(self)

    _verify_aux_output_compatibility._prime_rl_allows_pd = True
    VllmConfig._verify_aux_output_compatibility = _verify_aux_output_compatibility
    AuxOutputSchedulerConnector.build_connector_meta = build_connector_meta
    AuxOutputSchedulerConnector.request_finished = request_finished
    AuxOutputSchedulerConnector.max_cache_hit = max_cache_hit
    AuxOutputSchedulerConnector.cap_external_hit = cap_external_hit
    AuxOutputWorkerConnector.initialize_output_storage = initialize_output_storage


def skip_remote_prefill_partial_block(worker) -> None:
    """Make an AuxOutput worker skip the mid-block start of a remotely prefilled request's first forward."""
    from vllm.utils.math_utils import cdiv

    buffer = worker._buffer
    capture = buffer.capture
    block_size = buffer.block_size

    def _capture(request_id, token_start, rows):
        state = worker._requests[request_id]
        if state.capture_cursor is None and token_start % block_size and state.emit_cursor >= token_start:
            # The rows before this token exist only on the prefill instance, so this block is never
            # published. Buffer from the next block; this step's rows are still emitted directly.
            state.prime_rl_buffer_start = cdiv(token_start, block_size) * block_size
        skip = min(max(getattr(state, "prime_rl_buffer_start", 0) - token_start, 0), len(rows))
        return capture(request_id, token_start + skip, rows[skip:])

    buffer.capture = _capture


def _emit_floors(connector) -> dict[str, int]:
    """First output token of each remotely prefilled request on this instance."""
    return connector.__dict__.setdefault("_prime_rl_emit_floors", {})


def _is_pd_decode_request(request) -> bool:
    """A request on a P/D decode instance (NIXL clears ``do_remote_prefill`` once scheduled, not ``remote_engine_id``)."""
    params = request.kv_transfer_params or {}
    return bool(params.get("remote_engine_id")) and not params.get("do_remote_decode")


def _routed_experts_load_limit(request, num_computed_tokens: int, block_size: int) -> int | None:
    """How many tokens past ``num_computed_tokens`` an external KV load may cover under routed-expert capture.

    vLLM keeps routing rows in a local store keyed by block hash, and no connector moves them with
    the KV. A loaded block whose rows are not in that store fails the engine step
    (``BlockObjectStoreError``) once a request needs its rows. vLLM returns rows only from
    ``routed_experts_prompt_start`` on (the client already holds the rows before it), so loads
    that stop at that block boundary are always safe. Requests resumed after preemption and
    decode-side P/D requests are not bounded: they emit rows only from their first forward.
    """
    if request.num_output_tokens or _is_pd_decode_request(request):
        return None
    prompt_start = request.sampling_params.routed_experts_prompt_start
    return max(0, prompt_start // block_size * block_size - num_computed_tokens)


def monkey_patch_clamp_kv_offload_loads_for_routed_experts():
    """Clamp native (``OffloadingConnector``) and Mooncake KV offload loads with `_routed_experts_load_limit`.

    Multi-turn reloads stay below the next turn's ``routed_experts_prompt_start`` and keep their
    hits; offloaded prefixes past it (first turns, shared prompts) are recomputed.

    vLLM 0.33 (vllm-project/vllm#62591) also admits an external hit only when this instance still retains the
    routing rows of every block in it, and it retains rows for at most twice the GPU KV cache's blocks. A CPU
    offload tier holds far more than that, so every multi-turn reload of an evicted prefix was refused and
    recomputed. A load within the clamp needs none of its rows, so ``cap_external_hit`` admits it whole; the
    rest still goes through upstream's check. Blocks loaded this way carry no rows, which the local prefix-hit
    cut (`monkey_patch_routed_experts_cut_loaded_prefix_hits`) keeps later requests from reading.
    """
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector

    if getattr(OffloadingConnector._max_loadable_tokens, "_prime_rl_routed_experts_clamp", False):
        return

    original_max_loadable_tokens = OffloadingConnector._max_loadable_tokens
    original_get_num_new_matched_tokens = MooncakeStoreConnector.get_num_new_matched_tokens

    def _max_loadable_tokens(self, request, num_computed_tokens):
        bound = original_max_loadable_tokens(self, request, num_computed_tokens)
        if not self._vllm_config.aux_output_config.enable_return_routed_experts:
            return bound
        limit = _routed_experts_load_limit(request, num_computed_tokens, self._vllm_config.cache_config.block_size)
        if limit is None:
            return bound
        return limit if bound is None else min(bound, limit)

    def _get_num_new_matched_tokens(self, request, num_computed_tokens):
        num_tokens, load_async = original_get_num_new_matched_tokens(self, request, num_computed_tokens)
        if not num_tokens or not self._vllm_config.aux_output_config.enable_return_routed_experts:
            return num_tokens, load_async
        scheduler = self.connector_scheduler
        limit = _routed_experts_load_limit(request, num_computed_tokens, scheduler._block_size)
        if limit is None or num_tokens <= limit:
            return num_tokens, load_async
        if limit == 0:
            del scheduler.load_specs[request.request_id]
            return 0, False
        # The new end is a full block inside the hit, stored under its own hash, so the tail-key
        # overrides for the old (possibly partial) end no longer apply.
        load_spec = scheduler.load_specs[request.request_id]
        load_spec.kvpool_cached_tokens = num_computed_tokens + limit
        load_spec.tail_key_boundaries = ()
        return limit, load_async

    original_cap_external_hit = AuxOutputSchedulerConnector.cap_external_hit

    def cap_external_hit(self, request, num_local, num_external):
        limit = _routed_experts_load_limit(request, num_local, self._block_size)
        if limit is not None and num_external <= limit:
            return num_external
        return original_cap_external_hit(self, request, num_local, num_external)

    _max_loadable_tokens._prime_rl_routed_experts_clamp = True
    OffloadingConnector._max_loadable_tokens = _max_loadable_tokens
    MooncakeStoreConnector.get_num_new_matched_tokens = _get_num_new_matched_tokens
    AuxOutputSchedulerConnector.cap_external_hit = cap_external_hit


def _track_externally_loaded_blocks(kv_cache_manager, gid: int, block_size: int) -> set[int]:
    """Live set of block ids (in KV cache group ``gid``) filled by an external KV load since their last allocation.

    Such blocks enter the GPU prefix cache, but no forward pass on this instance stored their routing rows.
    """
    from vllm.utils.math_utils import cdiv

    loaded: set[int] = set()
    block_pool = kv_cache_manager.block_pool
    get_new_blocks = block_pool.get_new_blocks
    allocate_slots = kv_cache_manager.allocate_slots

    def _get_new_blocks(num_blocks):
        blocks = get_new_blocks(num_blocks)
        loaded.difference_update(block.block_id for block in blocks)
        return blocks

    def _allocate_slots(request, num_new_tokens, *args, **kwargs):
        blocks = allocate_slots(request, num_new_tokens, *args, **kwargs)
        num_external = kwargs.get("num_external_computed_tokens", 0)
        if blocks is not None and num_external:
            start = request.num_computed_tokens + kwargs.get("num_new_computed_tokens", 0)
            block_ids = kv_cache_manager.get_block_ids(request.request_id)[gid]
            loaded.update(block_ids[start // block_size : cdiv(start + num_external, block_size)])
        return blocks

    block_pool.get_new_blocks = _get_new_blocks
    kv_cache_manager.allocate_slots = _allocate_slots
    return loaded


def _first_loaded_block(block_ids: list[int], loaded: set[int], prompt_start: int, block_size: int) -> int | None:
    """Index of the first block in ``block_ids`` holding rows from ``prompt_start`` on that was filled by a load."""
    return next((i for i in range(prompt_start // block_size, len(block_ids)) if block_ids[i] in loaded), None)


def monkey_patch_routed_experts_cut_loaded_prefix_hits():
    """Cut local prefix-cache hits at externally loaded blocks that a request needs routing rows from.

    The load clamp keeps the loading request safe, but the loaded blocks then serve local prefix
    hits. A later request with a lower ``routed_experts_prompt_start`` would need their rows, which
    this instance never stored, and the engine step would fail with ``BlockObjectStoreError``. Such
    hits are cut at the first loaded block, so the rest is recomputed. Requests with output (resumed
    after preemption) and decode-side P/D requests need no prompt rows from the store.
    """
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import is_full_attention_spec

    if getattr(Scheduler._get_local_prefix_cache_hit, "_prime_rl_routed_experts_cut", False):
        return

    original_init = Scheduler.__init__
    original_get_local_prefix_cache_hit = Scheduler._get_local_prefix_cache_hit

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._prime_rl_loaded_blocks = None
        if self.aux_output_connector is None or self.connector is None:
            return
        groups = self.kv_cache_config.kv_cache_groups
        gid = next(i for i, g in enumerate(groups) if not g.host_resident and is_full_attention_spec(g.kv_cache_spec))
        block_size = groups[gid].kv_cache_spec.block_size
        self._prime_rl_loaded_blocks = (
            gid,
            block_size,
            _track_externally_loaded_blocks(self.kv_cache_manager, gid, block_size),
        )

    def _get_local_prefix_cache_hit(self, request, max_cache_hit_length=None):
        hit = original_get_local_prefix_cache_hit(self, request, max_cache_hit_length)
        if self._prime_rl_loaded_blocks is None or request.num_output_tokens or _is_pd_decode_request(request):
            return hit
        gid, block_size, loaded = self._prime_rl_loaded_blocks
        start = request.sampling_params.routed_experts_prompt_start
        cut = _first_loaded_block(hit[0].get_block_ids()[gid], loaded, start, block_size) if loaded else None
        if cut is None:
            return hit
        # Redo the lookup capped at the cut, so every KV cache group (e.g. Mamba) agrees on the hit.
        kv_cache_manager = self.kv_cache_manager
        keep = cut * block_size // self.block_size * self.block_size
        blocks, num_local, num_uncached = kv_cache_manager.coordinator.find_longest_cache_hit(
            request.block_hashes, keep
        )
        boundary = num_local + num_uncached if num_uncached else 0
        return kv_cache_manager.create_kv_cache_blocks(blocks), num_local, boundary, False

    _get_local_prefix_cache_hit._prime_rl_routed_experts_cut = True
    Scheduler.__init__ = __init__
    Scheduler._get_local_prefix_cache_hit = _get_local_prefix_cache_hit


def monkey_patch_strip_routed_experts_from_chat():
    """Drop routed_experts from chat-completions responses.

    routed_experts are only consumed via the serialized ``/generate``
    (serving_tokens) path used for router-replay training, which encodes them as a
    ``{data, shape, start}`` object the PD router can merge. The stock
    chat-completions path instead encodes them as a base64 ``np.save`` *string*,
    which the PD router rejects ("prefill routed_experts must be an object with
    base64 data and shape") and fails every eval rollout (evals go through chat
    completions). ``enable_return_routed_experts`` is a server-wide model-config
    flag with no per-request toggle, so strip the field on the chat path here.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if getattr(OpenAIServingChat.chat_completion_full_generator, "_prime_rl_strips_routed_experts", False):
        return

    _original = OpenAIServingChat.chat_completion_full_generator

    async def _strip(result_generator):
        async for res in result_generator:
            for output in res.outputs:
                output.routed_experts = None
            yield res

    async def _patched(self, request, result_generator, *args, **kwargs):
        return await _original(self, request, _strip(result_generator), *args, **kwargs)

    _patched._prime_rl_strips_routed_experts = True
    OpenAIServingChat.chat_completion_full_generator = _patched
    logger.info(
        "Stripped routed_experts from chat-completions responses (PD router merges only the /generate object form)."
    )


def monkey_patch_skip_prompt_routing_for_chat():
    """Start chat requests' routed experts at the end of the prompt.

    Chat responses drop routed_experts (``monkey_patch_strip_routed_experts_from_chat``), but with
    ``enable_return_routed_experts`` the engine still materializes every prompt's routing rows and
    ships them to the API server each turn: about 258 B per token, so tens of MB for a long eval
    prompt that is thrown away. Setting ``routed_experts_prompt_start`` to the prompt length keeps
    only the completion's rows. A request that sets its own start keeps it.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.logger import init_logger
    from vllm.sampling_params import SamplingParams

    if getattr(OpenAIServingChat._log_inputs, "_prime_rl_skips_prompt_routing", False):
        return
    original_log_inputs = OpenAIServingChat._log_inputs

    # The chat route calls _log_inputs right after it builds each prompt's SamplingParams, with the rendered prompt.
    def _log_inputs(self, request_id, inputs, params, lora_request):
        if isinstance(params, SamplingParams) and params.routed_experts_prompt_start == 0:
            prompt_ids = self._extract_prompt_components(inputs).token_ids
            if prompt_ids is not None:
                params.routed_experts_prompt_start = len(prompt_ids)
        return original_log_inputs(self, request_id, inputs, params, lora_request)

    _log_inputs._prime_rl_skips_prompt_routing = True
    OpenAIServingChat._log_inputs = _log_inputs
    init_logger("vllm.prime_rl.routed_experts").info("Chat requests return routed experts only for their completions.")


def _patch_qwen35_moe_lora_format():
    """Force Qwen3.5-MoE onto vLLM's 2D per-expert LoRA format.

    vLLM 0.24.0 still defaults ``Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = True``,
    which makes the LoRA loader expect 3D stacked-expert adapters
    (``base_layer.lora_{A,B}.weight`` / ``lora_{A,B}.weight``, experts folded into the
    rank dim; see ``_stack_moe_lora_weights``). Our trainer instead emits the 2D
    per-expert layout (``{expert_id}.gate_proj.lora_A.weight`` ...) from
    ``LoRAGroupedExperts.adapter_state_dict`` -- vLLM only consults that layout
    when ``is_3d_moe_weight`` is False (or ``enable_mixed_moe_lora_format=True``).
    Without this override the adapters fail to load with key/shape mismatches.

    The rest of the old Qwen3.5 LoRA shim (the in_proj_qkvz packed-mapping fix and the
    N-slice ``can_replace_layer`` / ``slice_lora_a`` generalizations for vllm#36372) is
    handled natively by 0.23.0 and was dropped. Remove this too once we either adopt the
    3D stacked save format (like gpt-oss) or start the engine with
    ``enable_mixed_moe_lora_format=True``.
    """
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = False


def _patch_lora_key_prefix():
    """Accept both bare-suffix and fully-qualified expert module names in LoRA adapters.

    Copy of vLLM 0.34.0's ``LoRAModel.from_local_checkpoint`` with one change: the
    ``.experts`` branch of ``check_unexpected_modules`` accepts either the bare suffix
    (``down_proj``) or the qualified per-expert name (``experts.N.down_proj``), where
    upstream only accepts the qualified form. Our trainer's 2D per-expert adapters
    (Qwen3.5-MoE) carry names whose qualified form is not in the expected set while
    the bare suffix is; Qwen3-30B-A3B adapters go the other way. Upstream fix
    vllm-project/vllm#38522 was closed unmerged, so this stays.
    """
    from vllm.lora.lora_model import (
        LoRAModel,
        MoEEPLoadSpec,
        PEFTHelper,
        WeightsMapper,
        _is_remote_expert_key,
        get_lora_id,
        is_base_embedding_weights,
        os,
        parse_fine_tuned_lora_name,
        safetensors,
    )

    def _patched_from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        skip_prefixes: list[str] | None = None,
        moe_ep_spec: MoEEPLoadSpec | None = None,
        unsupported_lora_modules: set[str] | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.
            skip_prefixes: List of module name prefixes to skip during loading.
                Models can define this to skip modules not used in inference
                (e.g., MTP layers). Format: ["mtp."]
            moe_ep_spec: When 2D FusedMoE LoRA modules are present with
                expert parallelism enabled, the (ep_rank, local, global)
                slicing metadata shared across all MoE layers. Non-local
                expert weights are skipped at read time instead of being
                loaded and discarded later.
            unsupported_lora_modules: Full names of model layers that match
                expected_lora_modules but are not wrapped by LoRA, such as
                BERT's pooler.dense. LoRA weights on them are unexpected.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        lora_pt_file_path = os.path.join(lora_dir, "adapter_model.pt")

        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[list[str] | str] = []

        def check_unexpected_modules(modules: dict):
            for lora_module in modules.keys():  # noqa
                if is_base_embedding_weights(lora_module):
                    continue
                # Handle PEFT file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                # Skip modules based on model-defined prefixes
                if skip_prefixes and cls._should_skip_module(lora_module, skip_prefixes):
                    continue
                module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)
                base_name = module_name.rsplit(".", 1)[-1]
                # Case for expert lora weights.
                ## START PATCHED CODE (upstream only accepts the qualified form)
                if ".experts" in module_name:
                    expert_suffix = module_name.split(".")[-1]
                    experts_qualified = "experts" + module_name.split(".experts", 1)[-1]
                    if expert_suffix not in expected_lora_modules and experts_qualified not in expected_lora_modules:
                        unexpected_modules.append(module_name)
                ## END PATCHED CODE

                elif base_name not in expected_lora_modules and base_name not in (peft_helper.modules_to_save or ()):
                    unexpected_modules.append(module_name)

                elif module_name in (unsupported_lora_modules or ()):
                    unexpected_modules.append(f"{module_name} (not a LoRA layer)")

            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )

        if os.path.isfile(lora_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path, framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    if moe_ep_spec is not None and _is_remote_expert_key(module, moe_ep_spec):
                        continue
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            lora_file_path = lora_bin_file_path if os.path.isfile(lora_bin_file_path) else lora_pt_file_path
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
            check_unexpected_modules(tensors)
            if moe_ep_spec is not None:
                # `.bin`/`.pt` adapters can't be lazy-loaded, but pruning
                # the dict here still frees the non-local expert tensors
                # before the dtype cast / pin_memory work that follows.
                tensors = {k: v for k, v in tensors.items() if not _is_remote_expert_key(k, moe_ep_spec)}
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes,
        )

    LoRAModel.from_local_checkpoint = classmethod(_patched_from_local_checkpoint)


def monkey_patch_minimax_m2_for_lora():
    """Patch vLLM's MiniMaxM2 model for LoRA compatibility.

    These patches are only needed when using LoRA with MiniMax M2 but are safe
    to apply unconditionally (verified with non-LoRA runs). We apply them
    unconditionally because the vLLM plugin runs before the vLLM config is
    available, so we can't check if LoRA is enabled.

    Problem 1 — Gate dtype mismatch:
        vLLM's MiniMaxM2MoE creates the gate (router) with params_dtype=float32
        and casts inputs to float32. When LoRA is enabled, vLLM wraps ALL
        ReplicatedLinear layers (including the gate) with LoRA support. Even
        though our adapter has no gate LoRA weights, the LoRA Triton kernel
        still runs for all wrapped layers when any adapter is active — and it
        asserts inputs are float16/bfloat16. Qwen3 MoE doesn't have this
        problem because its gate uses the model dtype.
        Fix: rebuild the gate as GateLinear with a bf16 weight (out_dtype=float32
        keeps fp32 router logits). vLLM 0.24.0's own forward already drops the
        float32 input cast. FusedMoE also has router_logits_dtype=float32, so
        routing precision is preserved inside the expert dispatch.

    Problem 2 — Adapter key naming mismatch:
        PrimeRL saves adapter keys using its internal naming convention
        (mlp.experts.{j}.gate_proj/down_proj/up_proj), which matches Qwen3 MoE
        but not MiniMax M2. vLLM's MiniMax M2 model expects HF-style keys
        (block_sparse_moe.experts.{j}.w1/w2/w3). For full model weights this
        is handled by vLLM's load_weights(), but LoRA adapters are loaded
        through a separate path (LoRAModel.from_local_checkpoint) that doesn't
        have model-specific key translation.
        Fix: set hf_to_vllm_mapper on the model class so vLLM remaps adapter
        keys during LoRA loading. This attribute is only read by _load_adapter
        in the LoRA worker manager — it has no effect without LoRA.
    """
    from vllm.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM, MiniMaxM2MoE
    from vllm.model_executor.models.utils import WeightsMapper

    # --- Gate dtype fix (only matters with LoRA, safe without) ---
    _original_init = MiniMaxM2MoE.__init__

    def _patched_init(self, config, quant_config=None, prefix=""):
        _original_init(self, config, quant_config, prefix)
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

        # vLLM 0.24.0 builds the gate as GateLinear with a float32 weight; rebuild it
        # with a bf16 weight (model dtype) so the LoRA Triton kernel's float16/bfloat16
        # assertion passes, keeping out_dtype=float32 so router logits stay fp32 (the
        # GateLinear bf16xbf16->fp32 path).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

    MiniMaxM2MoE.__init__ = _patched_init

    # --- Adapter key remapping (only read by vLLM's LoRA adapter loader) ---
    MiniMaxM2ForCausalLM.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.experts.": ".block_sparse_moe.experts.",
            ".gate_proj.": ".w1.",
            ".down_proj.": ".w2.",
            ".up_proj.": ".w3.",
        },
    )


def monkey_patch_no_moe_lora():
    """This disables LoRA for MoE layers and makes them pick better kernels.

    Otherwise, the oracle will always try to pick TritonExperts.
    For blackwells, we want TRTLLMFlashInfer.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    original_post_init = FusedMoEConfig.__post_init__

    def _patched__post_init__(self: FusedMoEConfig):
        original_post_init(self)
        # Disable LoRA for MoE layers. `is_lora_enabled` is only read later during
        # kernel selection (modular_kernel / unquantized oracle), never inside
        # `__post_init__`, so flipping it after the original runs is sufficient.
        self.is_lora_enabled = False

    FusedMoEConfig.__post_init__ = _patched__post_init__


def monkey_patch_dp_coordinator_startup_timeout():
    """Raise the DP coordinator startup timeout from vLLM's hard-coded 120s.

    The coordinator child process is spawned on the DP-rank-0 API server while
    every engine-core rank on the node is importing and loading weights, so its
    own spawn-time re-import can exceed the hard-coded timeout under that CPU/IO
    contention (seen on multi-node disaggregated GLM-5.1 launches). Configurable
    via PRIME_DP_COORDINATOR_STARTUP_TIMEOUT (seconds, default 300).
    """
    import multiprocessing.connection
    import os

    from vllm.v1.engine.coordinator import DPCoordinator

    timeout = float(os.environ.get("PRIME_DP_COORDINATOR_STARTUP_TIMEOUT", "300"))

    def _patched_wait_for_zmq_addrs(self, zmq_addr_pipe):
        try:
            ready = multiprocessing.connection.wait([zmq_addr_pipe, self.proc.sentinel], timeout=timeout)
            if not ready:
                raise RuntimeError(
                    f"DP Coordinator process failed to report ZMQ addresses within {timeout}s during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError("DP Coordinator process failed during startup.") from None
        finally:
            zmq_addr_pipe.close()

    DPCoordinator._wait_for_zmq_addrs = _patched_wait_for_zmq_addrs
