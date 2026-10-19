import os
from dataclasses import dataclass

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
    monkey_patch_online_fp8_parameter_cast()
    monkey_patch_online_mxfp8_moe_keep_kernel_on_reload()
    monkey_patch_strip_routed_experts_from_chat()
    monkey_patch_dp_coordinator_startup_timeout()
    monkey_patch_minimax_m2_for_lora()
    monkey_patch_packed_sampling_masks()
    # Set by `server()` when the LoRA target modules include no expert layers.
    if os.environ.get("PRIME_NO_MOE_LORA") == "1":
        monkey_patch_no_moe_lora()
    monkey_patch_deepseek_v4_c128_boundary()
    monkey_patch_deepseek_v4_attn_sink_loading()


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
    arrays and hand ``PrimeRlServingTokens`` a ``PackedSamplingMask`` instead. Each
    appended chunk is one position (``SamplingMaskLists.slice_request`` asserts it).
    Only non-streaming (``FINAL_ONLY``) outputs are packed, the form
    ``PrimeRlServingTokens.serve_tokens_full_generator`` reads; others keep vLLM's form.
    """
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.output_processor import RequestState

    original_init = RequestState.__init__
    original_new_completion_output = RequestState._new_completion_output

    class _MaskIds(list):
        def append(self, chunk):
            super().append(chunk.token_ids)

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


def monkey_patch_deepseek_v4_c128_boundary():
    """Fix DeepSeek V4 long-context KV corruption under piecewise CUDA graphs.

    On vLLM 0.30+, including the pinned 0.32.0, the piecewise CUDA graphs are captured on a step
    that skips writing C128 compressed KV, so every replay skips it too. Mixed batches of up to
    512 tokens never store their C128 entries, later queries attend over stale compressed KV, and
    long-context rollouts drift from the trainer, inflating mismatch KL.

    The skip is an optimization: ``DeepseekCompressor.forward`` returns early when
    ``_get_c128_boundary`` reports that no request completes a 128-token block this step.
    vllm-project/vllm#55353 made that helper return a bool under Model Runner V2, so capture on
    dummy batches records the early return. Returning None instead disables the skip, which is
    the path vLLM already takes under async speculative decoding and the one 0.29 always took.

    The upstream fix, vllm-project/vllm#59709, keeps the skip on eager steps and only stops it
    from being captured. That change sits inline in ``forward``, so matching it here would mean
    copying the whole method. Patching the helper costs one extra compress kernel launch on eager
    steps that complete no block, and the kernel already skips those tokens on the GPU.

    Remove this patch once vllm-project/vllm#59709 lands in the pinned vLLM version.
    """
    from vllm.logger import init_logger
    from vllm.models.deepseek_v4 import compressor

    if getattr(compressor._get_c128_boundary, "_prime_rl_always_compresses", False):
        return

    def _get_c128_boundary(metadata):
        return None

    _get_c128_boundary._prime_rl_always_compresses = True
    compressor._get_c128_boundary = _get_c128_boundary
    init_logger("vllm.prime_rl.deepseek_v4").info("DeepSeek V4 C128 compressor store runs on every step.")


def monkey_patch_deepseek_v4_attn_sink_loading():
    """Fix DeepSeek V4 weight updates silently keeping the boot-time attention sinks.

    ``DeepseekV4Model.load_weights`` writes the sinks with a bare slice copy instead of
    ``param.weight_loader``. Layerwise reload moves parameters to meta and only buffers writes
    that go through a loader, so the copy is silently dropped. This pads this rank's heads back
    to the full parameter with ``-inf`` (no sink) and loads that through ``param.weight_loader``.

    Remove this patch once vllm-project/vllm#57797 lands in the pinned vLLM version. Only
    ``nvidia/model.py`` is covered, not the MTP, DSpark or non-NVIDIA variants.
    """
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.models.deepseek_v4.nvidia import model as dsv4_model

    original_load_weights = dsv4_model.DeepseekV4Model.load_weights
    if getattr(original_load_weights, "_prime_rl_uses_weight_loaders", False):
        return

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        tp_size = dsv4_model.get_tensor_model_parallel_world_size()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = heads_per_rank * dsv4_model.get_tensor_model_parallel_rank()
        loaded_params: set[str] = set()

        def remaining_weights():
            for name, weight in weights:
                if "attn_sink" not in name or dsv4_model.is_pp_missing_parameter(name, self):
                    yield name, weight
                    continue
                param = params[name]
                sink = weight.new_full(tuple(param.shape), -float("inf"))
                sink[:heads_per_rank] = weight[head_start : head_start + heads_per_rank]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, sink)
                loaded_params.add(name)

        loaded_params.update(original_load_weights(self, remaining_weights()))
        return loaded_params

    load_weights._prime_rl_uses_weight_loaders = True
    dsv4_model.DeepseekV4Model.load_weights = load_weights
    init_logger("vllm.prime_rl.deepseek_v4").info("DeepSeek V4 attn_sink loads through param.weight_loader.")


def monkey_patch_online_mxfp8_moe_keep_kernel_on_reload():
    """Keep the boot-time online-MXFP8 MoE kernel across weight reloads.

    ``Mxfp8OnlineMoEMethod._setup_kernel`` rebuilds ``moe_quant_config`` and ``moe_kernel`` on every
    reload. Decode replays CUDA graphs captured against the boot-time kernel, so once the old
    kernel is dropped the graphs read freed memory for tensors only it referenced, and decode
    samples sporadic garbage tokens that eager prefill (and the trainer) do not reproduce. The
    boot-time kernel references the layer's original parameters, which layerwise reload refills
    in place, so on reload we only install the new weights and keep the kernel.

    Upstream fix: https://github.com/vllm-project/vllm/pull/62406; remove this patch when our
    minimum vLLM version includes it.
    """
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import convert_to_fp8_moe_kernel_format
    from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineMoEMethod
    from vllm.model_executor.utils import replace_parameter

    original_setup_kernel = Mxfp8OnlineMoEMethod._setup_kernel
    if getattr(original_setup_kernel, "_prime_rl_keeps_kernel", False):
        return

    def _setup_kernel(self, layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale):
        if getattr(self, "moe_kernel", None) is None:
            return original_setup_kernel(self, layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale)
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

    _setup_kernel._prime_rl_keeps_kernel = True
    Mxfp8OnlineMoEMethod._setup_kernel = _setup_kernel


def monkey_patch_online_fp8_parameter_cast():
    """Pass plain tensors to vLLM's compiled block-FP8 caster.

    Layerwise reload restores ``ModelWeightParameter`` objects before online
    quantization. TorchDynamo recursively dispatches that tensor subclass while
    tracing ``per_block_cast_to_fp8``. ``Parameter.data`` shares storage but is
    a plain tensor, matching the input accepted during the initial model load.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.online import fp8

    original_cast = fp8.per_block_cast_to_fp8
    if getattr(original_cast, "_prime_rl_unwraps_parameters", False):
        return

    def _per_block_cast_to_fp8(x, *args, **kwargs):
        if isinstance(x, Parameter):
            x = x.data
        return original_cast(x, *args, **kwargs)

    _per_block_cast_to_fp8._prime_rl_unwraps_parameters = True
    fp8.per_block_cast_to_fp8 = _per_block_cast_to_fp8


def monkey_patch_kv_xfer_finished_tolerate_freed():
    """Tolerate KV-transfer finish notifications for already-freed requests.

    In disaggregated P/D (NIXL, optionally + a KV store connector) a request can
    be finished — most often ``FINISHED_ABORTED`` from an off-policy cancel, a
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
    the same step (observed with zero off-policy cancellations, driven only by
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
            # aborted request whose send completion freed it). Nothing to do.
            if req_id not in self.requests:
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
                self._set_kv_fetch_stage(req, KV_FETCH_COMPLETED_WAITING)
            else:
                assert RequestStatus.is_finished(req.status)
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

    vLLM (0.32 included) rejects routed-experts capture with P/D connectors, because a decode instance never
    computes a remotely prefilled prompt, so it has no rows for it. This carries our upstream fix
    (AuxOutput patch 02): a request on a decode instance emits rows only from its first forward on
    this instance, and the worker does not buffer the
    partially known first block. The router takes the prompt rows from the prefill instance. Delete
    once vLLM supports routed experts with P/D connectors.
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

    original_worker_init = AuxOutputWorkerConnector.__init__

    def worker_init(self, *args, **kwargs):
        original_worker_init(self, *args, **kwargs)
        if self._buffer is not None:
            skip_remote_prefill_partial_block(self)

    _verify_aux_output_compatibility._prime_rl_allows_pd = True
    VllmConfig._verify_aux_output_compatibility = _verify_aux_output_compatibility
    AuxOutputSchedulerConnector.build_connector_meta = build_connector_meta
    AuxOutputSchedulerConnector.request_finished = request_finished
    AuxOutputWorkerConnector.__init__ = worker_init


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
    """
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

    _max_loadable_tokens._prime_rl_routed_experts_clamp = True
    OffloadingConnector._max_loadable_tokens = _max_loadable_tokens
    MooncakeStoreConnector.get_num_new_matched_tokens = _get_num_new_matched_tokens


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

    def _get_local_prefix_cache_hit(self, request):
        hit = original_get_local_prefix_cache_hit(self, request)
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

    Copy of vLLM 0.32.0's ``LoRAModel.from_local_checkpoint`` with one change: the
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
