import os

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
    monkey_patch_nano_v3_reasoning_parser()
    monkey_patch_minimax_m2_think_end_passthrough()
    monkey_patch_kv_xfer_finished_tolerate_freed()
    monkey_patch_online_fp8_parameter_cast()
    monkey_patch_deepseek_v4_allowed_layer_types()
    monkey_patch_tokenize_params_validation()
    monkey_patch_strip_routed_experts_from_chat()
    monkey_patch_dp_coordinator_startup_timeout()
    monkey_patch_minimax_m2_for_lora()
    # Set by `server()` when the LoRA target modules include no expert layers.
    if os.environ.get("PRIME_NO_MOE_LORA") == "1":
        monkey_patch_no_moe_lora()
    monkey_patch_deepseek_v4_request_tools_placement()
    monkey_patch_fp8_ue8m0_weight_scales()
    monkey_patch_triton_moe_swiglu_clamp()
    monkey_patch_fp8_stochastic_weight_rounding()
    monkey_patch_deepseek_v4_attn_sink_loading()
    monkey_patch_deepseek_v4_c128_boundary()
    # Last, so a failing local plugin cannot skip the patches above.
    load_vllm_plugins()


def load_vllm_plugins():
    """Run the ``inference.vllm_plugins`` callables exported in ``$PRIME_VLLM_PLUGINS``."""
    import json
    import os

    targets = json.loads(os.environ.get("PRIME_VLLM_PLUGINS") or "[]")
    if not targets:
        return

    from renderers.custom import load_target
    from vllm.logger import init_logger

    logger = init_logger("vllm.prime_rl.plugins")
    for target in targets:
        load_target(target)()
        logger.info(f"Loaded vLLM plugin {target}")


def monkey_patch_deepseek_v4_allowed_layer_types():
    """Let vLLM's `DeepseekV4Config` construct under a transformers pin below 5.15.

    `vllm/transformers_utils/configs/deepseek_v4.py` leaves `layer_types` to
    `PretrainedConfig`, whose validator checks it against a vocabulary that only gained V4's
    compressed attention names in transformers 5.15, so `AutoConfig.from_pretrained` on any
    DeepSeek V4 checkpoint raises. vLLM declares `transformers>=5.5.3`, so this is a gap in
    upstream's own version floor rather than something prime-rl introduced.

    `EngineArgs.__post_init__` loads general plugins before `create_model_config`, which is why
    patching here is early enough. The trainer applies the same shim on its own import path.
    """
    from prime_rl.utils.transformers_compat import allow_deepseek_v4_layer_types

    allow_deepseek_v4_layer_types()


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


def monkey_patch_triton_moe_swiglu_clamp():
    """Make vLLM's ``TritonExperts`` honour the swiglu clamp on its fused fp8 block-quant fast path.

    ``TritonExperts.apply`` fuses SiLU-and-mul with the per-block fp8 quantization of the
    second expert GEMM's input through ``ops.silu_and_mul_per_block_quant`` whenever the
    activation is SiLU, the weights are fp8 w8a8 with 128x128 blocks, no LoRA is active and
    DeepGEMM E8M0 is off. That op has no clamp argument, so a model's ``swiglu_limit``
    (DeepSeek V4 Flash: 10.0) is dropped on that path while every other path applies it,
    and tokens whose gate or up pre-activation exceeds the limit get a wrong expert output.
    With ``VLLM_USE_DEEP_GEMM_E8M0=0`` this is the path taken for every batch below 128
    tokens, i.e. every decode step.

    The fast-path condition's only reference to ``is_deep_gemm_e8m0_used`` is the
    module-level name in ``triton_moe``, so while an ``apply`` call runs with a clamp
    configured that name is bound to return True, which routes the call to the clamped
    branch (``self.activation`` followed by ``moe_kernel_quantize_input``). Calls without a
    clamp keep the fused fast path. Redundant once upstream gates the fast path on the
    clamp itself.
    """
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    logger = init_logger("vllm.prime_rl.fused_moe")
    original_apply = triton_moe.TritonExperts.apply
    if getattr(original_apply, "_prime_honours_swiglu_clamp", False):
        return

    def apply(self, *args, **kwargs):
        if self.activation_config.clamp_limit is None:
            return original_apply(self, *args, **kwargs)
        saved = triton_moe.is_deep_gemm_e8m0_used
        triton_moe.is_deep_gemm_e8m0_used = lambda: True
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            triton_moe.is_deep_gemm_e8m0_used = saved

    apply._prime_honours_swiglu_clamp = True
    triton_moe.TritonExperts.apply = apply
    logger.info("TritonExperts.apply takes the clamped activation path when a swiglu clamp is configured.")


def monkey_patch_fp8_ue8m0_weight_scales():
    """Off unless ``PRIME_FP8_UE8M0_WEIGHT_SCALES=1`` (``inference.fp8_ue8m0_weight_scales``): power-of-two block scales.

    ``Fp8PerBlockOnlineLinearMethod`` and its MoE counterpart both call
    ``per_block_cast_to_fp8(..., use_ue8m0=False)``, which picks the scale
    ``amax / 448``. The published DeepSeek V4 bf16 checkpoint is a dequantized
    FP8 release whose weights already lie exactly on the e4m3 grid with
    power-of-two block scales, so that scale choice rotates the grid and injects
    the full e4m3 rounding error on weights that would otherwise round-trip
    exactly. Forcing ``use_ue8m0=True`` restores the original grid.

    The scales stay fp32, so no kernel change is implied: a power of two is an
    ordinary fp32 scale. This is distinct from ``VLLM_USE_DEEP_GEMM_E8M0=1``,
    which re-quantizes an already-rounded weight and therefore double-rounds;
    with both on, the re-quantization finds the weights already on power-of-two
    scales and leaves them alone, so the pair gives UE8M0 activation scales and
    exact weights.
    """
    import os

    if os.environ.get("PRIME_FP8_UE8M0_WEIGHT_SCALES") != "1":
        return

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.online import fp8

    logger = init_logger(__name__)
    original_cast = fp8.per_block_cast_to_fp8
    if getattr(original_cast, "_prime_forces_ue8m0", False):
        return

    def _per_block_cast_to_fp8(x, *args, **kwargs):
        kwargs["use_ue8m0"] = True
        return original_cast(x, *args, **kwargs)

    _per_block_cast_to_fp8._prime_forces_ue8m0 = True
    fp8.per_block_cast_to_fp8 = _per_block_cast_to_fp8
    logger.info("PRIME_FP8_UE8M0_WEIGHT_SCALES=1: quantizing online FP8 weights with power-of-two block scales.")


_STOCHASTIC_ROUNDING_ROW_CHUNK = 2048
_E4M3_SIGN_BIT = 0x80
_E4M3_MAGNITUDE_MASK = 0x7F
_E4M3_MAX_FINITE_MAGNITUDE = 0x7E


def _expand_block_scales(scales: torch.Tensor, block_size: list[int], rows: int, cols: int) -> torch.Tensor:
    block_m, block_n = block_size
    return scales.repeat_interleave(block_m, dim=0)[:rows].repeat_interleave(block_n, dim=1)[:, :cols]


def stochastic_round_fp8(x_scaled: torch.Tensor, q_near: torch.Tensor) -> torch.Tensor:
    """Move each round-to-nearest e4m3 value ``q_near`` to its neighbour on the far side of ``x_scaled`` with probability equal to the fractional distance, so the result equals ``x_scaled`` in expectation."""
    q_near_float = q_near.float()
    bits = q_near.contiguous().view(torch.uint8).to(torch.int16)
    sign = bits & _E4M3_SIGN_BIT
    magnitude = bits & _E4M3_MAGNITUDE_MASK
    farther_from_zero = x_scaled.abs() > q_near_float.abs()
    neighbour_magnitude = torch.where(farther_from_zero, magnitude + 1, magnitude - 1).clamp(
        0, _E4M3_MAX_FINITE_MAGNITUDE
    )
    neighbour = (sign | neighbour_magnitude).to(torch.uint8).view(torch.float8_e4m3fn).float()
    gap = neighbour - q_near_float
    probability = torch.where(gap != 0, (x_scaled - q_near_float) / gap, torch.zeros_like(gap)).clamp(0, 1)
    pick_neighbour = torch.rand_like(probability) < probability
    return torch.where(pick_neighbour, neighbour, q_near_float).to(torch.float8_e4m3fn)


def monkey_patch_fp8_stochastic_weight_rounding():
    """Off unless ``PRIME_FP8_STOCHASTIC_WEIGHT_ROUNDING=1`` (``inference.fp8_stochastic_weight_rounding``): unbiased weight rounding.

    ``per_block_cast_to_fp8`` rounds to nearest, so a weight that starts on the e4m3
    grid keeps its served value until the trainer has moved it half a bin, about
    1000 steps at lr 1e-6. Until then the served policy is pinned at step 0 while
    the trainer drifts and the trainer-vs-inference mismatch grows. Rounding each
    weight to the far neighbour with probability equal to its fractional distance
    makes the served weight unbiased, so it tracks sub-bin updates in expectation.
    On-grid input has zero fractional distance and rounds to nearest, so the
    checkpoint itself quantizes to identical bytes.

    Wraps whatever ``per_block_cast_to_fp8`` is installed when this runs, so it
    composes with the UE8M0 wrapper registered just before it. The block scales
    come back unchanged; only the e4m3 payload is re-rounded, in row chunks so
    the fp32 temporaries stay bounded on large expert weights.
    """
    import os

    if os.environ.get("PRIME_FP8_STOCHASTIC_WEIGHT_ROUNDING") != "1":
        return

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.online import fp8
    from vllm.utils.deep_gemm import DEFAULT_BLOCK_SIZE

    logger = init_logger("vllm.prime_rl.fp8")
    original_cast = fp8.per_block_cast_to_fp8
    if getattr(original_cast, "_prime_rounds_stochastically", False):
        return

    def _per_block_cast_to_fp8(x, *args, **kwargs):
        quantized, scales = original_cast(x, *args, **kwargs)
        block_size = kwargs.get("block_size", args[0] if args else DEFAULT_BLOCK_SIZE)
        block_m = block_size[0]
        rows, cols = quantized.shape
        row_chunk = max(block_m, _STOCHASTIC_ROUNDING_ROW_CHUNK // block_m * block_m)
        out = torch.empty_like(quantized)
        for row_start in range(0, rows, row_chunk):
            row_end = min(row_start + row_chunk, rows)
            scale_rows = scales[row_start // block_m : -(-row_end // block_m)]
            scale_per_element = _expand_block_scales(scale_rows, block_size, row_end - row_start, cols)
            x_scaled = x[row_start:row_end].float() * (1.0 / scale_per_element)
            out[row_start:row_end] = stochastic_round_fp8(x_scaled, quantized[row_start:row_end])
        return out, scales

    _per_block_cast_to_fp8._prime_rounds_stochastically = True
    fp8.per_block_cast_to_fp8 = _per_block_cast_to_fp8
    logger.info("PRIME_FP8_STOCHASTIC_WEIGHT_ROUNDING=1: rounding online FP8 weights stochastically.")


def monkey_patch_deepseek_v4_c128_boundary():
    """Always run DeepSeek V4's C128 compressor store, as vLLM 0.29 did under Model Runner V2.

    ``DeepseekCompressor.forward`` returns before the C128 compress, norm, RoPE and KV-store
    kernel when the step is not a FULL CUDA graph and ``c128_boundary is False``. vLLM 0.31
    (vllm-project/vllm#55353) derives that flag from ``seq_lens_cpu_upper_bound``, which Model
    Runner V2 sets, so piecewise capture on dummy batches bakes the early return into every
    piecewise graph: mixed batches of up to 512 tokens never write C128 entries, and later
    queries read stale compressed KV. In 0.29 the flag read ``_num_computed_tokens_cpu``, which
    Model Runner V2 never sets, so it was always None and the kernel always ran. Returning None
    restores that.
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
    """Route DeepSeek V4's attention sinks through vLLM's weight loaders.

    ``DeepseekV4Model.load_weights`` writes the sinks with a bare
    ``params_dict[name][:n].copy_(narrow_weight)`` instead of going through
    ``param.weight_loader``. Layerwise reload works by moving a layer's tensors to meta
    and wrapping each loader to buffer the incoming tensor, so that copy lands in a meta
    tensor and is discarded: ``meta[:n].copy_(real)`` succeeds silently. The module's
    ``load_numel`` stays 0, finalize restores the boot value with only a warning, and the
    loader still does ``loaded_params.add(name)``, so a ``named_parameters() -
    loaded_params`` diff cannot see the loss either. Attention sinks are trainable, so
    every reload keeps serving the sinks the server booted with. This is live on the
    existing fp8 broadcast path too, not only on a bf16 one.

    The parameter is padded to the platform's Q head count (``torch.full((padded_heads,),
    -inf)`` in ``vllm/models/deepseek_v4/attention.py``), which is why upstream writes a
    prefix rather than the whole tensor. Padding this rank's heads back up with ``-inf``,
    the parameter's own init value meaning no sink, makes it an ordinary full-parameter
    load, so ``load_numel`` reaches ``load_numel_total`` and no new loader contract is
    needed. The loader must be reached through ``param.weight_loader`` rather than
    attached to the parameter later, because ``initialize_layerwise_reload`` captures the
    original loader at the moment it wraps.

    Remove this patch when the pinned vLLM version loads ``attn_sink`` through a weight
    loader. 0.29.0 and vLLM main both still write the bare slice copy; the same fix
    appears only in vllm-project/vllm#54955, an open draft marked do-not-merge, so no
    release carries it. This covers the NVIDIA path only, and the ``amd`` and ``xpu``
    model files carry the same bare copy.
    """
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


def monkey_patch_nano_v3_reasoning_parser():
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

    class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
        def extract_reasoning(self, model_output, request):
            reasoning_content, final_content = super().extract_reasoning(model_output, request)
            chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

            if chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False and final_content is None:
                reasoning_content, final_content = final_content, reasoning_content

            return reasoning_content, final_content

    ReasoningParserManager.register_module("nano_v3", module=NanoV3ReasoningParser)


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
    ``MultiLoRAGroupedExperts.state_dict_for_adapter`` -- vLLM only consults that layout
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

    Copy of vLLM 0.24.0's ``LoRAModel.from_local_checkpoint`` with one change: the
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
        TensorizerConfig,
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
        tensorizer_config_dict: dict | None = None,
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

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_dir = tensorizer_config.tensorizer_dir
            if tensorizer_dir is None:
                raise ValueError("tensorizer_dir must be set in tensorizer config.")
            lora_tensor_path = os.path.join(tensorizer_dir, "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                device=device,
                **tensorizer_args.deserialization_kwargs,
            )
            check_unexpected_modules(tensors)

        elif os.path.isfile(lora_tensor_path):
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


# Monkeypatch TokenizeParams to fix overly conservative validation
def monkey_patch_tokenize_params_validation():
    """
    Patch TokenizeParams validation to only reject requests where the prompt
    itself exceeds max_model_len, not where prompt + max_tokens > max_model_len.

    Original behavior:
        - Rejects if prompt_len > (max_model_len - max_tokens)

    Patched behavior:
        - Only rejects if prompt_len > max_model_len
        - Lets the engine naturally cap generation at max_model_len
    """
    from vllm.exceptions import VLLMValidationError
    from vllm.renderers.params import TokenizeParams

    def _patched_token_len_check(self, tokenizer, tokens):
        """Only validate that prompt fits in max_model_len, not prompt+max_tokens"""
        if self.max_total_tokens is not None and len(tokens) > self.max_total_tokens:
            raise VLLMValidationError(
                f"The prompt is {len(tokens)} tokens, which exceeds the "
                f"model's maximum context length of {self.max_total_tokens} tokens. "
                f"Please reduce the length of the input prompt.",
                parameter="input_tokens",
                value=len(tokens),
            )
        return tokens

    def _patched_text_len_check(self, tokenizer, text):
        """Only validate text length against max_model_len, not max_input_tokens"""
        if self.max_total_tokens is None or tokenizer is None:
            return text

        if self.truncate_prompt_tokens is None:
            max_chars = self.max_total_tokens * tokenizer.max_chars_per_token
            if len(text) > max_chars:
                raise VLLMValidationError(
                    f"You passed {len(text)} input characters. "
                    f"However, the model's context length is only "
                    f"{self.max_total_tokens} tokens "
                    f"(at most {max_chars} characters). "
                    f"Please reduce the length of the input prompt.",
                    parameter="input_text",
                    value=len(text),
                )
        return text

    def _patched_get_encode_kwargs(self):
        """Use max_total_tokens (max_model_len) instead of max_input_tokens for HF tokenizer truncation.

        The original uses max_input_tokens (= max_model_len - max_tokens) + 1, which causes HuggingFace's
        tokenizer.encode() to left-truncate prompts before _token_len_check even runs.
        """
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_total_tokens
        elif max_length is None and self.max_total_tokens is not None:
            max_length = self.max_total_tokens + 1

        return dict(
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=self.add_special_tokens,
        )

    TokenizeParams._token_len_check = _patched_token_len_check
    TokenizeParams._text_len_check = _patched_text_len_check
    TokenizeParams.get_encode_kwargs = _patched_get_encode_kwargs


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
