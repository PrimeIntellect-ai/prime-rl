import torch


def apply_shared_vllm_patches():
    """vLLM general plugin: prime-rl patches that must run in every vLLM process.

    Registered as a ``vllm.general_plugins`` entry-point so it runs automatically
    in every vLLM process, including spawned workers. Note vLLM swallows plugin
    load failures (``load_plugins_by_group`` logs and continues), so a broken
    entry-point target silently skips ALL of these patches.
    """
    from prime_rl.inference.vllm.gpt_oss_weight_loading import patch_gpt_oss_weight_loading

    patch_gpt_oss_weight_loading()
    _patch_lora_key_prefix()
    _patch_qwen35_moe_lora_format()
    monkey_patch_nano_v3_reasoning_parser()
    monkey_patch_minimax_m2_think_end_passthrough()
    monkey_patch_return_routed_experts_with_nixl_connector()
    monkey_patch_kv_xfer_finished_tolerate_freed()
    monkey_patch_online_fp8_parameter_cast()
    monkey_patch_deepseek_v4_allowed_layer_types()
    monkey_patch_deepseek_v4_per_layer_rope()
    monkey_patch_diag_stash_bf16_o_proj_weight()
    monkey_patch_fp8_ue8m0_weight_scales()
    monkey_patch_deepseek_v4_bf16_o_proj()
    monkey_patch_deepseek_v4_attn_sink_loading()


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


def monkey_patch_deepseek_v4_bf16_o_proj():
    """Run DeepSeek V4's output projection in bf16 when the weights are not FP8.

    vLLM's DeepSeek V4 output projection is FP8-only. All three CUDA attention
    classes (``DeepseekV4FlashMLAAttention``,
    ``DeepseekV4FlashInferMLAAttention``, ``DeepseekV4FlashInferSM120Attention``)
    implement ``_o_proj`` by calling ``deep_gemm_fp8_o_proj``
    (``vllm/models/deepseek_v4/nvidia/ops/o_proj.py``), which dereferences
    ``wo_a.weight_scale``, falling back to ``wo_a.weight_scale_inv``, and hands
    the result to DeepGEMM's ``fp8_einsum``. There is no branch on quant config
    or weight dtype, and ``_o_proj`` is an ``@abstractmethod`` dispatched purely
    by platform subclass, so no attention backend routes around it. A bf16
    checkpoint has neither attribute and therefore fails with
    ``AttributeError: 'ColumnParallelLinear' object has no attribute
    'weight_scale_inv'``.

    Quantizing is not an answer, because nothing quantizes by default:
    ``[inference.vllm] quantization`` is unset, which leaves vLLM to infer the
    scheme from the checkpoint. bf16 DeepSeek V4 checkpoints are a first-class
    artifact here, written by ``tools/convert_fp8_to_bf16.py``,
    ``tools/convert_dcp_to_bf16.py``, and trainer exports, and they carry no
    scales for the op to find. Passing ``--quantization fp8_per_block`` does
    produce a correct ``weight_scale_inv`` and is a genuine alternative on
    hardware whose DeepGEMM build has block-scaled FP8 kernels, but it changes
    the served numerics, so it is a deployment decision rather than a fix for
    serving the bf16 weights as given.

    Only the unquantized case is redirected: an FP8 weight keeps using vLLM's own
    kernel, so a real FP8-serialized checkpoint behaves exactly as before. The
    condition is the weight dtype rather than the presence of
    ``weight_scale_inv``, so a checkpoint that *is* quantized but whose scales
    ended up in the wrong layout still fails loudly instead of being silently
    rerouted.

    The replacement mirrors the Triton kernel's arithmetic
    (``vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:95-105``)
    without the quantization step. The trailing ``rope_dim`` channels of each head
    carry an inverse RoPE over interleaved pairs, with ``cos_sin_cache`` holding
    ``rope_dim // 2`` cosines followed by the matching sines:

        y[even] = x[even] * cos + x[odd] * sin
        y[odd]  = x[odd]  * cos - x[even] * sin

    The remaining ``nope_dim`` channels pass through. The projection is the same
    grouped contraction ``fp8_einsum`` performs, with the group axis recovered by
    reshaping: head ``g * heads_per_group + i`` belongs to group ``g`` at offset
    ``i * head_dim``, matching the kernel's own
    ``g = pid_gh // heads_per_group`` indexing.

    Rotation is computed in fp32 and the contraction runs in bf16, whose CUDA
    matmuls accumulate in fp32. That is strictly more accurate than the FP8 path
    it replaces, and it avoids materializing an fp32 copy of the attention output.
    """
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse, flashmla
    from vllm.models.deepseek_v4.nvidia.ops import o_proj as o_proj_module

    original_o_proj = o_proj_module.deep_gemm_fp8_o_proj
    if getattr(original_o_proj, "_prime_rl_has_bf16_fallback", False):
        return

    def _inverse_rope(o, positions, cos_sin_cache, rope_dim):
        num_tokens, num_heads, head_dim = o.shape
        half_rope = rope_dim // 2
        cos_sin = cos_sin_cache[positions].float()
        cos = cos_sin[:, :half_rope].view(num_tokens, 1, half_rope)
        sin = cos_sin[:, half_rope:].view(num_tokens, 1, half_rope)

        rotated = o[:, :, head_dim - rope_dim :].float().reshape(num_tokens, num_heads, half_rope, 2)
        even, odd = rotated[..., 0], rotated[..., 1]
        pairs = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)

        out = o.clone()
        out[:, :, head_dim - rope_dim :] = pairs.reshape(num_tokens, num_heads, rope_dim).to(o.dtype)
        return out

    def _patched_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups, heads_per_group, **kwargs):
        if wo_a.weight.dtype == torch.float8_e4m3fn:
            return original_o_proj(
                o,
                positions,
                cos_sin_cache,
                wo_a,
                wo_b,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                **kwargs,
            )

        x = _inverse_rope(o, positions, cos_sin_cache, kwargs["rope_dim"])
        x = x.reshape(o.shape[0], n_groups, heads_per_group * o.shape[-1])
        weight = wo_a.weight.view(n_groups, kwargs["o_lora_rank"], -1)
        z = torch.einsum("tgr,gdr->tgd", x, weight)
        return wo_b(z.flatten(1))

    _patched_o_proj._prime_rl_has_bf16_fallback = True
    o_proj_module.deep_gemm_fp8_o_proj = _patched_o_proj

    # Both attention modules did `from ...ops.o_proj import deep_gemm_fp8_o_proj` at
    # import time, so patching the defining module alone would not reach the names
    # their `_o_proj` methods actually call.
    flashmla.deep_gemm_fp8_o_proj = _patched_o_proj
    flashinfer_sparse.deep_gemm_fp8_o_proj = _patched_o_proj


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


def monkey_patch_diag_stash_bf16_o_proj_weight():
    """Measurement only, off unless ``PRIME_DIAG_BF16_OPROJ=1``: keep ``wo_a`` in bf16 under online FP8.

    DeepSeek V4's output projection is the prime suspect for a precision floor
    under FP8. ``deep_gemm_fp8_o_proj`` quantizes the attention output with
    ``fused_inv_rope_fp8_quant``, whose Triton kernel uses power-of-two (UE8M0)
    activation scales unconditionally and so ignores ``VLLM_USE_DEEP_GEMM_E8M0=0``,
    and then contracts against the blockwise-quantized ``wo_a`` weight with
    DeepGEMM's grouped ``fp8_einsum``. This diagnostic removes both, leaving the
    rest of the model on the FP8 path, so the o_proj's contribution to the gap
    against a bf16 server can be read off directly.

    Dequantizing inside the forward is not viable: by then ``wo_a.weight`` is a
    3D FP8 tensor and ``wo_a.weight_scale_inv`` has been through
    ``transform_sf_into_required_layout``. So take the copy before it is
    destroyed. ``Fp8PerBlockOnlineLinearMethod.process_weights_after_loading``
    runs with ``layer.weight`` still the loaded bf16 tensor, and ``is_bmm`` marks
    exactly the layer this applies to (``vllm/models/deepseek_v4/attention.py``
    sets it on ``wo_a`` and nowhere else). ``monkey_patch_deepseek_v4_bf16_o_proj``
    then prefers the stashed copy over ``wo_a.weight``.

    Layerwise reload deletes and re-registers the weight on every weight update
    and ``_layerwise_process`` drops the
    ``_already_called_process_weights_after_loading`` sentinel before re-running
    the quantization, so the stash is refreshed from the newly received weights
    rather than going stale.

    ``wo_b`` is untouched and still runs its FP8 kernel, so what this isolates is
    the activation quantization plus the ``wo_a`` contraction, not the whole
    output projection.
    """
    import os

    if os.environ.get("PRIME_DIAG_BF16_OPROJ") != "1":
        return

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod

    logger = init_logger(_DIAG_LOGGER_NAME)
    original_process = Fp8PerBlockOnlineLinearMethod.process_weights_after_loading
    if getattr(original_process, "_prime_diag_stashes_bf16_weight", False):
        return

    def process_weights_after_loading(self, layer):
        already_processed = getattr(layer, "_already_called_process_weights_after_loading", False)
        if getattr(layer, "is_bmm", False) and not already_processed:
            setattr(layer, _DIAG_BF16_WEIGHT_ATTR, layer.weight.data.clone().to(torch.bfloat16))
        return original_process(self, layer)

    process_weights_after_loading._prime_diag_stashes_bf16_weight = True
    Fp8PerBlockOnlineLinearMethod.process_weights_after_loading = process_weights_after_loading
    logger.info(
        "PRIME_DIAG_BF16_OPROJ=1: stashing a bf16 copy of every online-FP8 is_bmm weight for the bf16 o_proj path."
    )


_FAKE_QUANT_BLOCK_SIZE = [128, 128]

# DeepSeek V4 builds these `LinearBase` modules with `quant_config=None` (the MoE router
# gate through `GateLinear`), so online FP8 leaves them in bf16 and so must this.
_FAKE_QUANT_EXCLUDED_LINEARS = frozenset({"gate", "fused_wkv_wgate", "weights_proj"})


def _fake_quantize_block_fp8_(weight: torch.Tensor) -> None:
    """Round-trip a 2D weight through 128x128 block FP8 in place."""
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    block_m, block_n = _FAKE_QUANT_BLOCK_SIZE
    rows, cols = weight.shape
    quantized, scale = per_block_cast_to_fp8(weight, block_size=_FAKE_QUANT_BLOCK_SIZE, use_ue8m0=False)
    block_scale = scale.repeat_interleave(block_m, dim=0)[:rows].repeat_interleave(block_n, dim=1)[:, :cols]
    weight.copy_(quantized.to(torch.float32) * block_scale)


def _fake_quantize_model_weights(model, ignore: re.Pattern[str] | None = None) -> int:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase

    quantized_tensors = 0
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in _FAKE_QUANT_EXCLUDED_LINEARS:
            continue
        if ignore is not None and ignore.search(name):
            continue
        if isinstance(module, LinearBase):
            _fake_quantize_block_fp8_(module.weight.data)
            quantized_tensors += 1
        elif isinstance(module, RoutedExperts):
            for experts in (module.w13_weight, module.w2_weight):
                for expert in range(experts.shape[0]):
                    _fake_quantize_block_fp8_(experts.data[expert])
                    quantized_tensors += 1
    return quantized_tensors


def monkey_patch_diag_fake_quant_weights():
    """Measurement only, off unless ``PRIME_DIAG_FAKE_QUANT_WEIGHTS=1``: serve bf16 weights that FP8 rounded.

    Pair this with a **bf16** server config, i.e. ``[inference.vllm] quantization``
    unset. Every weight the FP8 path would have quantized is round-tripped
    ``bf16 -> per_block_cast_to_fp8 -> dequantize -> bf16`` and then served by the
    unmodified bf16 kernel stack. What is left is pure weight quantization error,
    with no activation quantization and no change of kernel, which separates the
    three contributions to an FP8 rollout gap.

    The round trip calls vLLM's own ``per_block_cast_to_fp8``, so the arithmetic
    matches production exactly, including the ``1e-4`` amax clamp and the multiply
    by the reciprocal scale rather than a divide. Dequantization multiplies the
    e4m3 values back by their fp32 block scale. The product is then rounded to
    bf16, which the real FP8 path does not do (it applies fp32 scales in the GEMM
    epilogue); that adds a relative error of about ``2**-9``, roughly thirty times
    smaller than e4m3's own, so it does not move the measurement.

    Scope follows ``OnlineQuantizationConfig.get_quant_method``: ``LinearBase``
    weights and routed-expert ``w13``/``w2``, nothing else. ``lm_head`` and
    ``embed_tokens`` are ``VocabParallelEmbedding``, not ``LinearBase``, and are
    out for that reason as well as prime-rl's ``head_dtype=float32``. The three
    DeepSeek V4 linears that online FP8 skips are listed in
    ``_FAKE_QUANT_EXCLUDED_LINEARS``. ``PRIME_DIAG_FAKE_QUANT_IGNORE`` takes a regex
    that is searched against each module name and skips matches, mirroring the
    production ``quantization_config.ignore`` list so the fake-quant scope can be
    aligned with it or bisected by module family.

    Both entry points matter. ``BaseModelLoader.load_model`` covers the cold start,
    which ``finalize_layerwise_processing`` does not, since the loader only calls
    it when the model has an online-quant method and a bf16 model has none.
    ``finalize_layerwise_processing`` covers every ``/update_weights`` reload, which
    restores fresh bf16 weights and would otherwise silently stop the diagnostic
    after the first update. Every reload engine reaches it through
    ``finalize_layerwise_reload``, which resolves it as a module global at call
    time, so patching the defining module is enough. Weights are rewritten in
    place so cudagraph references survive.
    """
    import os

    if os.environ.get("PRIME_DIAG_FAKE_QUANT_WEIGHTS") != "1":
        return

    from vllm.logger import init_logger
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.model_loader.reload import layerwise

    logger = init_logger(_DIAG_LOGGER_NAME)
    ignore_pattern = os.environ.get("PRIME_DIAG_FAKE_QUANT_IGNORE")
    ignore = re.compile(ignore_pattern) if ignore_pattern else None
    original_load_model = base_loader.BaseModelLoader.load_model
    if getattr(original_load_model, "_prime_diag_fake_quants_weights", False):
        return

    def _fake_quantize_and_log(model, stage):
        logger.info(
            "PRIME_DIAG_FAKE_QUANT_WEIGHTS=1: round-tripped %d weight tensors through 128x128 block FP8 after %s (ignore=%r).",
            _fake_quantize_model_weights(model, ignore),
            stage,
            ignore_pattern,
        )

    def load_model(self, *args, **kwargs):
        model = original_load_model(self, *args, **kwargs)
        _fake_quantize_and_log(model, "the initial load")
        return model

    original_finalize = layerwise.finalize_layerwise_processing

    def finalize_layerwise_processing(model, model_config):
        original_finalize(model, model_config)
        _fake_quantize_and_log(model, "a weight reload")

    load_model._prime_diag_fake_quants_weights = True
    base_loader.BaseModelLoader.load_model = load_model
    layerwise.finalize_layerwise_processing = finalize_layerwise_processing
    logger.info("PRIME_DIAG_FAKE_QUANT_WEIGHTS=1: serving bf16 weights round-tripped through block FP8.")


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


def monkey_patch_return_routed_experts_with_nixl_connector():
    from vllm.config.vllm import VllmConfig
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    original_post_init = VllmConfig.__post_init__

    if getattr(original_post_init, "_prime_rl_allows_nixl_routed_experts", False):
        return

    def _is_nixl_routed_experts_pd_config(config: VllmConfig) -> bool:
        kv_transfer_config = config.kv_transfer_config
        return (
            config.model_config is not None
            and config.model_config.enable_return_routed_experts
            and kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "NixlConnector"
            and kv_transfer_config.is_kv_transfer_instance
        )

    def _post_init(config: VllmConfig):
        if not _is_nixl_routed_experts_pd_config(config):
            return original_post_init(config)

        if config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError("--enable-return-routed-experts is incompatible with pipeline parallelism (PP > 1).")
        if config.use_v2_model_runner:
            raise ValueError(
                "Routed-expert capture with NIXL requires the V1 model runner. Set VLLM_USE_V2_MODEL_RUNNER=0."
            )

        # vLLM rejects every KV connector, but our P/D path uses NIXL and
        # stitches prefill/decode routed experts in the router. CPU KV offload
        # remains rejected by prime-rl config validation.
        config.model_config.enable_return_routed_experts = False
        try:
            return original_post_init(config)
        finally:
            config.model_config.enable_return_routed_experts = True

    _post_init._prime_rl_allows_nixl_routed_experts = True
    VllmConfig.__post_init__ = _post_init
    logger.warning("Enabled vLLM routed-experts capture with NIXL connector patch.")


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
                # Case for expert lora weights.
                ## START PATCHED CODE (upstream only accepts the qualified form)
                if ".experts" in module_name:
                    expert_suffix = module_name.split(".")[-1]
                    experts_qualified = "experts" + module_name.split(".experts", 1)[-1]
                    if expert_suffix not in expected_lora_modules and experts_qualified not in expected_lora_modules:
                        unexpected_modules.append(module_name)
                ## END PATCHED CODE

                elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
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
    to apply unconditionally (verified with non-LoRA runs). We apply them at
    import time because the worker __init__ runs before the vLLM config is
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


_DEEPSEEK_V4_YARN_ROPE_TYPES = frozenset({"yarn", "deepseek_yarn", "deepseek_llama_scaling"})


def _deepseek_v4_rope_parameters(rope_parameters, *, compress_ratio, max_position_embeddings):
    """Flat, single-layer rope parameters for `build_deepseek_v4_rope`.

    Handles both on-disk schemas. The real checkpoint ships a flat legacy `rope_scaling`, which
    vLLM's config shim normalizes in place. HF's own `DeepseekV4Config` nests the parameters
    under `main`/`compress` rope-type labels instead, and so does this repo's port. vLLM's shim
    cannot read that: `is_rope_parameters_nested` only recognizes nesting by *attention* layer
    type, so transformers leaves the sub-dicts alone and injects a top-level
    `rope_type="default"` beside them. Upstream then builds unscaled RoPE for every layer and
    hands `get_rope` a cache key containing a dict, which is unhashable. Selecting the branch
    this layer wants resolves both. Upstream picks the base from `config.rope_theta` /
    `config.compress_rope_theta` on its own either way.
    """
    rope_parameters = rope_parameters if isinstance(rope_parameters, dict) else {}
    # `compress_ratio` is already clamped to >= 1 by the caller, so <= 1 is "no compressor",
    # i.e. the reference's `else` branch. vLLM lumps the MTP layers in here too.
    is_sliding = compress_ratio <= 1

    if any(isinstance(value, dict) for value in rope_parameters.values()):
        label = "main" if is_sliding else "compress"
        branch = rope_parameters.get(label)
        if not isinstance(branch, dict):
            raise ValueError(
                f"DeepSeek V4 rope_parameters is nested by rope type but has no {label!r} "
                f"sub-dict; found {sorted(rope_parameters)}."
            )
        parameters = dict(branch)
    else:
        parameters = dict(rope_parameters)

    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    if is_sliding or rope_type not in _DEEPSEEK_V4_YARN_ROPE_TYPES:
        # Plain RoPE, written as YaRN with `factor=1` rather than as `rope_type="default"`.
        # The table is the same either way, since `_compute_inv_freq`'s interpolated and
        # extrapolated frequencies coincide at `factor=1` and `yarn_get_mscale` returns 1.0,
        # but `"default"` would route `get_rope` to a plain `RotaryEmbedding`, whose cache
        # follows the ambient dtype and so fails the fp32 assertion in
        # `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`.
        # `original_max_position_embeddings` is widened because the cache is
        # `original_max * factor` rows and must still span the context.
        rope_type = "yarn"
        parameters["factor"] = 1
        parameters["original_max_position_embeddings"] = max_position_embeddings
    parameters["rope_type"] = rope_type
    return parameters


def monkey_patch_deepseek_v4_per_layer_rope():
    """Give each DeepSeek V4 layer the RoPE its reference implementation uses.

    DeepSeek ships the reference inside the checkpoint we serve, at
    https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference. Its
    `model.py:481-485` picks the frequencies per layer:

        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta

    and `precompute_freqs_cis` (`model.py:206-235`) applies the NTK-by-parts ramp only when
    `original_seq_len > 0`. So a pure sliding-window layer gets plain RoPE at `rope_theta` and
    every compressed layer gets YaRN at `compress_rope_theta`. HF's `DeepseekV4Config` splits its
    `rope_parameters` into `main`/`compress` for the same reason, and so does this repo's trainer.

    `vllm/models/deepseek_v4/common/rope.py`'s `build_deepseek_v4_rope` branches the base on
    `compress_ratio` but never the scaling, so on the real checkpoint it hands YaRN to layers 0
    and 1 as well. No config value can correct that: `rope_type` is one global field there.
    So this patch resolves `rope_parameters` per layer instead, which also gives upstream a
    schema and a dtype it can handle; see `_deepseek_v4_rope_parameters` above.

    Measured against the reference's own frequencies, at the test's `factor=16`,
    `original_max_position_embeddings=4096` geometry, this takes the sliding layers from 5.1e-03
    max absolute error to 6e-08, the float32 residual the compressed layers already have. The
    rope tests in `tests/unit/train/models/test_deepseek_v4.py` are that comparison, over both
    on-disk schemas and both scaled and unscaled configs.

    Note that nothing in vLLM's DeepSeek V4 calls the module's `forward`; every consumer reads
    `cos_sin_cache` and hands it to a fused kernel. The class choice is therefore about the
    cache's dtype and row count, not about which channels the module would rotate.

    Remove this patch once the pin includes vllm-project/vllm#54815, which ships in 0.29.1.
    """
    from vllm.models.deepseek_v4.common import rope as dsv4_rope

    original_build = dsv4_rope.build_deepseek_v4_rope

    if getattr(original_build, "_prime_rl_matches_deepseek_reference", False):
        return

    def _patched_build(config, *, head_dim, rope_head_dim, max_position_embeddings, compress_ratio):
        # Swapping a copy in also spares `config.rope_parameters` upstream's in-place mutation,
        # which otherwise leaves the last-built layer's `rope_theta` on the shared config.
        original_parameters = config.rope_parameters
        config.rope_parameters = _deepseek_v4_rope_parameters(
            original_parameters,
            compress_ratio=compress_ratio,
            max_position_embeddings=max_position_embeddings,
        )
        try:
            return original_build(
                config,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                compress_ratio=compress_ratio,
            )
        finally:
            config.rope_parameters = original_parameters

    _patched_build._prime_rl_matches_deepseek_reference = True
    dsv4_rope.build_deepseek_v4_rope = _patched_build

    # `attention.py` imports the builder into its own namespace, so the defining module alone
    # would not reach the already-bound reference.
    from vllm.models.deepseek_v4 import attention as dsv4_attention

    dsv4_attention.build_deepseek_v4_rope = _patched_build
