import os

import torch
from vllm.triton_utils import tl, triton


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def transformers_v5_compat():
    """vLLM general plugin: patch transformers v5 config attrs that vLLM still expects.

    Registered as a ``vllm.general_plugins`` entry-point so it runs automatically
    in every vLLM process, including spawned workers.
    """
    from transformers import Qwen3VLMoeTextConfig

    if not hasattr(Qwen3VLMoeTextConfig, "tie_word_embeddings"):
        Qwen3VLMoeTextConfig.tie_word_embeddings = False

    _patch_qwen35_lora()
    _patch_lora_key_prefix()
    monkey_patch_deep_gemm_ep_scatter()
    monkey_patch_deep_gemm_silu_mul_quant_int64()
    monkey_patch_dp_engine_core_pause_resume_deadlock()
    monkey_patch_vllm_layerwise_reload_alias_buffers()


def monkey_patch_vllm_layerwise_reload_alias_buffers():
    # vLLM's layerwise reload materializes each buffer as an independent tensor
    # and then copies it back into the original kernel storage. When a buffer
    # aliases a parameter (e.g. NemotronH Mamba's mixer.conv_weights, a view of
    # mixer.conv1d.weight), the buffer copy stamps garbage into the parameter's
    # storage *after* the parameter has been correctly reloaded. Skip the copy
    # for any buffer that shares storage with a parameter; _place_kernel_tensors
    # re-registers the original view, which trivially reflects the parameter.
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader.reload import layerwise as reload_layerwise

    logger = init_logger(__name__)

    def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: reload_layerwise.LayerReloadingInfo):
        assert info.kernel_tensors is not None
        parameters, buffers = info.kernel_tensors
        param_storage_ptrs = {p.untyped_storage().data_ptr() for p in layer.parameters(recurse=True)}
        for name, param in parameters.items():
            param.data.copy_(getattr(layer, name))
        for name, buffer in buffers.items():
            if buffer.untyped_storage().data_ptr() in param_storage_ptrs:
                continue
            buffer.data.copy_(getattr(layer, name))

        reload_layerwise._place_kernel_tensors(layer, info)

    reload_layerwise._copy_and_restore_kernel_tensors = _copy_and_restore_kernel_tensors
    logger.warning("Enabled vLLM layerwise reload alias-buffer patch.")


@triton.jit
def _apply_expert_map_triton(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id


@triton.jit
def _fwd_kernel_ep_scatter_2_int64(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    output_tensor_stride0 = output_tensor_stride0.to(tl.int64)

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s,
            mask=mask_s,
        )

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = _apply_expert_map_triton(expert_id, expert_map)

            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index_i64 = dest_token_index.to(tl.int64)

                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )

                output_tensor_ptr = output_tensor + dest_token_index_i64 * output_tensor_stride0
                output_tensor_scale_ptr = output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s)


def _triton_ep_scatter_int64(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: torch.Tensor | None,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    from vllm.model_executor.layers.fused_moe import deep_gemm_utils

    block_e = 128
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]

    assert m_indices.shape[0] % block_e == 0
    assert expert_start_loc.shape[0] == num_experts

    deep_gemm_utils._fwd_kernel_ep_scatter_1[(num_experts,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_2_int64[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=recv_x_scale.shape[1],
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(recv_x_scale.shape[1]),
    )


def monkey_patch_deep_gemm_ep_scatter():
    # Temporary local carry of the upstream fix while it is under review:
    # issue: https://github.com/vllm-project/vllm/issues/39211
    # PR:    https://github.com/vllm-project/vllm/pull/39213
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe import deep_gemm_utils

    logger = init_logger(__name__)

    deep_gemm_utils.ep_scatter = torch.no_grad()(_triton_ep_scatter_int64)
    logger.warning("Enabled int64-addressing Triton patch for vLLM DeepGEMM ep_scatter.")


@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor_int64_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M: tl.int64,
    N: tl.int64,
    y_s_col_stride: tl.int64,
    eps,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = (pid_m * BLOCK_M).to(tl.int64)
    n_offset = (pid_n * BLOCK_N).to(tl.int64)
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset
    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


def _silu_mul_per_token_group_quant_fp8_colmajor_int64(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
):
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    group_size = 128
    assert input.ndim == 2
    if output is not None:
        assert output.ndim == 2
    assert input.size(0) % group_size == 0
    assert input.size(1) % (group_size * 2) == 0

    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()

    M, N = input.size()
    N_2 = N // 2

    fp8_dtype = current_platform.fp8_dtype()
    if output is None:
        output = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)

    output_scales = torch.empty(((N_2 // group_size), M), dtype=torch.float32, device=input.device).transpose(0, 1)

    block_m = 8
    block_n = group_size
    assert M % block_m == 0
    assert N_2 % block_n == 0

    finfo = torch.finfo(fp8_dtype)
    fp8_min = -224.0 if current_platform.is_fp8_fnuz() else finfo.min
    fp8_max = 224.0 if current_platform.is_fp8_fnuz() else finfo.max

    grid = (M // block_m, N_2 // block_n)
    _silu_mul_per_token_group_quant_fp8_colmajor_int64_kernel[grid](
        input,
        output,
        output_scales,
        M,
        N,
        output_scales.stride(-1),
        eps,
        fp8_min,
        fp8_max,
        use_ue8m0,
        group_size,
        block_m,
        block_n,
    )

    return output, output_scales


def monkey_patch_deep_gemm_silu_mul_quant_int64():
    # Temporary local carry for large DeepGEMM profile shapes whose row offsets
    # exceed signed int32 address arithmetic in vLLM's Triton kernel.
    import sys

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    logger = init_logger(__name__)

    fp8_utils.silu_mul_per_token_group_quant_fp8_colmajor = _silu_mul_per_token_group_quant_fp8_colmajor_int64

    deep_gemm_moe_module = sys.modules.get("vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe")
    if deep_gemm_moe_module is not None:
        deep_gemm_moe_module.silu_mul_per_token_group_quant_fp8_colmajor = (
            _silu_mul_per_token_group_quant_fp8_colmajor_int64
        )

    logger.warning("Enabled int64-addressing Triton patch for vLLM DeepGEMM SiLU/mul FP8 quant.")


def _patch_qwen35_lora():
    """Fix Qwen3.5 LoRA: align packed_modules_mapping with output_sizes.

    Qwen3.5's GDN layers use create_qkvz_proj with 4 output_sizes (q, k, v, z)
    but packed_modules_mapping only lists 2 entries, causing an IndexError
    during LoRA initialization.

    Also generalizes MergedColumnParallelLinearWithLoRA.can_replace_layer
    to accept any number of packed modules (not just 2), and generalizes
    MergedColumnParallelLinearWithShardedLoRA.slice_lora_a to handle N
    subloras instead of the hardcoded 2 (needed for fully_sharded_loras=True).

    Upstream: https://github.com/vllm-project/vllm/issues/36372
    """
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithShardedLoRA,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
    )

    qkvz_fix = ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"]

    Qwen3_5ForCausalLMBase.packed_modules_mapping["in_proj_qkvz"] = qkvz_fix
    Qwen3_5ForConditionalGeneration.packed_modules_mapping["in_proj_qkvz"] = qkvz_fix

    Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = False

    from vllm.lora.layers.utils import _not_fully_sharded_can_replace

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        return type(source_layer) is MergedColumnParallelLinear and len(packed_modules_list) == len(
            source_layer.output_sizes
        )

    MergedColumnParallelLinearWithLoRA.can_replace_layer = can_replace_layer

    def slice_lora_a(self, lora_a):
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        return [
            a[output_start_idx : output_start_idx + output_shard_size, :] if a is not None else None for a in lora_a
        ]

    MergedColumnParallelLinearWithShardedLoRA.slice_lora_a = slice_lora_a


def _patch_lora_key_prefix():
    """Patch vLLM's LoRA loading to handle keys without base_model.model. prefix.

    This is a copy of the upstream patch: https://github.com/vllm-project/vllm/pull/38522
    We can remove this patch once that PR makes it into a release.
    """
    from vllm.lora.lora_model import (
        LoRAModel,
        PEFTHelper,
        TensorizerConfig,
        WeightsMapper,
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
                # For standard MoE models the name ends in "...experts",
                # so expert_idx+1 yields "experts" which is in
                # expected_lora_modules.
                # For Qwen 3.5 MoE (and similar models) the expert index
                # is embedded: "...experts.N.down_proj".  Taking everything
                # after ".experts" gives "experts.N.down_proj" which is
                # never in the expected set even though "down_proj" is.
                # Qwen3-30B-A3B goes the other way: the expected set
                # contains the fully-qualified per-expert name
                # ("experts.N.down_proj") but not the bare suffix.
                # Accept either form.
                if ".experts" in module_name:
                    expert_suffix = module_name.split(".")[-1]
                    experts_qualified = "experts" + module_name.split(".experts", 1)[-1]
                    if expert_suffix not in expected_lora_modules and experts_qualified not in expected_lora_modules:
                        unexpected_modules.append(module_name)

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
            lora_tensor_path = os.path.join(tensorizer_config.tensorizer_dir, "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
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
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            lora_file_path = lora_bin_file_path if os.path.isfile(lora_bin_file_path) else lora_pt_file_path
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
            check_unexpected_modules(tensors)
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


# Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
# TODO: may be removable if we pass load_inplace=True (supported since vLLM 0.18, PR #31326)
def monkey_patch_load_lora_adapter():
    from http import HTTPStatus

    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.models.serving import (
        OpenAIServingModels,
        create_error_response,
    )
    from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
    from vllm.logger import init_logger
    from vllm.lora.request import LoRARequest

    logger = init_logger(__name__)

    async def _patched_load_lora_adapter(
        self: OpenAIServingModels, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        lora_name = request.lora_name
        env_load_inplace = _env_flag("PRIME_RL_LORA_LOAD_INPLACE")

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            lora_path = request.lora_path
            ## START PATCHED CODE
            existing_request = self.lora_requests.get(lora_name)
            request_load_inplace = bool(getattr(request, "load_inplace", False))
            load_inplace = request_load_inplace or env_load_inplace
            if existing_request is not None:
                unique_id = existing_request.lora_int_id
                if not load_inplace:
                    logger.warning(
                        "PrimeRL LoRA reload is reusing existing adapter id without load_inplace; "
                        "the worker cache may keep the old adapter resident. name=%s id=%s old_path=%s new_path=%s",
                        lora_name,
                        unique_id,
                        existing_request.lora_path,
                        lora_path,
                    )
            else:
                unique_id = self.lora_id_counter.inc(1)
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=unique_id,
                lora_path=lora_path,
                load_inplace=load_inplace,
            )
            logger.info(
                "PrimeRL LoRA load request: name=%s id=%s path=%s existed=%s request_load_inplace=%s "
                "env_load_inplace=%s effective_load_inplace=%s",
                lora_name,
                unique_id,
                lora_path,
                existing_request is not None,
                request_load_inplace,
                env_load_inplace,
                load_inplace,
            )
            ## END PATCHED CODE
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also preload it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(e):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND

                return create_error_response(message=str(e), err_type=error_type, status_code=status_code)

            self.lora_requests[lora_name] = lora_request
            logger.info("Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path)
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    OpenAIServingModels.load_lora_adapter = _patched_load_lora_adapter


# Monkeypatch LRUCacheWorkerLoRAManager to allow loading adapter inplace without doing it every request
# TODO: may be removable if we pass load_inplace=True (supported since vLLM 0.18, PR #31326)
def monkey_patch_LRUCacheWorkerLoRAManager():
    from vllm.logger import init_logger
    from vllm.lora.worker_manager import LoRARequest, LRUCacheLoRAModelManager, LRUCacheWorkerLoRAManager

    logger = init_logger(__name__)

    # The dunder is intended. It's a private method that we're patching.
    def _patched__apply_adapters(self: LRUCacheWorkerLoRAManager, lora_requests: set[LoRARequest]) -> None:
        loras_map = {lora_request.lora_int_id: lora_request for lora_request in lora_requests if lora_request}
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            ## START PATCHED CODE
            self.add_adapter(lora, force_load=False)
            ## END PATCHED CODE

    def _patched_add_adapter(
        self: LRUCacheWorkerLoRAManager, lora_request: LoRARequest, force_load: bool = True
    ) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        ## START PATCHED CODE
        already_loaded = lora_request.lora_int_id in self.list_adapters()
        load_inplace = bool(getattr(lora_request, "load_inplace", False))
        should_load = not already_loaded or force_load or load_inplace
        logger.info(
            "PrimeRL LoRA worker add_adapter: name=%s id=%s path=%s already_loaded=%s force_load=%s "
            "load_inplace=%s will_load=%s",
            lora_request.lora_name,
            lora_request.lora_int_id,
            lora_request.lora_path,
            already_loaded,
            force_load,
            load_inplace,
            should_load,
        )
        if should_load:
            ## END PATCHED CODE
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded lora adapters to very temporarily
            # exceed `--max-cpu-loras`.
            lora = self._load_adapter(lora_request)
            ## START PATCHED CODE
            self._adapter_manager.remove_adapter(lora.id)
            ## END PATCHED CODE

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            logger.warning(
                "PrimeRL LoRA worker skipped reload for already-loaded adapter. name=%s id=%s path=%s",
                lora_request.lora_name,
                lora_request.lora_int_id,
                lora_request.lora_path,
            )
            loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded

    LRUCacheWorkerLoRAManager._apply_adapters = _patched__apply_adapters
    LRUCacheWorkerLoRAManager.add_adapter = _patched_add_adapter


# Monkeypatch WorkerLoRAManager._load_adapter to skip the per-module regex
# warning loop. On wide MoE models (Qwen3.5-35B-A3B) it spends minutes
# recompiling regex patterns inside is_supported_lora_module — purely to emit
# logger.warning_once about modules that will be ignored. Adapter validity is
# already enforced by from_local_checkpoint, so dropping the warnings is safe.
def monkey_patch_skip_lora_module_warnings():
    from vllm.exceptions import LoRAAdapterNotFoundError
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.request import LoRARequest
    from vllm.lora.utils import get_adapter_absolute_path
    from vllm.lora.worker_manager import WorkerLoRAManager

    def _patched_load_adapter(self: WorkerLoRAManager, lora_request: LoRARequest) -> LoRAModel:
        try:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)
            expected_lora_modules = set(expected_lora_lst)
            lora_path = get_adapter_absolute_path(lora_request.lora_path)

            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                lora_request.tensorizer_config_dict,
            )
            peft_helper.validate_legal(self.lora_config)

            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            lora = self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                model_vocab_size=self.vocab_size,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=hf_to_vllm_mapper,
                skip_prefixes=lora_skip_prefixes,
            )
        except FileNotFoundError as e:
            raise LoRAAdapterNotFoundError(lora_request.lora_name, lora_request.lora_path) from e

        return lora

    WorkerLoRAManager._load_adapter = _patched_load_adapter


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
        Fix: recreate the gate in model dtype and remove the float32 cast.
        FusedMoE already has router_logits_dtype=float32, so routing precision
        is preserved inside the expert dispatch.

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
        from vllm.model_executor.layers.linear import ReplicatedLinear

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def _patched_forward(self, hidden_states):
        from vllm.distributed import tensor_model_parallel_all_reduce

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)

    MiniMaxM2MoE.__init__ = _patched_init
    MiniMaxM2MoE.forward = _patched_forward

    # --- Adapter key remapping (only read by vLLM's LoRA adapter loader) ---
    MiniMaxM2ForCausalLM.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.experts.": ".block_sparse_moe.experts.",
            ".gate_proj.": ".w1.",
            ".down_proj.": ".w2.",
            ".up_proj.": ".w3.",
        },
    )


def monkey_patch_harmony_stop_token_propagation():
    """Fix: vLLM doesn't merge harmony stop tokens into per-request SamplingParams.

    The harmony mode sets stop_token_ids (including <|call|> and <|return|>) in
    default_sampling_params at server init, but ChatCompletionRequest.to_sampling_params()
    ignores them, using only self.stop_token_ids (which defaults to []).

    Upstream: https://github.com/vllm-project/vllm/issues/22519
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    _original_to_sampling_params = ChatCompletionRequest.to_sampling_params

    def _patched_to_sampling_params(self, max_tokens, default_sampling_params):
        params = _original_to_sampling_params(self, max_tokens, default_sampling_params)
        # Merge harmony stop tokens from default_sampling_params
        default_stop_ids = default_sampling_params.get("stop_token_ids", [])
        if default_stop_ids:
            existing = set(params.stop_token_ids or [])
            merged = list(existing | set(default_stop_ids))
            params.stop_token_ids = merged
        return params

    ChatCompletionRequest.to_sampling_params = _patched_to_sampling_params


def monkey_patch_dp_engine_core_pause_resume_deadlock():
    """Fix DP pause/resume deadlocks around weight updates.

    Bug 1 (job 3756): while paused, START_DP_WAVE can wake idle ranks into the
    DP loop. Those ranks then run dummy batches and hit DP collectives while
    other ranks are still in NCCL weight transfer.

    Bug 2 (jobs 3769/3771): resume ties the DP running state to local
    unfinished requests, but the DP wave state is global. Ranks with no local
    work still need to re-enter the loop so they can participate in the same
    DP collectives as ranks that are resuming remote-KV or decode work.

    Fix:
    - ignore START_DP_WAVE wakeups while paused
    - on resume, wake every DP rank and force an immediate global unfinished
      sync instead of waiting for the normal 32-step cadence

    This keeps the upstream pause-side fix from
    https://github.com/vllm-project/vllm/pull/37024 and extends it with the
    resume-side wave-state fix.
    """
    from vllm.config import ParallelConfig
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
    from vllm.v1.request import Request

    _base_add_request = EngineCore.add_request
    _base_handle_client_request = EngineCoreProc._handle_client_request
    _base_resume_scheduler = DPEngineCoreProc.resume_scheduler

    def _patched_add_request(self, request: Request, request_wave: int = 0):
        _base_add_request(self, request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running and self.scheduler.pause_state == PauseState.UNPAUSED:
                self.engines_running = True
                self.output_queue.put_nowait((-1, EngineCoreOutputs(start_wave=self.current_wave)))

    def _patched_handle_client_request(self, request_type, request):
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and new_wave >= self.current_wave:
                self.current_wave = new_wave
                if not self.engines_running and self.scheduler.pause_state == PauseState.UNPAUSED:
                    self.engines_running = True
        else:
            _base_handle_client_request(self, request_type, request)

    def _patched_resume_scheduler(self):
        was_paused = self.scheduler.pause_state != PauseState.UNPAUSED
        _base_resume_scheduler(self)
        if was_paused:
            self.engines_running = True
            self._force_dp_running_state_sync = True

    def _patched_has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        self.step_counter += 1
        if getattr(self, "_force_dp_running_state_sync", False):
            self._force_dp_running_state_sync = False
            return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        if self.step_counter % 32 != 0:
            return True
        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    DPEngineCoreProc.add_request = _patched_add_request
    DPEngineCoreProc._handle_client_request = _patched_handle_client_request
    DPEngineCoreProc.resume_scheduler = _patched_resume_scheduler
    DPEngineCoreProc._has_global_unfinished_reqs = _patched_has_global_unfinished_reqs


def monkey_patch_no_moe_lora():
    """This disables LoRA for MoE layers and makes them pick better kernels.

    Otherwise, the oracle will always try to pick TritonExperts.
    For blackwells, we want TRTLLMFlashInfer.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, logger

    def _patched__post_init__(self: FusedMoEConfig):
        if self.dp_size > 1:
            logger.debug_once("Using FusedMoEConfig::max_num_tokens=%d", self.max_num_tokens)

        assert self.max_num_tokens > 0

        if self.router_logits_dtype is None:
            self.router_logits_dtype = self.in_dtype

        if self.hidden_dim_unpadded is None:
            self.hidden_dim_unpadded = self.hidden_dim
        if self.intermediate_size_per_partition_unpadded is None:
            self.intermediate_size_per_partition_unpadded = self.intermediate_size_per_partition

        # Disable LoRA for MoE layers
        self.is_lora_enabled = False

    FusedMoEConfig.__post_init__ = _patched__post_init__


def monkey_patch_fp32_lm_head():
    """Run the lm_head projection in fp32, via a native bf16xbf16 -> fp32 GEMM.

    Uses ``torch.mm(..., out_dtype=torch.float32)`` (PyTorch >= 2.10) so the
    matmul accumulates and emits fp32 directly without zero-padding the bf16
    operands or maintaining a separate fp32 weight copy. This avoids the
    epilogue truncation to bf16 that `F.linear(bf16, bf16)` does, which is
    where lm_head precision actually leaks before the sampler's softmax.

    Activated by setting ``additional_config["fp32_lm_head"] = True`` on the
    vLLM namespace; the launcher does this when ``inference.enable_fp32_lm_head``
    is set. The flag is captured once on ``LogitsProcessor.__init__`` (where
    vLLM guarantees a ``set_current_vllm_config()`` context) and stored on the
    instance — reading it from ``_get_logits`` during serving doesn't work
    because vLLM doesn't keep the context set during forwards.

    Tracks vllm-project/vllm#24567 (which uses the operand-upcast approach).
    Per @Jackmin801 on PR #2438, native ``out_dtype=fp32`` mm is more efficient
    and just as correct.
    """
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.logger import init_logger
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    logger = init_logger(__name__)

    _original_init = LogitsProcessor.__init__
    _original_get_logits = LogitsProcessor._get_logits

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        vllm_config = get_current_vllm_config()
        additional_config = vllm_config.additional_config or {}
        self._fp32_lm_head_enabled = additional_config.get("fp32_lm_head", False)
        if self._fp32_lm_head_enabled:
            logger.warning("fp32 lm_head ENABLED for this LogitsProcessor instance.")

    def _patched_get_logits(self, hidden_states, lm_head, embedding_bias):
        if not getattr(self, "_fp32_lm_head_enabled", False):
            return _original_get_logits(self, hidden_states, lm_head, embedding_bias)

        # Native bf16xbf16 -> fp32 GEMM. torch.mm requires 2D inputs; vLLM v1's
        # generative path passes 2D [num_tokens, hidden_size] hidden_states, but
        # flatten defensively in case some future caller passes 3D.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = torch.mm(flat, lm_head.weight.t(), out_dtype=torch.float32)
        if embedding_bias is not None:
            logits = logits + embedding_bias.to(torch.float32)
        if hidden_states.dim() > 2:
            logits = logits.reshape(*hidden_states.shape[:-1], -1)

        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    LogitsProcessor.__init__ = _patched_init
    LogitsProcessor._get_logits = _patched_get_logits
    logger.info("Installed fp32 lm_head patch (native out_dtype=fp32 mm).")
