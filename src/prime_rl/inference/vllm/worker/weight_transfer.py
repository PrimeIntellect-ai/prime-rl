from typing import Generator, Iterable

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


def load_weights_checkpoint_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    vllm_config,
) -> None:
    logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
    device = next(model.parameters()).device
    with torch.device(device), set_current_vllm_config(vllm_config):
        initialize_layerwise_reload(model)
        model.load_weights(state_iter)  # type: ignore
        finalize_layerwise_reload(model, model_config)


def _invert_logical_to_physical_map(logical_to_physical_map: torch.Tensor, num_physical_experts: int) -> torch.Tensor:
    """Build a physical expert -> logical expert map from vLLM EPLB state."""
    physical_to_logical = torch.full(
        (num_physical_experts,),
        -1,
        dtype=torch.long,
        device=logical_to_physical_map.device,
    )
    logical_indices = torch.arange(
        logical_to_physical_map.shape[0],
        dtype=torch.long,
        device=logical_to_physical_map.device,
    )[:, None].expand_as(logical_to_physical_map)
    physical_indices = logical_to_physical_map.to(torch.long)
    invalid = (physical_indices < -1) | (physical_indices >= num_physical_experts)
    if invalid.any():
        invalid_indices = physical_indices[invalid].unique().tolist()
        raise ValueError(f"EPLB maps to invalid physical experts: {invalid_indices}")

    valid = physical_indices >= 0
    physical_to_logical[physical_indices[valid]] = logical_indices[valid]
    return physical_to_logical


def _build_expert_source_indices(module) -> torch.Tensor | None:
    if module._expert_map is None:
        return None

    physical_indices = torch.where(module._expert_map >= 0)[0]
    local_indices = module._expert_map[physical_indices]
    physical_indices = physical_indices[local_indices.argsort()]

    eplb_layer_state = getattr(module, "eplb_state", None)
    logical_to_physical_map = getattr(eplb_layer_state, "logical_to_physical_map", None)
    if logical_to_physical_map is None:
        return physical_indices

    physical_to_logical = _invert_logical_to_physical_map(logical_to_physical_map, module.global_num_experts)
    logical_indices = physical_to_logical[physical_indices.to(physical_to_logical.device)]
    if (logical_indices < 0).any():
        missing = physical_indices[(logical_indices < 0).to(physical_indices.device)].tolist()
        raise ValueError(f"EPLB has no logical mapping for local physical experts: {missing}")

    return logical_indices.to(physical_indices.device)


def build_expert_map(model: Module) -> dict[str, torch.Tensor]:
    """Map FusedMoE module names to source expert indices local to this worker."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    source_indices_by_module: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        source_indices = _build_expert_source_indices(module)
        if source_indices is None:
            continue
        source_indices_by_module[module_name] = source_indices
    return source_indices_by_module


def _try_e8m0_scale_conversion(
    name: str,
    received_scale: torch.Tensor,
    param: torch.Tensor,
    params: dict[str, torch.Tensor],
    updated_weights: set[str],
) -> bool:
    """Convert a trainer-format blockwise FP8 scale to vLLM's E8M0 TMA layout.

    When DeepGEMM E8M0 is active (GB200 default), vLLM stores weight_scale_inv
    tensors in a TMA-aligned [N, K/512] packed UE8M0 layout.  The trainer sends
    [N/128, K/128] float32 blockwise scales.

    This delegates to vLLM's own unified helper
    ``deepgemm_post_process_fp8_weight_block``, which is the *exact* call vLLM
    invokes at initial model load (via ``DeepGemmFp8BlockScaledMMKernel`` for
    dense linears and ``prepare_fp8_moe_layer_for_deepgemm`` for MoE).  The
    helper:

      1. Re-quantises the FP8 weight to UE8M0 (power-of-two) scales in-place
         (``requant_weight_ue8m0_inplace``).
      2. Repacks the scale tensor to the TMA-aligned ``[..., N, K/512]`` layout
         (``transform_sf_into_required_layout`` with recipe ``(1, 128, 128)``).
      3. Handles 2D (dense) vs 3D (MoE) dispatch internally.

    Reusing vLLM's wrapper guarantees the broadcast path produces the same
    in-memory layout as vLLM's own startup load path — drift-free by
    construction.

    Returns True on success, False if conversion is not applicable or fails.
    The caller is responsible for raising shape-mismatch errors for False returns.

    ORDERING REQUIREMENT: the corresponding FP8 weight param must have been
    updated (copied from the received tensor) before this function is called,
    i.e. the weight must appear before its weight_scale_inv in the state iterator.
    """
    # Only handle weight_scale_inv tensors.
    if not name.endswith("weight_scale_inv"):
        return False

    # Derive the corresponding weight parameter name.
    # e.g. "...qkv_proj.weight_scale_inv" -> "...qkv_proj.weight"
    #      "...w13_weight_scale_inv"      -> "...w13_weight"
    weight_name = name[: -len("weight_scale_inv")] + "weight"
    if weight_name not in params:
        return False

    # The weight must have been updated in this broadcast pass so that
    # requant_weight_ue8m0_inplace dequantises the *new* FP8 values.
    if weight_name not in updated_weights:
        logger.warning(
            "E8M0 scale conversion for %s: weight %s was not updated in this pass "
            "(scale arrived before weight). Skipping conversion.",
            name,
            weight_name,
        )
        return False

    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
        )
    except ImportError as exc:
        logger.debug("E8M0 conversion unavailable: %s", exc)
        return False

    weight_param = params[weight_name]

    try:
        # Single unified call: re-quantises weight in-place with UE8M0 scales
        # AND repacks the scale tensor to the TMA-aligned [..., N, K/512] layout
        # that vLLM stores.  Handles 2D (dense) and 3D (MoE) dispatch internally.
        # The first return value is the same underlying storage as weight_param
        # (modified in-place); we only need the new scale tensor.
        _wq, dg_scale = deepgemm_post_process_fp8_weight_block(
            wq=weight_param,
            ws=received_scale,
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )
        param.copy_(dg_scale.view_as(param))
        logger.debug("E8M0 scale conversion applied for %s", name)
        return True

    except Exception as exc:
        logger.warning("E8M0 scale conversion failed for %s: %s", name, exc)
        return False


@torch.no_grad()
def load_weights_kernel(model: Module, state_iter: Generator[tuple[str, torch.Tensor], None, None]) -> None:
    """Load vLLM kernel-format tensors using in-place copy_ updates.

    Handles the GB200 DeepGEMM E8M0 scale mismatch: the trainer broadcasts
    blockwise [N/128, K/128] float32 scales, while vLLM with E8M0 enabled
    stores weight_scale_inv in [N, K/512] TMA-aligned packed UE8M0 layout.
    When is_deep_gemm_e8m0_used() is True, scale tensors with a shape mismatch
    are automatically converted via requant_weight_ue8m0_inplace +
    transform_sf_into_required_layout.
    """
    params = dict(model.named_parameters())
    expert_source_indices = build_expert_map(model)

    # Detect E8M0 once (cached call, cheap).
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
        _use_e8m0 = is_deep_gemm_e8m0_used()
    except ImportError:
        _use_e8m0 = False

    # Track which weight params have been updated in this call so we can
    # safely call requant_weight_ue8m0_inplace on them (it must dequantise
    # the *new* FP8 values, not the stale ones from the initial model load).
    updated_weights: set[str] = set()

    loaded = 0
    skipped: list[str] = []
    shape_mismatches: list[str] = []

    for name, tensor in state_iter:
        if name not in params:
            skipped.append(name)
            continue

        param = params[name]
        if param.shape != tensor.shape:
            for module_name, source_indices in expert_source_indices.items():
                if not name.startswith(f"{module_name}."):
                    continue
                tensor = tensor[source_indices.to(tensor.device)]
                break

            if param.shape != tensor.shape:
                # Attempt E8M0 scale conversion before declaring a mismatch.
                if _use_e8m0 and _try_e8m0_scale_conversion(
                    name, tensor, param, params, updated_weights
                ):
                    loaded += 1
                    continue

                shape_mismatches.append(f"{name}: param={list(param.shape)} != received={list(tensor.shape)}")
                continue

        param.copy_(tensor)
        loaded += 1
        updated_weights.add(name)

    if shape_mismatches:
        raise ValueError(f"Kernel weight transfer had {len(shape_mismatches)} shape mismatches: {shape_mismatches}")
    if skipped:
        raise ValueError(f"Kernel weight transfer skipped {len(skipped)} weights not found in model: {skipped}")
    logger.debug(f"Kernel weight transfer copied {loaded} weights in-place")


@torch.no_grad()
def update_mla_absorbed_weights(model: Module) -> None:
    """Recompute MLA absorbed KV weights after in-place kv_b_proj updates."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights

    for name, module in model.named_modules():
        has_absorbed_weights = hasattr(module, "W_UV") or hasattr(module, "W_UK_T")
        if not has_absorbed_weights or not hasattr(module, "kv_b_proj"):
            continue

        if hasattr(module, "W_UV"):
            out_dtype = module.W_UV.dtype
        else:
            out_dtype = torch.bfloat16

        kv_b_proj_weight = get_and_maybe_dequant_weights(module.kv_b_proj, out_dtype=out_dtype).T
        kv_b_proj_weight = kv_b_proj_weight.view(
            module.kv_lora_rank,
            module.num_heads,
            module.qk_nope_head_dim + module.v_head_dim,
        )
        w_uk, w_uv = kv_b_proj_weight.split([module.qk_nope_head_dim, module.v_head_dim], dim=-1)

        if hasattr(module, "W_UV"):
            module.W_UV.copy_(w_uv.transpose(0, 1))
        if hasattr(module, "W_UK_T"):
            module.W_UK_T.copy_(w_uk.permute(1, 2, 0))

        logger.debug(f"Updated MLA absorbed weights for module {name}")
