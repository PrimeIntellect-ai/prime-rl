import hashlib
from typing import Iterable, Literal

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")

WeightTarget = Literal["model", "draft"]


def get_weight_model(model_runner, target: WeightTarget) -> Module:
    if target == "model":
        model = model_runner.get_model()
    elif target == "draft":
        model = model_runner.get_draft_model()
    else:
        raise ValueError(f"Unknown weight target: {target}")
    if model is None:
        raise ValueError(f"The inference worker has no {target} model")
    return model.runnable if hasattr(model, "runnable") else model


class WeightInspectionMixin:
    def get_weight_fingerprint(self, target: WeightTarget = "model") -> dict:
        """Exact per-worker tensor digests for validating distributed reloads."""
        model = get_weight_model(self.model_runner, target)
        tensors = dict(model.named_parameters())
        tensors.update(
            (f"@{module_name}.{name}", value)
            for module_name, module in model.named_modules()
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor) and value.is_cuda and ("weight" in name or "bias" in name)
        )
        return {
            name: {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "data_ptr": parameter.data_ptr(),
                "sha256": hashlib.sha256(
                    parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
                "nonzero": parameter.count_nonzero().item(),
            }
            for name, parameter in tensors.items()
        }


def load_policy_weights(model_runner, state_iter, vllm_config, target: WeightTarget = "model") -> None:
    """Load a policy broadcast, routing the optional speculator namespace to its model."""
    model = get_weight_model(model_runner, target)
    if target == "draft":
        target_storage = {
            parameter.untyped_storage().data_ptr() for parameter in get_weight_model(model_runner, "model").parameters()
        }
        shared_names = {
            name.removeprefix("model.")
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.untyped_storage().data_ptr() in target_storage
        }

        def owned_draft_weights():
            for name, tensor in state_iter:
                if name.removeprefix("model.") in shared_names:
                    logger.info("Preserving target-owned shared draft weight %s", name)
                    continue
                yield name, tensor

        load_weights_checkpoint_layerwise(
            model,
            owned_draft_weights(),
            vllm_config.speculative_config.draft_model_config,
            vllm_config,
            preserve_tensor_attributes=True,
        )
        return

    draft_weights = []

    def policy_weights():
        for name, tensor in state_iter:
            if name.startswith("speculator."):
                if vllm_config.speculative_config is None:
                    raise ValueError("Received speculator weights without speculative decoding enabled")
                draft_weights.append((name.removeprefix("speculator."), tensor))
            else:
                yield name, tensor

    load_weights_checkpoint_layerwise(model, policy_weights(), model_runner.model_config, vllm_config)
    if draft_weights:
        load_policy_weights(model_runner, draft_weights, vllm_config, target="draft")


@torch.no_grad()
def load_weights_checkpoint_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    vllm_config,
    *,
    preserve_tensor_attributes: bool = False,
) -> None:
    logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
    device = next(model.parameters()).device
    # Compiled draft forwards can retain tensor attributes such as fused KV projections.
    tensor_attributes = [
        (module, name, value)
        for module in model.modules()
        for name, value in vars(module).items()
        if preserve_tensor_attributes and isinstance(value, torch.Tensor)
    ]
    with torch.device(device), set_current_vllm_config(vllm_config):
        initialize_layerwise_reload(model)
        model.load_weights(state_iter)  # type: ignore
        finalize_layerwise_reload(model, model_config)
        if preserve_tensor_attributes:
            for module in model.modules():
                rebuild = getattr(module, "_build_fused_kv_buffers", None)
                if rebuild is not None:
                    rebuild()
        for module, name, original in tensor_attributes:
            updated = getattr(module, name)
            if updated is not original:
                if updated.shape != original.shape or updated.dtype != original.dtype:
                    raise ValueError(f"Weight reload changed the layout of {type(module).__name__}.{name}")
                original.copy_(updated)
                setattr(module, name, original)


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
