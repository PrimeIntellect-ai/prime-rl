import hashlib
from typing import TYPE_CHECKING, Iterable, Iterator, Literal

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.model_loader.utils import get_draft_load_config

from prime_rl.inference.vllm.eplb import (
    current_expert_placement,
    install_trainer_placement,
    remap_expert_weight_loader,
)

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbState

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


def checkpoint_weights(model: Module, path: str, load_config) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream an HF-format checkpoint directory with vLLM's default loader."""
    model_loader = get_model_loader(load_config)
    assert isinstance(model_loader, DefaultModelLoader)
    source = DefaultModelLoader.Source(
        path,
        revision=None,
        prefix="",
        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    return model_loader._get_weights_iterator(source)


def target_owned_storage(model_runner) -> set[int]:
    """Storage of target parameters; the draft aliases some of them (e.g. embed_tokens, lm_head)."""
    return {
        parameter.untyped_storage().data_ptr() for parameter in get_weight_model(model_runner, "model").parameters()
    }


class WeightRPCMixin:
    """Weight RPCs shared by every weight-transport worker extension."""

    def update_draft_weights(self, weight_dir: str) -> None:
        """Load a draft checkpoint from disk, independent of the policy weight transport. The checkpoint is read
        in full first: an unreadable file then fails before the reload takes the served draft apart."""
        draft = get_weight_model(self.model_runner, "draft")
        weights = list(checkpoint_weights(draft, weight_dir, get_draft_load_config(self.vllm_config)))
        load_draft_weights(self.model_runner, weights, self.vllm_config)

    def get_weight_fingerprint(self, target: WeightTarget = "model") -> dict:
        """Exact per-worker tensor digests for validating distributed reloads.

        Draft fingerprints skip target-owned shared weights, so they change only with the draft."""
        model = get_weight_model(self.model_runner, target)
        shared_storage = target_owned_storage(self.model_runner) if target == "draft" else set()
        target_modules = (
            {id(module) for module in get_weight_model(self.model_runner, "model").modules()}
            if target == "draft"
            else set()
        )
        tensors = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.untyped_storage().data_ptr() not in shared_storage
        }
        tensors.update(
            (f"@{module_name}.{name}", value)
            for module_name, module in model.named_modules()
            if id(module) not in target_modules
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


def load_draft_weights(model_runner, state_iter, vllm_config) -> None:
    """Load draft-owned weights in place, leaving target-owned shared weights to the policy."""
    model = get_weight_model(model_runner, "draft")
    shared_storage = target_owned_storage(model_runner)
    shared_names = {
        name.removeprefix("model.")
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.untyped_storage().data_ptr() in shared_storage
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


@torch.no_grad()
def load_weights_checkpoint_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    vllm_config,
    eplb_state: "EplbState | None" = None,
    weight_dir: str | None = None,
    *,
    preserve_tensor_attributes: bool = False,
) -> None:
    logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
    device = next(model.parameters()).device
    place_experts_from_trainer(eplb_state, model_config, weight_dir)
    # Compiled draft forwards can retain tensor attributes such as fused KV projections.
    tensor_attributes = [
        (module, name, value)
        for module in model.modules()
        for name, value in vars(module).items()
        if preserve_tensor_attributes and isinstance(value, torch.Tensor)
    ]
    weight_loaders = []
    moe_kernels = []
    with torch.device(device), set_current_vllm_config(vllm_config):
        initialize_layerwise_reload(model)
        if eplb_state is not None:
            _, layers = current_expert_placement(eplb_state, model_config)
            for experts, destinations in layers:
                moe_kernels += stash_moe_kernels([experts])
                kernel_parameters, _ = get_layerwise_info(experts).kernel_tensors
                for name, param in experts.named_parameters(recurse=False):
                    # vLLM reuses these metadata parameters on subsequent reloads.
                    weight_loaders.append((param, param.weight_loader))
                    param.weight_loader = remap_expert_weight_loader(param.weight_loader, destinations)
                    # A layer can finish before the checkpoint stream reaches its
                    # remaining nonlocal experts. Keep those calls in the completed
                    # layerwise loader after vLLM restores the kernel parameters.
                    kernel_param = kernel_parameters[name]
                    weight_loaders.append((kernel_param, kernel_param.weight_loader))
                    kernel_param.weight_loader = param.weight_loader
        try:
            model.load_weights(state_iter)  # type: ignore
            finalize_layerwise_reload(model, model_config)
        finally:
            # Restored even when the update fails, so a retry does not remap already-physical expert ids
            for param, loader in weight_loaders:
                param.weight_loader = loader
            restore_moe_kernels(moe_kernels)
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


def stash_moe_kernels(experts_modules) -> list:
    """The kernels EPLB registered for these routed experts: a reload's quantization rebuilds them against temporary
    scales, and EPLB's expert moves must keep targeting the ones that reference the live storage."""
    return [
        (experts.quant_method, experts.quant_method.moe_quant_config, experts.quant_method.moe_kernel)
        for experts in experts_modules
    ]


def restore_moe_kernels(moe_kernels: list) -> None:
    for quant_method, quant_config, kernel in moe_kernels:
        quant_method.moe_quant_config = quant_config
        quant_method.moe_kernel = kernel


def place_experts_from_trainer(eplb_state: "EplbState | None", model_config, weight_dir: str | None) -> None:
    """Place the experts by the counts the trainer published with this update, before its weights are written (see
    ``install_trainer_placement``)."""
    if eplb_state is None:
        return
    moved = install_trainer_placement(eplb_state, model_config, weight_dir)
    if moved is not None:
        logger.info("Placed EPLB experts from the trainer's expert counts: %d physical slots changed", moved)


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
