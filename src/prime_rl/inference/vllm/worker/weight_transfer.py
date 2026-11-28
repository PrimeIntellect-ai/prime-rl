import hashlib
from dataclasses import replace
from typing import Iterable, Iterator, Literal

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.utils import get_draft_load_config

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
    """Stream an HF-format checkpoint directory with vLLM's default loader, also for a server that started from dummy
    weights."""
    if load_config.load_format == "dummy":
        load_config = replace(load_config, load_format="auto")
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

    def get_weight_fingerprint(self, target: WeightTarget = "model", buffers: bool = False) -> dict:
        """Exact per-worker tensor digests for validating distributed reloads: parameters, and bare CUDA tensor
        attributes of modules named like weights or biases (fused or post-processed copies), as ``@module.name``.
        With ``buffers``, also every materialized buffer and bare tensor attribute, except views of the KV cache.

        Draft fingerprints skip target-owned shared weights, so they change only with the draft."""
        model = get_weight_model(self.model_runner, target)
        shared_storage = target_owned_storage(self.model_runner) if target == "draft" else set()
        target_modules = (
            {id(module) for module in get_weight_model(self.model_runner, "model").modules()}
            if target == "draft"
            else set()
        )
        tensors = {
            name: ("parameter", parameter)
            for name, parameter in model.named_parameters(remove_duplicate=not buffers)
            if parameter.untyped_storage().data_ptr() not in shared_storage
        }
        tensors.update(
            (f"@{module_name}.{name}", ("attribute", value))
            for module_name, module in model.named_modules(remove_duplicate=not buffers)
            if id(module) not in target_modules
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor)
            and (not value.is_meta if buffers else value.is_cuda and ("weight" in name or "bias" in name))
        )
        if buffers:
            tensors.update(
                (name, ("buffer", buffer))
                for name, buffer in model.named_buffers(remove_duplicate=False)
                if not buffer.is_meta and buffer.untyped_storage().data_ptr() not in shared_storage
            )
            kv_cache_storage = {
                (cache.device, cache.untyped_storage().data_ptr())
                for cache in getattr(self.model_runner, "kv_caches", [])
            }
            tensors = {
                name: (kind, tensor)
                for name, (kind, tensor) in tensors.items()
                if kind == "parameter" or (tensor.device, tensor.untyped_storage().data_ptr()) not in kv_cache_storage
            }
        entries = {}
        for name, (kind, tensor) in tensors.items():
            logical = tensor.detach()
            if buffers and name.endswith(".input_mix_weight_down_block_inject.weight"):
                parent = model.get_submodule(name.rsplit(".", 2)[0])
                # The Qwen HC loader only writes the two logical projections, leaving trailing GEMM padding.
                if padding := parent.pad_size:
                    logical = logical[:-padding]
            entries[name] = {
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "data_ptr": tensor.data_ptr(),
                "sha256": hashlib.sha256(
                    logical.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
                **(
                    {
                        "stride": list(tensor.stride()),
                        "storage_offset": tensor.storage_offset(),
                        "storage_ptr": tensor.untyped_storage().data_ptr() if tensor.numel() else None,
                        "hashed_shape": list(logical.shape),
                        "device": str(tensor.device),
                    }
                    if buffers
                    else {"nonzero": tensor.count_nonzero().item()}
                ),
            }
        return entries


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
        for module, name, original in tensor_attributes:
            updated = getattr(module, name)
            if updated is not original:
                if updated.shape != original.shape or updated.dtype != original.dtype:
                    raise ValueError(f"Weight reload changed the layout of {type(module).__name__}.{name}")
                original.copy_(updated)
                setattr(module, name, original)
