import hashlib
from typing import Iterable, Iterator, Literal

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")

WeightTarget = Literal["model", "draft"]

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def check_fp8_source(name: str, param_dtype: torch.dtype, source_dtype: torch.dtype) -> None:
    """A pre-quantized (fp8) param only takes fp8 checkpoint tensors. Loading a bf16/fp32 tensor would cast
    it into the fp8 param without its scale, which silently corrupts the layer."""
    if param_dtype in FP8_DTYPES and source_dtype not in FP8_DTYPES:
        raise ValueError(
            f"Cannot load a {source_dtype} tensor into the fp8 param {name} of a pre-quantized checkpoint. "
            'The trainer sends bf16 weights: serve the BF16 checkpoint with `quantization = "fp8_per_block"`.'
        )


def _guard_fp8_params(model: Module) -> None:
    """Wrap the loaders of the fp8 params of pre-quantized layers with ``check_fp8_source``. The wrapper keeps the
    layerwise loader's name and ``__wrapped__`` so vLLM's reload unwraps it like its own."""
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.dtype not in FP8_DTYPES:
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            name = f"{module_name}.{param_name}"

            def guarded(*args, loader=loader, name=name, param_dtype=param.dtype, **kwargs):
                source = kwargs["loaded_weight"] if "loaded_weight" in kwargs else args[1]
                check_fp8_source(name, param_dtype, source.dtype)
                return loader(*args, **kwargs)

            guarded.__name__ = loader.__name__
            guarded.__wrapped__ = loader
            param.weight_loader = guarded


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
        _guard_fp8_params(model)
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
