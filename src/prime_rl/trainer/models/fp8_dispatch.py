from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise, quantize_to_fp8_channelwise

QuantSchemeName = Literal["fp8_blockwise", "fp8_channelwise"]


class QuantScheme(Protocol):
    scale_name_suffix: str
    scale_dtype: torch.dtype
    modules_to_not_convert: list[str]

    def quantize(self, weight: Tensor) -> tuple[Tensor, Tensor]: ...


class FP8BlockwiseQuantScheme:
    scale_name_suffix = "weight_scale_inv"
    scale_dtype = torch.float32

    def __init__(self, block_size: int = 128, modules_to_not_convert: list[str] | None = None):
        self.block_size = block_size
        self.modules_to_not_convert = modules_to_not_convert or []

    def quantize(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        return quantize_to_fp8_blockwise(weight, self.block_size)


class FP8ChannelwiseQuantScheme:
    scale_name_suffix = "weight_scale"
    scale_dtype = torch.float32

    def __init__(self, modules_to_not_convert: list[str] | None = None):
        self.modules_to_not_convert = modules_to_not_convert or []

    def quantize(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        return quantize_to_fp8_channelwise(weight)


def detect_quant_scheme(target_model: str) -> tuple[QuantSchemeName, QuantScheme] | None:
    """Parse quantization_config from a HF model; return (name, scheme) or None."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    qconfig = getattr(config, "quantization_config", None)
    if qconfig is None:
        return None

    qconfig = qconfig if isinstance(qconfig, dict) else qconfig.to_dict()
    quant_method = qconfig.get("quant_method", "")

    if quant_method == "fp8":
        block_size = qconfig.get("weight_block_size")
        skip = qconfig.get("modules_to_not_convert") or qconfig.get("ignored_layers") or []
        if block_size:
            return "fp8_blockwise", FP8BlockwiseQuantScheme(block_size[0], skip)
        return "fp8_channelwise", FP8ChannelwiseQuantScheme(skip)

    if quant_method == "compressed-tensors":
        ignore = qconfig.get("ignore") or []
        for group in qconfig.get("config_groups", {}).values():
            weights = group.get("weights", {})
            strategy = weights.get("strategy", "tensor")
            block_structure = weights.get("block_structure")
            if strategy == "block" or block_structure:
                bs = block_structure[0] if block_structure else 128
                return "fp8_blockwise", FP8BlockwiseQuantScheme(bs, ignore)
            return "fp8_channelwise", FP8ChannelwiseQuantScheme(ignore)

    return None


def get_quant_scheme(name: QuantSchemeName) -> QuantScheme:
    """Translate a scheme name string to a QuantScheme instance."""
    if name == "fp8_blockwise":
        return FP8BlockwiseQuantScheme()
    if name == "fp8_channelwise":
        return FP8ChannelwiseQuantScheme()
    raise ValueError(f"Unknown quant scheme: {name}")


def _expand_fused_moe_experts(layer_state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Expand fused 3D MoE expert tensors to per-expert 2D (HF checkpoint format).

    Transformers 5.x stores experts as fused 3D:
      experts.gate_up_proj: (num_experts, 2*intermediate, hidden)
      experts.down_proj:    (num_experts, hidden, intermediate)

    HF checkpoints and VLLM load_weights expect per-expert 2D:
      experts.{j}.gate_proj.weight: (intermediate, hidden)
      experts.{j}.up_proj.weight:   (intermediate, hidden)
      experts.{j}.down_proj.weight: (hidden, intermediate)
    """
    expanded: dict[str, Tensor] = {}
    for name, tensor in layer_state_dict.items():
        if tensor.ndim == 3 and ".mlp.experts." in name:
            base = name.rsplit(".", 1)[0]
            suffix = name.rsplit(".", 1)[1]

            if suffix == "gate_up_proj":
                gates, ups = tensor.chunk(2, dim=1)
                for j, (gate, up) in enumerate(zip(gates, ups)):
                    expanded[f"{base}.{j}.gate_proj.weight"] = gate
                    expanded[f"{base}.{j}.up_proj.weight"] = up
            elif suffix == "down_proj":
                for j, down in enumerate(tensor):
                    expanded[f"{base}.{j}.down_proj.weight"] = down
            else:
                expanded[name] = tensor
        else:
            expanded[name] = tensor
    return expanded


def quantize_layer_for_scheme(layer_state_dict: dict[str, Tensor], scheme: QuantScheme) -> dict[str, Tensor]:
    """Quantize 2D tensors in a layer state dict according to the scheme."""
    layer_state_dict = _expand_fused_moe_experts(layer_state_dict)
    quantized: dict[str, Tensor] = {}
    for name, tensor in layer_state_dict.items():
        should_skip = any(skip in name for skip in scheme.modules_to_not_convert)
        if tensor.ndim == 2 and not should_skip:
            fp8_weight, scale = scheme.quantize(tensor)
            quantized[name] = fp8_weight
            quantized[name.removesuffix(".weight") + f".{scheme.scale_name_suffix}"] = scale.to(scheme.scale_dtype)
        else:
            quantized[name] = tensor
    return quantized
