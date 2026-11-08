import re
from typing import Callable, List

import torch
import torch.nn as nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.models.layers.lora import (
    LoRAGptOssGroupedExperts,
    LoRAGroupedExperts,
    LoRALinear,
    LoRAModule,
)
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.utils.logger import get_logger


class LoRAState:
    """Registry of the adapted modules, used for adapter state dicts and parameter resets."""

    def __init__(self):
        self._modules: list[tuple[str, LoRAModule]] = []
        self._adapter_state_dict_converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None

    def register_adapter_state_dict_converter(
        self, converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    ) -> None:
        """Register a converter applied to adapter state dicts (e.g. model.convert_adapter_to_hf)."""
        self._adapter_state_dict_converter = converter

    def register_module(self, prefix: str, module: LoRAModule) -> None:
        """Register an adapted module with its FQN prefix (e.g. "model.layers.0.self_attn.q_proj")."""
        self._modules.append((prefix, module))

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Adapter-only state dict, converted for HF compatibility when a converter is registered."""
        state_dict = {}
        for prefix, module in self._modules:
            for name, tensor in module.adapter_state_dict().items():
                state_dict[f"{prefix}.{name}"] = tensor

        if self._adapter_state_dict_converter is not None:
            state_dict = self._adapter_state_dict_converter(state_dict)
        return state_dict

    def reset_adapter_parameters(self) -> None:
        """Reset the adapter to fresh initialization across all registered modules."""
        for _, module in self._modules:
            module.reset_parameters()


_LORA_STATE: LoRAState | None = None


def get_lora_state() -> LoRAState:
    """Returns the LoRAState singleton. Initialized by ``apply_lora_to_model``."""
    if _LORA_STATE is None:
        raise RuntimeError("LoRAState not initialized. Apply LoRA to the model first (`apply_lora_to_model`).")
    return _LORA_STATE


def setup_lora_state() -> LoRAState:
    global _LORA_STATE
    _LORA_STATE = LoRAState()
    return _LORA_STATE


def strip_lora_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip LoRA from the state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        new_state_dict[key] = value
    return new_state_dict


def _has_regex_metacharacters(pattern: str) -> bool:
    """Check if a pattern contains regex metacharacters."""
    regex_metachars = {".", "*", "+", "?", "^", "$", "[", "]", "{", "}", "|", "(", ")", "\\"}
    return any(char in pattern for char in regex_metachars)


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if a name matches a pattern.

    For simple patterns (no regex metacharacters), checks if any component
    in the module path matches the pattern exactly. For regex patterns, uses
    re.search() to match anywhere in the name (mirroring PEFT behavior).

    This handles cases where Linear layers might be nested (e.g.,
    "model.layers.0.q_proj.linear") while still matching standard architectures
    where they're direct children (e.g., "model.layers.0.self_attn.q_proj").
    """
    if _has_regex_metacharacters(pattern):
        return re.search(pattern, name) is not None
    else:
        return pattern in name.split(".")


def _find_target_modules(model: nn.Module, target_patterns: List[str]) -> List[str]:
    """Find all module names that match any of the target patterns.

    Patterns can be simple module names (e.g., "q_proj") or regex patterns
    (e.g., r".*\\.q_proj$"). Simple names match any component in the module path.

    Supports both nn.Linear layers and GroupedExperts (MoE) modules.
    """
    target_modules = []

    for name, module in model.named_modules():
        # Check if module is Linear or a supported expert class
        if not isinstance(module, (nn.Linear, GroupedExperts)):
            continue

        for pattern in target_patterns:
            if _matches_pattern(name, pattern):
                target_modules.append(name)
                break

    return target_modules


def freeze_all_except_lora(model: nn.Module) -> None:
    """Freeze all parameters except the LoRA adapters."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """
    Apply LoRA to target modules in the model and freeze non-LoRA parameters.

    WARNING: This function modifies requires_grad on parameters. If using FSDP2,
    this MUST be called BEFORE setup_fsdp() to avoid dtensor/sharding issues.

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
    """
    logger = get_logger()
    from prime_rl.trainer.models import PreTrainedModelPrimeRL

    lora_state = setup_lora_state()
    if isinstance(model, PreTrainedModelPrimeRL):
        lora_state.register_adapter_state_dict_converter(type(model).convert_adapter_to_hf)
    uses_gpt_oss_moe_adapter = (
        isinstance(model, PreTrainedModelPrimeRL) and getattr(model.config, "model_type", None) == "gpt_oss"
    )

    from torch.distributed.fsdp import FSDPModule

    if any(isinstance(m, FSDPModule) for m in model.modules()):
        logger.error(
            "Model is already wrapped with FSDP! LoRA must be applied BEFORE FSDP setup to avoid dtensor issues."
        )
        raise RuntimeError("Cannot apply LoRA to FSDP-wrapped model. Apply LoRA before setup_fsdp().")

    logger.debug(f"Applying LoRA to {type(model).__name__} (target_modules={config.target_modules})")
    target_modules = _find_target_modules(model, config.target_modules)
    logger.debug(
        f"Found {len(target_modules)} target modules for LoRA: {target_modules[:10]} ... {target_modules[-10:]}"
    )

    if not target_modules:
        raise ValueError(f"No LoRA target modules found for patterns {config.target_modules}.")

    for module_name in target_modules:
        base_module = model.get_submodule(module_name)
        if isinstance(base_module, nn.Linear):
            wrapper = LoRALinear
        elif uses_gpt_oss_moe_adapter:
            wrapper = LoRAGptOssGroupedExperts
        else:
            wrapper = LoRAGroupedExperts
        lora_module = wrapper(base_module, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
        lora_state.register_module(module_name, lora_module)
        model.set_submodule(module_name, lora_module)

    freeze_all_except_lora(model)

    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA enabled: {adapter_params:,} adapter params on {len(target_modules)} modules ({total_params:,} total)"
    )


def has_lora_layers(model: nn.Module) -> bool:
    return any(isinstance(module, LoRAModule) for module in model.modules())


def save_lora_config(model: nn.Module, save_path, rank: int, alpha: float, dropout: float) -> None:
    """
    Save LoRA configuration as JSON for adapter portability.

    Args:
        model: Model with LoRA layers to introspect
        save_path: Path object or string pointing to directory where adapter_config.json will be saved
        rank: LoRA rank
        alpha: LoRA alpha scaling parameter
        dropout: LoRA dropout rate
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)

    target_modules = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, LoRAModule)}

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model.config._name_or_path,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": sorted(target_modules),
    }

    config_path = save_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
