import re
from typing import Dict, List

import torch
import torch.nn as nn

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.models.layers.lora import LoRALinear
from prime_rl.utils.logger import get_logger


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Get a module by its fully qualified name."""
    parts = module_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by its fully qualified name."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


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
    """
    target_modules = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        for pattern in target_patterns:
            if _matches_pattern(name, pattern):
                target_modules.append(name)
                break

    return target_modules


def _should_keep_trainable(param_name: str, modules_to_save_patterns: List[str]) -> bool:
    """Check if a parameter should remain fully trainable.

    Checks both the full parameter name and the parent module name against patterns.
    For example, for param "model.embed_tokens.weight", it checks both:
    - "model.embed_tokens.weight" (full parameter name)
    - "model.embed_tokens" (module name)

    Patterns can be simple module names (e.g., "embed_tokens") or regex patterns.
    """
    for pattern in modules_to_save_patterns:
        if _matches_pattern(param_name, pattern):
            return True

    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
    for pattern in modules_to_save_patterns:
        if _matches_pattern(module_name, pattern):
            return True

    return False


def freeze_all_except_lora_and_specified(model: nn.Module, config: LoRAConfig) -> None:
    """
    Freeze all parameters except LoRA adapters and specified trainable modules.

    Args:
        model: The model to freeze parameters in
        config: LoRA configuration with modules_to_save patterns
    """
    for name, param in model.named_parameters():
        if any(lora_param in name for lora_param in ["lora_A", "lora_B"]):
            param.requires_grad = True
        elif _should_keep_trainable(name, config.modules_to_save):
            param.requires_grad = True
        else:
            param.requires_grad = False


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

    from torch.distributed.fsdp import FSDPModule

    if any(isinstance(m, FSDPModule) for m in model.modules()):
        logger.error(
            "Model is already wrapped with FSDP! LoRA must be applied BEFORE FSDP setup to avoid dtensor issues."
        )
        raise RuntimeError("Cannot apply LoRA to FSDP-wrapped model. Apply LoRA before setup_fsdp().")

    target_modules = _find_target_modules(model, config.target_modules)

    if not target_modules:
        logger.warning("No target modules found for LoRA. Check your target_modules patterns.")
        return

    for module_name in target_modules:
        base_module = _get_module_by_name(model, module_name)

        if not isinstance(base_module, nn.Linear):
            logger.warning(f"Module {module_name} is not nn.Linear, skipping")
            continue

        lora_module = LoRALinear(
            base_linear=base_module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )

        _set_module_by_name(model, module_name, lora_module)

    freeze_all_except_lora_and_specified(model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_adapter_params = 0
    lora_adapted_params = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_adapter_params += module.lora_A.numel() + module.lora_B.numel()
            lora_adapted_params += module.base_linear.weight.numel()

    fully_trainable = trainable_params - lora_adapter_params
    adapted_or_trainable = lora_adapted_params + fully_trainable

    logger.info(f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params")
    logger.info(f"LoRA: {fully_trainable:,} fully trainable parameters")
    logger.info(f"LoRA: {adapted_or_trainable:,} adapted or fully trainable out of {total_params:,} parameters")


def has_lora_layers(model: nn.Module) -> bool:
    """Check if model has LoRA layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return True
    return False


def clean_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove LoRA parameters and fix LoRA base layer key names for HF compatibility."""
    clean_state_dict = {}

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue

        if ".base_linear." in key:
            new_key = key.replace(".base_linear.", ".")
            clean_state_dict[new_key] = value
        else:
            clean_state_dict[key] = value

    return clean_state_dict


def save_lora_config(config: LoRAConfig, model: nn.Module, save_path) -> None:
    """
    Save LoRA configuration as JSON for adapter portability.

    Args:
        config: LoRA configuration to save
        model: Model with LoRA layers to introspect
        save_path: Path object or string pointing to directory where adapter_config.json will be saved
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)

    # Extract actual target modules from the model
    target_modules = set()
    modules_to_save = set()

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module_suffix = name.split(".")[-1]
            target_modules.add(module_suffix)

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name:
            module_name = name.rsplit(".", 1)[0].split(".")[-1]
            modules_to_save.add(module_name)

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model.config._name_or_path,
        "r": config.rank,
        "lora_alpha": config.alpha,
        "lora_dropout": config.dropout,
        "bias": "none",
        "target_modules": sorted(list(target_modules)),
        "modules_to_save": sorted(list(modules_to_save)) if modules_to_save else None,
    }

    config_path = save_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
