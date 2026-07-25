import re
from typing import Callable, List

import torch
import torch.nn as nn

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.models.layers.lora import (
    MultiLoRALinear,
    MultiLoRAModule,
    get_lora_num_tokens,
    get_multilora_scaling,
    set_lora_num_tokens,
    set_multilora_scaling,
)
from prime_rl.trainer.models.layers.lora.multi_moe import (
    MultiLoRAGptOssGroupedExperts,
    MultiLoRAGroupedExperts,
    MultiLoRAReLU2GroupedExperts,
)
from prime_rl.trainer.models.layers.moe import GptOssGroupedExperts, GroupedExperts
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class LoRAState:
    """Module registry + shared tensors for the single LoRA adapter.

    Owns the canonical ``lora_num_tokens`` / ``scaling_factors`` tensors the
    LoRA layers read (the layers capture references at construction, so this
    must exist before ``apply_lora_to_model`` builds them) and the registry of
    adapted modules used for optimizer setup, adapter state dicts, and
    parameter resets."""

    def __init__(self, config: LoRAConfig, device: torch.device):
        set_lora_num_tokens(None, reset_reference=True)
        set_multilora_scaling(None, reset_reference=True)
        set_lora_num_tokens(torch.zeros(1, dtype=torch.int32, device=device), reset_reference=True)
        set_multilora_scaling(
            torch.full((1,), config.alpha / config.rank, dtype=torch.bfloat16, device=device),
            reset_reference=True,
        )
        self.lora_num_tokens = get_lora_num_tokens()
        self.scaling_factors = get_multilora_scaling()
        self._modules: list[tuple[str, MultiLoRAModule]] = []
        self._adapter_state_dict_converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None

    def register_adapter_state_dict_converter(
        self, converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    ) -> None:
        """Register a converter applied to adapter state dicts (e.g. model.convert_adapter_to_hf)."""
        self._adapter_state_dict_converter = converter

    def register_module(self, prefix: str, module: MultiLoRAModule) -> None:
        """Register an adapted module with its FQN prefix (e.g. "model.layers.0.self_attn.q_proj")."""
        self._modules.append((prefix, module))

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Adapter-only state dict, converted for HF compatibility when a converter is registered."""
        state_dict = {}
        for prefix, module in self._modules:
            # MoE modules expose a custom state_dict_for_adapter returning the
            # vLLM-compatible per-expert format
            if hasattr(module, "state_dict_for_adapter"):
                for name, tensor in module.state_dict_for_adapter(0).items():
                    state_dict[f"{prefix}.{name}"] = tensor.detach()
            else:
                for name, param in module.named_parameters_for_adapter(0):
                    state_dict[f"{prefix}.{name}.weight"] = param.detach()

        if self._adapter_state_dict_converter is not None:
            state_dict = self._adapter_state_dict_converter(state_dict)
        return state_dict

    def reset_adapter_parameters(self) -> None:
        """Reset the adapter to fresh initialization across all registered modules."""
        for _, module in self._modules:
            module.reset_parameters(0)


_LORA_STATE: LoRAState | None = None


def get_lora_state() -> LoRAState:
    """Returns the LoRAState singleton. Initialized by ``apply_lora_to_model``."""
    if _LORA_STATE is None:
        raise RuntimeError("LoRAState not initialized. Apply LoRA to the model first (`apply_lora_to_model`).")
    return _LORA_STATE


def setup_lora_state(config: LoRAConfig, device: torch.device) -> LoRAState:
    global _LORA_STATE
    _LORA_STATE = LoRAState(config, device)
    return _LORA_STATE


def strip_lora_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip LoRA from the state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        new_state_dict[key] = value
    return new_state_dict


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

    Supports both nn.Linear layers and GroupedExperts (MoE) modules.
    """
    target_modules = []

    for name, module in model.named_modules():
        # Check if module is Linear or one of the supported expert classes
        if not isinstance(module, (nn.Linear, GroupedExperts, GptOssGroupedExperts)):
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
    from prime_rl.trainer.models import PreTrainedModelPrimeRL

    lora_state = setup_lora_state(config, torch.device("cuda", get_world().local_rank))
    if isinstance(model, PreTrainedModelPrimeRL):
        lora_state.register_adapter_state_dict_converter(type(model).convert_adapter_to_hf)

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
        base_module = _get_module_by_name(model, module_name)

        # Handle Linear layers
        if isinstance(base_module, nn.Linear):
            lora_module = MultiLoRALinear(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=1,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        # Handle GroupedExperts (MoE)
        elif isinstance(base_module, GroupedExperts):
            lora_cls = (
                MultiLoRAReLU2GroupedExperts if base_module.input_weight_names == ("w1",) else MultiLoRAGroupedExperts
            )
            lora_module = lora_cls(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=1,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        # Handle GptOssGroupedExperts (gpt-oss fused gate_up + biases)
        elif isinstance(base_module, GptOssGroupedExperts):
            lora_module = MultiLoRAGptOssGroupedExperts(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=1,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        else:
            logger.warning(
                f"Module {module_name} is type {type(base_module).__name__}, "
                f"expected nn.Linear, GroupedExperts, or GptOssGroupedExperts. Skipping."
            )
            continue

        lora_state.register_module(module_name, lora_module)
        _set_module_by_name(model, module_name, lora_module)

    freeze_all_except_lora_and_specified(model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_adapter_params = 0
    lora_adapted_params = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            adapter_params, adapted_params = module.get_lora_param_counts()
            lora_adapter_params += adapter_params
            lora_adapted_params += adapted_params

    fully_trainable = trainable_params - lora_adapter_params
    adapted_or_trainable = lora_adapted_params + fully_trainable

    logger.info(f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params")
    logger.info(f"LoRA: {fully_trainable:,} fully trainable parameters")
    logger.info(f"LoRA: {adapted_or_trainable:,} adapted or fully trainable out of {total_params:,} parameters")


def has_lora_layers(model: nn.Module) -> bool:
    """Check if model has LoRA layers."""
    for module in model.modules():
        if isinstance(module, MultiLoRAModule):
            return True
    return False


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

    # Extract actual target modules from the model
    target_modules = set()
    modules_to_save = set()

    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
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
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": sorted(list(target_modules)),
        "modules_to_save": sorted(list(modules_to_save)) if modules_to_save else None,
    }

    config_path = save_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
