import math
import re
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from prime_rl.trainer.config import LoRAConfig
from prime_rl.utils.logger import get_logger


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer.

    Implements the low-rank decomposition: ΔW = B @ A
    where A ∈ R^(rank x in_features), B ∈ R^(out_features x rank)

    Forward pass: y = x @ (W + ΔW).T = x @ W.T + x @ A.T @ B.T * (alpha / rank)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = Parameter(torch.empty(base_layer.out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def _init_parameters(self):
        """Initialize LoRA parameters following standard LoRA initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + lora_output"""
        base_output = self.base_layer(x)
        lora_x = self.lora_dropout(x)
        lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base layer and return a new linear layer."""
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        merged_layer = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )

        merged_layer.weight.data = self.base_layer.weight.data + delta_weight
        if self.base_layer.bias is not None:
            merged_layer.bias.data = self.base_layer.bias.data.clone()

        return merged_layer


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


def _find_target_modules(model: nn.Module, target_patterns: List[str]) -> List[str]:
    """Find all module names that match any of the target regex patterns."""
    target_modules = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        for pattern in target_patterns:
            if re.match(pattern, name):
                target_modules.append(name)
                break

    return target_modules


def _should_keep_trainable(param_name: str, trainable_patterns: List[str]) -> bool:
    """Check if a parameter should remain fully trainable.

    Checks both the full parameter name and the parent module name against patterns.
    For example, for param "model.embed_tokens.weight", it checks both:
    - "model.embed_tokens.weight" (full parameter name)
    - "model.embed_tokens" (module name)
    """
    for pattern in trainable_patterns:
        if re.match(pattern, param_name):
            return True

    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
    for pattern in trainable_patterns:
        if re.match(pattern, module_name):
            return True

    return False


def freeze_all_except_lora_and_specified(model: nn.Module, config: LoRAConfig) -> None:
    """
    Freeze all parameters except LoRA adapters and specified trainable modules.

    Args:
        model: The model to freeze parameters in
        config: LoRA configuration with trainable_modules patterns
    """
    frozen_params = 0
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1

        if any(lora_param in name for lora_param in ["lora_A", "lora_B"]):
            param.requires_grad = True
            trainable_params += 1
        elif _should_keep_trainable(name, config.trainable_modules):
            param.requires_grad = True
            trainable_params += 1
        else:
            param.requires_grad = False
            frozen_params += 1


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

    if not config.enabled:
        return

    target_modules = _find_target_modules(model, config.target_modules)

    if not target_modules:
        logger.warning("No target modules found for LoRA. Check your target_modules regex patterns.")
        return

    for module_name in target_modules:
        base_module = _get_module_by_name(model, module_name)

        if not isinstance(base_module, nn.Linear):
            logger.warning(f"Module {module_name} is not nn.Linear, skipping")
            continue

        lora_module = LoRALinear(
            base_layer=base_module,
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
            lora_adapted_params += module.base_layer.weight.numel()

    fully_trainable = trainable_params - lora_adapter_params

    logger.info(f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params")
    logger.info(f"LoRA: {fully_trainable:,} fully trainable parameters")
    logger.info(f"LoRA: {trainable_params:,} trainable out of {total_params:,} parameters")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights in the model back into base layers.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with LoRA weights merged into base layers
    """
    logger = get_logger()
    merged_count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            merged_layer = module.merge_weights()

            _set_module_by_name(model, name, merged_layer)
            merged_count += 1

    if merged_count > 0:
        logger.info(f"Merged {merged_count} LoRA modules back into base model")

    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model state dict.

    Returns:
        Dictionary containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.clone()

    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA parameters into model.

    Args:
        model: Model with LoRA modules
        lora_state_dict: Dictionary containing LoRA parameters
    """
    logger = get_logger()
    loaded_params = 0

    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
            loaded_params += 1
        else:
            logger.warning(f"LoRA parameter {name} not found in model")
