import math
from abc import abstractmethod
from typing import Any

import torch
from torch import nn

_LORA_PREFIX = "base_layer."


def lora_parameter(*shape: int, like: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, device=like.device, dtype=like.dtype))


class LoRAModule(nn.Module):
    """Base class for LoRA-wrapped modules.

    Subclasses register their ``*lora_A`` / ``*lora_B`` parameters directly on the module and call
    ``reset_parameters()`` at the end of ``__init__``.
    """

    base_layer: nn.Module

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for param in self.base_layer.parameters():
            param.requires_grad = False

        # state_dict post hook to remove prefix (to save base_layer parameters)
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to add back prefix (to load base_layer parameters)
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    def reset_parameters(self) -> None:
        """Kaiming uniform for A, zeros for B."""
        for name, param in self.named_parameters(recurse=False):
            if name.endswith("lora_A"):
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            else:
                nn.init.zeros_(param)

    @abstractmethod
    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Adapter tensors in the vLLM/PEFT layout, keyed relative to this module."""
        ...

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_layer, name)

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this LoRA module is executed.
        For LoRA modules, it will strip the LoRA module prefix,
        so that this module can be loaded into non-LoRA modules.
        It would still be able to be loaded into LoRA modules as this class
        adds the prefix back before loading the state_dict.
        """
        old_prefix = f"{prefix}{_LORA_PREFIX}"
        new_prefix = prefix
        for key in list(state_dict.keys()):
            if not key.startswith(old_prefix):
                continue
            new_key = new_prefix + key[len(old_prefix) :]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_load_state_dict_hook`` is called before ``self._load_from_state_dict()`` is called.
        For LoRA modules, it will add back the module prefix so that non-LoRA modules
        can be loaded into LoRA modules properly.
        """
        old_prefix = prefix
        new_prefix = f"{prefix}{_LORA_PREFIX}"
        for key in list(state_dict.keys()):
            if not key.startswith(old_prefix) or "lora_A" in key or "lora_B" in key:
                continue
            new_key = new_prefix + key[len(old_prefix) :]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
