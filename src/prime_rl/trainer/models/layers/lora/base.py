from typing import Any

from torch import nn

_LORA_PREFIX = "base_layer."


class LoRAModule(nn.Module):
    """
    Base class for LoRA modules with shared functionality for state dict hooks
    and attribute forwarding to the base layer.
    """

    base_layer: nn.Module

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        self.base_layer = base_layer

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Register state dict hooks for LoRA compatibility
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_layer, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.base_layer.__getitem__(key)  # type: ignore[operator]

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
