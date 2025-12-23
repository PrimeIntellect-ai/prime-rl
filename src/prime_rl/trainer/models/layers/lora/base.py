from abc import abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from prime_rl.trainer.runs import Runs

_LORA_PREFIX = "base_layer."


def set_offsets(offsets: torch.Tensor, reset_reference: bool = False) -> None:
    """Set offsets for all LoRA modules."""
    from prime_rl.trainer.models.layers.lora.multi_linear import set_offsets as set_offsets_multi_linear

    set_offsets_multi_linear(offsets, reset_reference)


class MultiLoRAModule(nn.Module):
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

    @abstractmethod
    def reset_parameters(self, index: int | None = None) -> None:
        """Reset LoRA parameters.

        Args:
            index: If provided, reset only the parameters for that adapter index.
                   If None, reset all adapter parameters.
        """
        ...

    def register_with_runs(self, runs: "Runs", prefix: str) -> None:
        """Register this module with the Runs system.

        This method should be called after FSDP/compile/AC setup as these
        transformations may change the underlying parameters while preserving
        the module.

        The Runs class will use this registration to:
        - Get named parameters for specific run indices (for optimizer setup)
        - Reset parameters when new runs are created
        - Construct sliced state dicts for weight broadcast

        Args:
            runs: The Runs instance to register with
            prefix: The module's name/prefix in the model (e.g., "model.layers.0.self_attn.q_proj")
        """
        runs.register_module(prefix, self)

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

    def state_dict(
        self,
        *args,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
        index: int | None = None,
    ):
        """If index is not provided, do the standard state_dict.
        If index is provided, save only the parameters for the given index.
        This means skipping the base_layer and slicing into nn.ParameterList.
        """
        if index is None:
            return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if name == "base_layer":
                continue
            if isinstance(module, nn.ParameterList):
                destination[prefix + name] = module[index] if keep_vars else module[index].detach()
            elif module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if not getattr(hook, "_from_public_api", False):
                if hook_result is not None:
                    destination = hook_result
            else:
                if hook_result is not None:
                    raise RuntimeError("state_dict post-hook must return None")
        return destination
