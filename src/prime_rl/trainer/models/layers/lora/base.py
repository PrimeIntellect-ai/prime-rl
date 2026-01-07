from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from prime_rl.trainer.runs import Runs

_LORA_PREFIX = "base_layer."

# Global state for multilora - initialized once in train.py, referenced by modules
LORA_NUM_TOKENS: torch.Tensor | None = None  # [n_adapters] token counts per adapter
SCALING_FACTORS: torch.Tensor | None = None  # [max_runs] scaling factors per run


def set_lora_num_tokens(num_tokens: torch.Tensor, reset_reference: bool = False) -> None:
    """Set the number of tokens per adapter.

    Args:
        num_tokens: Tensor of shape [n_adapters] with token counts per adapter.
        reset_reference: If True, replace the tensor reference. If False, copy values in-place.
    """
    global LORA_NUM_TOKENS
    if LORA_NUM_TOKENS is None or reset_reference:
        if SCALING_FACTORS is not None:
            assert num_tokens.shape == SCALING_FACTORS.shape, (
                f"lora_num_tokens shape {num_tokens.shape} != scaling_factors shape {SCALING_FACTORS.shape}"
            )
        LORA_NUM_TOKENS = num_tokens
    else:
        LORA_NUM_TOKENS.copy_(num_tokens)


def get_lora_num_tokens() -> torch.Tensor:
    """Get the current lora_num_tokens tensor.

    Raises:
        AssertionError: If called before set_lora_num_tokens() has been called.
    """
    assert LORA_NUM_TOKENS is not None, "LORA_NUM_TOKENS not initialized. Call set_lora_num_tokens() first."
    return LORA_NUM_TOKENS


def set_multilora_scaling(scaling_factors: torch.Tensor, reset_reference: bool = False) -> None:
    """Set per-run scaling factors (alpha / rank).

    Args:
        scaling_factors: Tensor of shape [max_runs] with scaling factors per run.
        reset_reference: If True, replace the tensor reference. If False, copy values in-place.
    """
    global SCALING_FACTORS
    if SCALING_FACTORS is None or reset_reference:
        if LORA_NUM_TOKENS is not None:
            assert scaling_factors.shape == LORA_NUM_TOKENS.shape, (
                f"scaling_factors shape {scaling_factors.shape} != lora_num_tokens shape {LORA_NUM_TOKENS.shape}"
            )
        SCALING_FACTORS = scaling_factors
    else:
        SCALING_FACTORS.copy_(scaling_factors)


def get_multilora_scaling() -> torch.Tensor:
    """Get the current scaling factors tensor.

    Raises:
        AssertionError: If called before set_multilora_scaling() has been called.
    """
    assert SCALING_FACTORS is not None, "SCALING_FACTORS not initialized. Call set_multilora_scaling() first."
    return SCALING_FACTORS


class MultiLoRAModule(nn.Module):
    """
    Base class for Multi run LoRA modules.
    """

    base_layer: nn.Module

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        self.base_layer = base_layer

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # state_dict post hook to remove prefix (to save base_layer parameters)
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to add back prefix (to load base_layer parameters)
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    @abstractmethod
    def reset_parameters(self, index: int | None = None) -> None:
        """Reset LoRA parameters.

        Args:
            index: If provided, reset only the parameters for that adapter index.
                   If None, reset all adapter parameters.
        """
        ...

    @abstractmethod
    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        """Get named parameters for a specific adapter index.

        Args:
            idx: The adapter index to get parameters for

        Returns:
            List of (name, parameter) tuples for the specified adapter
        """
        ...

    @abstractmethod
    def get_lora_param_counts(self) -> tuple[int, int]:
        """Get the number of LoRA adapter parameters and adapted base parameters.

        Returns:
            A tuple of (adapter_params, adapted_params) where:
            - adapter_params: Number of parameters in ONE LoRA adapter (lora_A + lora_B)
            - adapted_params: Number of base layer parameters being adapted by LoRA
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
