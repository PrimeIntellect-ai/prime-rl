from torch import Tensor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, require_flash_attn_kernels


class PreTrainedModelPrimeRL(PreTrainedModel):
    """
    Base class for all PrimeRL models that extends HuggingFace PreTrainedModel.

    Provides a unified interface for state dict conversion between different formats
    (e.g., HuggingFace format vs. training-optimized format) and buffer initialization
    after loading with meta device.
    """

    # Attention implementations this model's modeling code can dispatch. The default covers the
    # shared `layers.attn` classes; models with their own attention declare a different set or
    # override `validate_attn_impl` for conditional (e.g. hardware-dependent) constraints.
    supported_attn_impls: frozenset[str] = frozenset(ATTN_IMPL2CLASS)

    @classmethod
    def validate_attn_impl(cls, config: PretrainedConfig, attn_impl: str) -> None:
        """Validate the requested attention implementation at load time.

        Raises a ValueError naming the model and its supported implementations instead of
        failing with a KeyError mid-init or a kernel error at the first forward pass.
        """
        if attn_impl not in cls.supported_attn_impls:
            raise ValueError(
                f"{cls.__name__} does not support attn='{attn_impl}'. "
                f"Supported: {sorted(cls.supported_attn_impls)}. Set [trainer.model] attn accordingly."
            )
        require_flash_attn_kernels(attn_impl)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Public from_config that mirrors the Auto class API."""
        return cls._from_config(config, **kwargs)

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """PrimeRL models use custom MoE implementations and don't support dynamic experts implementation."""
        return False

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        """PrimeRL models always use eager experts implementation."""
        return "eager"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """
        Check if the state dict is in HuggingFace format.

        Args:
            state_dict: The state dict to check.

        Returns:
            True if the state dict is in HuggingFace format, False otherwise.
        """
        raise NotImplementedError(f"is_hf_state_dict is not implemented for {cls.__name__}")

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """
        Check if the state dict is in PrimeRL training format.

        Args:
            state_dict: The state dict to check.

        Returns:
            True if the state dict is in PrimeRL format, False otherwise.
        """
        raise NotImplementedError(f"is_prime_state_dict is not implemented for {cls.__name__}")

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Convert state dict from PrimeRL training format to HuggingFace format in-place.

        This is used when saving checkpoints or broadcasting weights to inference engines
        that expect HuggingFace-compatible format.

        Args:
            state_dict: The state dict to convert (modified in-place).
        """
        raise NotImplementedError(f"convert_to_hf is not implemented for {cls.__name__}")

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Convert state dict from HuggingFace format to PrimeRL training format in-place.

        This is used when loading pretrained HuggingFace models for training with
        PrimeRL-specific optimizations.

        Args:
            state_dict: The state dict to convert (modified in-place).
        """
        raise NotImplementedError(f"convert_to_prime is not implemented for {cls.__name__}")

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """
        Convert a single layer's state dict from PrimeRL format to HuggingFace format in-place.

        This is used for layer-by-layer conversion during NCCL broadcast to reduce memory usage.

        Args:
            state_dict: The state dict containing the layer to convert (modified in-place).
            layer_idx: The index of the layer to convert.
        """
        raise NotImplementedError(f"convert_layer_to_hf is not implemented for {cls.__name__}")

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """
        Convert a single layer's state dict from HuggingFace format to PrimeRL format in-place.

        This is used for layer-by-layer conversion during loading.

        Args:
            state_dict: The state dict containing the layer to convert (modified in-place).
            layer_idx: The index of the layer to convert.
        """
        raise NotImplementedError(f"convert_layer_to_prime is not implemented for {cls.__name__}")

    @classmethod
    def convert_adapter_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Convert a LoRA adapter state dict from PrimeRL training format to HuggingFace format.

        Unlike convert_to_hf, this operates on a partial state dict containing only LoRA
        adapter parameters (e.g. `model.layers.N.<submodule>.<proj>.lora_A.weight`). Models
        whose HF naming differs from PrimeRL naming at the submodule level (e.g. NemotronH's
        unified `mixer` attribute) should override this to perform the rename.

        Implementations may mutate state_dict in-place or return a new dict; callers must
        use the returned value. Default implementation is a no-op.
        """
        return state_dict

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls,
        state_dict: dict[str, Tensor],
        layer_idx: int,
        quantize_fp8: bool = False,
    ) -> dict[str, Tensor]:
        """
        Convert a single layer's state dict from PrimeRL format to vLLM kernel format.

        Args:
            state_dict: Layer weights in PrimeRL format.
            layer_idx: Layer index to convert.
            quantize_fp8: Whether to emit FP8 (e4m3) kernel weights with per-block scales.
        """
        raise NotImplementedError(f"convert_layer_to_vllm_kernel is not implemented for {cls.__name__}")

    def init_buffers_post_meta(self) -> None:
        """
        Initialize buffers that are not in the state dict after loading with meta device.

        Some models have buffers (non-trainable tensors) that are not saved in the state dict
        but need to be properly initialized after loading the model on meta device and then
        moving to the actual device. This method should initialize such buffers.

        This is called after loading the model from a checkpoint with meta device.
        """
        raise NotImplementedError(f"init_buffers_post_meta is not implemented for {self.__class__.__name__}")


__all__ = ["PreTrainedModelPrimeRL"]
