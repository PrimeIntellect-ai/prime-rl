import torch
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.slots import Slot, build_slots_for_conversion_spec
from prime_rl.trainer.parallel_dims import ParallelDims


class PreTrainedModelPrimeRL(PreTrainedModel):
    """
    Base class for all PrimeRL models that extends HuggingFace PreTrainedModel.

    Provides a unified interface for state dict conversion between different formats
    (e.g., HuggingFace format vs. training-optimized format) and buffer initialization
    after loading with meta device.
    """

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

    def get_conversion_specs_for_layer(self, layer_idx: int) -> list[ConversionSpec]:
        raise NotImplementedError(f"get_conversion_specs_for_layer is not implemented for {self.__class__.__name__}")

    @property
    def non_layer_specs(self) -> tuple[ConversionSpec, ...]:
        raise NotImplementedError(f"non_layer_specs is not implemented for {self.__class__.__name__}")

    def build_slots(self, parallel_dims: ParallelDims, default_conversion: str, base_dtype: torch.dtype) -> list[Slot]:
        state_dict = self.state_dict()
        slots: list[Slot] = []

        for layer_idx in range(self.config.num_hidden_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            conversion_specs = self.get_conversion_specs_for_layer(layer_idx)

            for spec in conversion_specs:
                slots.extend(
                    build_slots_for_conversion_spec(
                        spec,
                        prefix=layer_prefix,
                        state_dict=state_dict,
                        parallel_dims=parallel_dims,
                        default_conversion=default_conversion,
                        base_dtype=base_dtype,
                    )
                )

        for spec in self.non_layer_specs:
            slots.extend(
                build_slots_for_conversion_spec(
                    spec,
                    prefix="",
                    state_dict=state_dict,
                    parallel_dims=parallel_dims,
                    default_conversion=default_conversion,
                    base_dtype=base_dtype,
                )
            )
        return slots


__all__ = ["PreTrainedModelPrimeRL"]
