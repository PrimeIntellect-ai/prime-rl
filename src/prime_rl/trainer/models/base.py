from torch import Tensor
from transformers.modeling_utils import PreTrainedModel


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

    def conversion_chain(self) -> "list":
        """Declarative HF<->prime conversion as a list of
        :class:`~prime_rl.trainer.models.conversion_ops.ConvOp`.

        Each model defines this in its ``converting_<model>.py`` and the model
        class returns it here (using ``self.config``). The ``convert_*`` methods
        below play it forward (HF -> prime) or backward (prime -> HF); since
        every op is present-guarded, the same chain works on a full state dict,
        a single layer's keys, or a local shard. Models whose HF and prime
        layouts coincide return an empty chain."""
        raise NotImplementedError(f"conversion_chain is not implemented for {self.__class__.__name__}")

    def convert_to_hf(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a state dict from PrimeRL training format to HuggingFace
        format in-place (saving checkpoints, broadcasting to inference)."""
        from prime_rl.trainer.models.conversion_ops import apply_tt_to_hf

        return apply_tt_to_hf(state_dict, self.conversion_chain())

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a state dict from HuggingFace format to PrimeRL training
        format in-place (loading pretrained HF models)."""
        from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt

        return apply_hf_to_tt(state_dict, self.conversion_chain())

    def convert_layer_to_hf(self, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Per-layer PrimeRL -> HuggingFace (NCCL broadcast). Present-guarded
        ops make playing the full chain over a single layer's keys equivalent
        to converting just that layer."""
        return self.convert_to_hf(state_dict)

    def convert_layer_to_prime(self, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Per-layer HuggingFace -> PrimeRL (see :meth:`convert_layer_to_hf`)."""
        return self.convert_to_prime(state_dict)

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
