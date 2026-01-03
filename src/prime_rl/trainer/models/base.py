from torch import Tensor
from transformers.modeling_utils import PreTrainedModel

from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear, VanillaOutputLinear
from prime_rl.utils.logger import get_logger


class PreTrainedModelPrimeRL(PreTrainedModel):
    """
    Base class for all PrimeRL models that extends HuggingFace PreTrainedModel.

    This class provides a unified interface for state dict conversion between different
    formats (e.g., HuggingFace format vs. training-optimized format) and buffer initialization
    after loading with meta device.
    """

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

    def init_buffers_post_meta(self) -> None:
        """
        Initialize buffers that are not in the state dict after loading with meta device.

        Some models have buffers (non-trainable tensors) that are not saved in the state dict
        but need to be properly initialized after loading the model on meta device and then
        moving to the actual device. This method should initialize such buffers.

        This is called after loading the model from a checkpoint with meta device.
        """
        raise NotImplementedError(f"init_buffers_post_meta is not implemented for {self.__class__.__name__}")

    def wrap_lm_head(self, chunk_size: int | None = None) -> None:
        old_lm_head = self.lm_head

        logger = get_logger()
        logger.info(f"Wrapping LM head with chunk size {chunk_size}")

        if chunk_size is not None:
            self.lm_head = FusedOutputLinear(
                in_features=old_lm_head.in_features, out_features=old_lm_head.out_features, chunk_size=chunk_size
            )
        else:
            self.lm_head = VanillaOutputLinear(
                in_features=old_lm_head.in_features, out_features=old_lm_head.out_features
            )

        self.lm_head.weight = old_lm_head.weight
        del old_lm_head


__all__ = ["PreTrainedModelPrimeRL"]
