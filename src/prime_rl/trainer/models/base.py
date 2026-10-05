from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.utils.cp import CPContext

CPStyle = Literal["ring", "ulysses"]
ALL_CP_STYLES: frozenset[CPStyle] = frozenset({"ring", "ulysses"})


@dataclass(frozen=True)
class CPSupport:
    """Which context-parallel styles an architecture can train under, and why not the rest."""

    styles: frozenset[CPStyle]
    reason: str = ""


class PrimeModel(nn.Module):
    """Base class for PrimeRL causal LMs.

    A subclass builds its modules from its ``PrimeModelConfig`` and implements a ``forward`` that takes the
    packed batch (``input_ids``, ``position_ids``, ``seq_lens``, optional ``labels``/``temperature``/
    ``sampling_mask``) and returns the LM head's ``PrimeLmOutput``. It also declares how its state dict
    converts between the HuggingFace checkpoint format and its own training format.
    """

    cp_context: CPContext = CPContext()

    def __init__(self, config: PrimeModelConfig):
        super().__init__()
        if config.tie_word_embeddings:
            raise ValueError(
                f"{config.model_type!r} checkpoint ties its LM head to the input embeddings "
                "(tie_word_embeddings=true), which PrimeRL does not support."
            )
        self.config = config

    @classmethod
    def cp_support(cls, config: PrimeModelConfig) -> CPSupport:
        """CP styles this architecture supports, given its config.

        Softmax attention runs through the shared ``FlashAttention._compute_attention``, which both
        ``substitute_ring_attn`` and ``substitute_ulysses_attn`` rebind, so the default is both styles.
        Architectures whose attention sits outside that path declare their own support.
        """
        return CPSupport(ALL_CP_STYLES)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        """Whether a tensor is stored in FP32 in the source checkpoint.

        Runtime upcasts for training or inference do not change the wire dtype.
        """
        return False

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
    def conversion_chain(cls, config: PrimeModelConfig) -> list:
        """Declarative operations converting between HF and PrimeRL state dicts."""
        return []

    def convert_to_hf(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a PrimeRL state dict to HuggingFace format in-place."""
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        return apply_prime_to_hf(state_dict, self.conversion_chain(self.config))

    def convert_to_prime(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert a HuggingFace state dict to PrimeRL format in-place."""
        from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime

        return apply_hf_to_prime(state_dict, self.conversion_chain(self.config))

    def convert_layer_to_hf(self, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert one layer from PrimeRL to HuggingFace format in-place."""
        return self.convert_to_hf(state_dict)

    def convert_layer_to_prime(self, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert one layer from HuggingFace to PrimeRL format in-place."""
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

    def init_buffers_post_meta(self) -> None:
        """
        Initialize buffers that are not in the state dict after loading with meta device.

        Some models have buffers (non-trainable tensors) that are not saved in the state dict
        but need to be properly initialized after loading the model on meta device and then
        moving to the actual device. This method should initialize such buffers.

        This is called after loading the model from a checkpoint with meta device.
        """
        raise NotImplementedError(f"init_buffers_post_meta is not implemented for {self.__class__.__name__}")


__all__ = ["ALL_CP_STYLES", "CPStyle", "CPSupport", "PrimeModel"]
