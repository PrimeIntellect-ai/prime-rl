from dataclasses import dataclass
from typing import Literal

from torch import Tensor
from transformers.modeling_utils import PreTrainedModel

CPStyle = Literal["ring", "ulysses"]
ALL_CP_STYLES: frozenset[CPStyle] = frozenset({"ring", "ulysses"})

# Suffix pairing a scale with its weight between `quantize_shard_for_weight_transfer` and
# `convert_to_hf`. A colon cannot occur in a module path, so the pair cannot be mistaken for a
# real parameter whose own name ends in "scale" -- DeepSeek V4 has several.
WEIGHT_TRANSFER_SCALE_SUFFIX = ":scale"


@dataclass(frozen=True)
class CPSupport:
    """Which context-parallel styles an architecture can train under, and why not the rest."""

    styles: frozenset[CPStyle]
    reason: str = ""


class PreTrainedModelPrimeRL(PreTrainedModel):
    """
    Base class for all PrimeRL models that extends HuggingFace PreTrainedModel.

    Provides a unified interface for state dict conversion between different formats
    (e.g., HuggingFace format vs. training-optimized format) and buffer initialization
    after loading with meta device.
    """

    @classmethod
    def cp_support(cls, config) -> CPSupport:
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

    def quantize_for_weight_transfer(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Re-encode an HF-format state dict into the representation the inference engine expects.

        Wire representation only, never a saved checkpoint: this runs inside the transport,
        after ``convert_to_hf``, so ``dcp_save`` and the HF export path never see it. Where
        ``keep_in_fp32_for_weight_transfer`` picks a dtype per tensor, this picks a whole
        representation, which is what a checkpoint quantized on disk needs: vLLM builds its
        parameters from that checkpoint's ``quantization_config``, so a dequantized broadcast
        does not fit them. Implementations may mutate in-place; callers must use the return
        value. Default is the identity.
        """
        return state_dict

    def quantize_shard_for_weight_transfer(self, name: str, shard: Tensor) -> tuple[Tensor, Tensor] | None:
        """Quantize one dim-0 shard of a parameter, on its own device, ahead of the all-gather.

        Where ``quantize_for_weight_transfer`` re-encodes a whole gathered state dict on CPU,
        this claims parameters one at a time, while they are still sharded across ranks and
        still on the accelerator. Only claim a parameter whose encoding is independent along
        dim 0: the caller all-gathers the result as if it were the original shard, so a format
        whose blocks run along the last dim qualifies and anything reducing over dim 0 does not.

        Returns ``(weight, scale)``, or ``None`` to leave the parameter on the plain gather.
        The scale is gathered alongside the weight and reaches ``convert_to_hf`` under
        ``name + WEIGHT_TRANSFER_SCALE_SUFFIX``, which is where the model decides what the
        checkpoint calls its sibling. The decision must follow from ``name`` alone and come out
        the same on every rank, because the gathers it commits the caller to are collective.
        Default is to claim nothing.
        """
        return None

    @classmethod
    def from_config(cls, config, **kwargs):
        """Public from_config that mirrors the Auto class API."""
        return cls._from_config(config, **kwargs)

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """PrimeRL models use custom MoE implementations and don't support dynamic experts implementation."""
        return False

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        """Bypass transformers' flash attention availability checks.

        PrimeRL custom models dispatch attention through their own ``ATTN_IMPL2CLASS``
        dictionaries, not through transformers' ``ALL_ATTENTION_FUNCTIONS``.  The default
        ``_check_and_adjust_attn_implementation`` validates that the requested flash
        attention package is installed and the device is supported, which fails on
        CPU-only machines and is unnecessary because we never call transformers'
        attention dispatch for custom models.
        """
        if attn_implementation is None:
            attn_implementation = "flash_attention_3"
        return attn_implementation

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
    def conversion_chain(cls, config) -> list:
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


__all__ = ["ALL_CP_STYLES", "WEIGHT_TRANSFER_SCALE_SUFFIX", "CPStyle", "CPSupport", "PreTrainedModelPrimeRL"]
