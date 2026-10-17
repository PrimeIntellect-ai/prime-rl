from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from torch import Tensor, nn
from transformers.modeling_utils import PreTrainedModel

from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.utils.cp import CPContext

CPStyle = Literal["ring", "ulysses"]
ALL_CP_STYLES: frozenset[CPStyle] = frozenset({"ring", "ulysses"})


@runtime_checkable
class AttentionFlops(Protocol):
    """A sequence-mixing layer (softmax, sliding-window, sparse or linear attention, Mamba)."""

    def attention_flops(self, seq_len: int) -> int:
        """Training FLOPs of this layer's sequence mixing for one document of `seq_len` tokens."""
        ...


def count_active_params(model: nn.Module, params: Iterable[nn.Parameter] | None = None) -> int:
    """Elements of `params` (default: all of the model's) that a token multiplies through.

    Every parameter counts (attention and MLP projections, norms, routers, the lm_head) except the
    routed experts a token skips (`1 - top_k / num_experts` of them) and embedding tables, which
    count only when tied to the lm_head.
    """
    params = {id(param): param for param in (model.parameters() if params is None else params)}
    lm_head_params = {id(param) for param in model.lm_head.parameters()}
    inactive = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            inactive += sum(
                param.numel()
                for param in module.parameters(recurse=False)
                if id(param) in params and id(param) not in lm_head_params
            )
        elif isinstance(module, MoE):
            # Expert weights lead with `num_experts`, so the division is exact.
            num_experts, top_k = module.experts.num_experts, module.router.top_k
            expert_params = sum(param.numel() for param in module.experts.parameters() if id(param) in params)
            inactive += expert_params * (num_experts - top_k) // num_experts
    return sum(param.numel() for param in params.values()) - inactive


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

    cp_context: CPContext = CPContext()

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

    @classmethod
    def from_config(cls, config, trust_remote_code: bool = False, **kwargs):
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

    def num_active_params(self) -> int:
        """Parameters a text token multiplies through, LoRA adapters included."""
        return count_active_params(self)

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

    def init_buffers_post_meta(self) -> None:
        """
        Initialize buffers that are not in the state dict after loading with meta device.

        Some models have buffers (non-trainable tensors) that are not saved in the state dict
        but need to be properly initialized after loading the model on meta device and then
        moving to the actual device. This method should initialize such buffers.

        This is called after loading the model from a checkpoint with meta device.
        """
        raise NotImplementedError(f"init_buffers_post_meta is not implemented for {self.__class__.__name__}")


__all__ = ["ALL_CP_STYLES", "CPStyle", "CPSupport", "PreTrainedModelPrimeRL"]
