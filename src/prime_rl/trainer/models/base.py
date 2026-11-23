from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.models.layers.ngram import NGramTable
from prime_rl.trainer.models.layers.rotary_emb import ROPE_INIT_FUNCTIONS
from prime_rl.utils.cp import CPContext


@runtime_checkable
class AttentionFlops(Protocol):
    """A sequence-mixing layer (softmax, sliding-window, sparse or linear attention, Mamba)."""

    def attention_flops(self, seq_len: int) -> int:
        """Training FLOPs of this layer's sequence mixing for one document of `seq_len` tokens."""
        ...


@dataclass
class TextModelOutput:
    """Final hidden states of an inner text model."""

    last_hidden_state: Tensor


@dataclass
class VisionModelOutput:
    """Per-patch hidden states of a vision encoder and the merged embeddings it feeds the text model."""

    last_hidden_state: Tensor
    pooler_output: Tensor


def count_active_params(model: nn.Module, params: Iterable[nn.Parameter] | None = None) -> int:
    """Elements of `params` (default: all of the model's) that a token multiplies through.

    Every parameter counts (attention and MLP projections, norms, routers, the lm_head) except the
    routed experts a token skips (`1 - top_k / num_experts` of them) and embedding tables (n-gram
    tables included), which count only when tied to the lm_head.
    """
    params = {id(param): param for param in (model.parameters() if params is None else params)}
    lm_head_params = {id(param) for param in model.lm_head.parameters()}
    inactive = 0
    for module in model.modules():
        if isinstance(module, (nn.Embedding, NGramTable)):
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


class PreTrainedModelPrimeRL(nn.Module):
    """
    Base class for all PrimeRL models.

    Builds a model from its HF config, initializes and ties its weights, and provides a unified interface
    for state dict conversion between different formats (e.g., HuggingFace format vs. training-optimized
    format) and buffer initialization after loading with meta device.
    """

    cp_context: CPContext = CPContext()
    ties_word_embeddings: ClassVar[bool] = False

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if config._attn_implementation is None:
            config._attn_implementation = "flash_attention_3"

    @classmethod
    def from_config(cls, config, *, attn_implementation: str | None = None, dtype: torch.dtype | None = None):
        """Build the model with `dtype` (default `config.dtype`) as torch's default dtype, recorded on the config."""
        dtype = config.dtype if dtype is None else dtype
        config.dtype = dtype
        for sub_config_key in config.sub_configs:
            if (sub_config := getattr(config, sub_config_key)) is not None:
                sub_config.dtype = dtype
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        previous_dtype = torch.get_default_dtype()
        if dtype is not None:
            torch.set_default_dtype(dtype)
        try:
            return cls(config)
        finally:
            torch.set_default_dtype(previous_dtype)

    def post_init(self) -> None:
        """Initialize the weights unless building on the meta device, then tie them."""
        if torch.empty(0).device.type != "meta":
            self.initialize_weights()
        self.tie_weights()

    def tie_weights(self) -> None:
        """Share the input embedding weight with `lm_head` when the class ties them and the config asks for it."""
        if self.ties_word_embeddings and getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.get_input_embeddings().weight

    def get_input_embeddings(self) -> nn.Embedding:
        """The input embedding table: `embed_tokens` on an inner model, `model.embed_tokens` on a causal LM."""
        embed_tokens = getattr(self, "embed_tokens", None)
        return embed_tokens if embed_tokens is not None else self.model.embed_tokens

    @torch.no_grad()
    def initialize_weights(self) -> None:
        """Run `_init_weights` on each module not yet initialized, children first, nested models with their own."""

        def initialize(module: nn.Module, initialize_module: Callable[[nn.Module], None]) -> None:
            for child in module.children():
                if isinstance(child, PreTrainedModelPrimeRL):
                    initialize(child, child._initialize_module)
                else:
                    initialize(child, initialize_module)
            initialize_module(module)

        initialize(self, self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if getattr(module, "_weights_initialized", False):
            return
        if next(chain(module.parameters(recurse=False), module.buffers(recurse=False)), None) is not None:
            self._init_weights(module)
        module._weights_initialized = True

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        """Why this architecture cannot train at context-parallel degree `cp`, or ``None`` if it can.

        The default is the Ulysses constraint: query heads, and KV heads unless replicated, shard over `cp`.
        """
        text_config = getattr(config, "text_config", config)
        q_heads = text_config.num_attention_heads
        kv_heads = getattr(text_config, "num_key_value_heads", None) or q_heads
        if q_heads % cp != 0 or (kv_heads % cp != 0 and cp % kv_heads != 0):
            return f"cp={cp} must divide the {q_heads} attention heads and divide or be a multiple of the {kv_heads} KV heads for Ulysses CP"
        return None

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        """Whether a tensor is stored in FP32 in the source checkpoint.

        Runtime upcasts for training or inference do not change the wire dtype.
        """
        return False

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

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the parameters and buffers `module` owns directly."""
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        module_name = type(module).__name__
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            if getattr(module, "weight", None) is not None:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif "LayerNorm" in module_name or "RMSNorm" in module_name:
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif "RotaryEmbedding" in module_name and hasattr(module, "original_inv_freq"):
            inv_freq, _ = ROPE_INIT_FUNCTIONS[module.rope_type](module.config)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq.copy_(inv_freq)
        if isinstance(module, MoE):
            module.init_weights(self.config.get_text_config().initializer_range, module.tokens_per_expert.device)

    def init_buffers_post_meta(self) -> None:
        """
        Initialize buffers that are not in the state dict after loading with meta device.

        Some models have buffers (non-trainable tensors) that are not saved in the state dict
        but need to be properly initialized after loading the model on meta device and then
        moving to the actual device. This method should initialize such buffers.

        This is called after loading the model from a checkpoint with meta device.
        """
        raise NotImplementedError(f"init_buffers_post_meta is not implemented for {self.__class__.__name__}")


__all__ = ["PreTrainedModelPrimeRL", "TextModelOutput", "VisionModelOutput"]
