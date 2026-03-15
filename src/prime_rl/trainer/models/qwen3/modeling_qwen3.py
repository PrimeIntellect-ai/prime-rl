from torch import Tensor, nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig


def _replace_qwen3_rms_norm_modules(module: nn.Module, *, impl: str) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, Qwen3RMSNorm):
            replacement = RMSNorm(
                RMSNormConfig(hidden_size=child.weight.shape[0], eps=child.variance_epsilon, impl=impl)
            )
            replacement.weight = child.weight
            setattr(module, name, replacement)
            continue
        _replace_qwen3_rms_norm_modules(child, impl=impl)


class Qwen3ForCausalLM(HFQwen3ForCausalLM, PreTrainedModelPrimeRL):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        _replace_qwen3_rms_norm_modules(self, impl=getattr(config, "rms_norm_impl", "torch"))

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict is in HuggingFace format."""
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict is in PrimeRL training format."""
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert state dict from PrimeRL training format to HuggingFace format in-place."""
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert state dict from HuggingFace format to PrimeRL training format in-place."""
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's state dict from PrimeRL format to HuggingFace format in-place."""
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's state dict from HuggingFace format to PrimeRL format in-place."""
        return state_dict

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["Qwen3ForCausalLM"]
from torch import Tensor, nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig


def _replace_qwen3_rms_norm_modules(module: nn.Module, *, impl: str) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, Qwen3RMSNorm):
            replacement = RMSNorm(
                RMSNormConfig(hidden_size=child.weight.shape[0], eps=child.variance_epsilon, impl=impl)
            )
            replacement.weight = child.weight
            setattr(module, name, replacement)
            continue
        _replace_qwen3_rms_norm_modules(child, impl=impl)


class Qwen3ForCausalLM(HFQwen3ForCausalLM, PreTrainedModelPrimeRL):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        _replace_qwen3_rms_norm_modules(self, impl=getattr(config, "rms_norm_impl", "torch"))

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict is in HuggingFace format."""
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict is in PrimeRL training format."""
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert state dict from PrimeRL training format to HuggingFace format in-place."""
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert state dict from HuggingFace format to PrimeRL training format in-place."""
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's state dict from PrimeRL format to HuggingFace format in-place."""
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's state dict from HuggingFace format to PrimeRL format in-place."""
        return state_dict

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["Qwen3ForCausalLM"]
