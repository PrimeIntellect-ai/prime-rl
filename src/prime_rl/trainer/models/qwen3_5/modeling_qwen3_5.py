import functools
from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel as HFQwen3_5PreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedAttentionConfig,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeGatedFlashAttention,
    Qwen3_5MoeGatedSDPAAttention,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRotaryEmbedding,
    normalize_qwen3_5_attn_implementation,
)
from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids


class Qwen3_5GatedSDPAAttention(Qwen3_5MoeGatedSDPAAttention):
    pass


class Qwen3_5GatedFlashAttention(Qwen3_5MoeGatedFlashAttention):
    pass


QWEN35_ATTN_IMPL2CLASS = {
    "sdpa": Qwen3_5GatedSDPAAttention,
    "flash_attention_2": functools.partial(Qwen3_5GatedFlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(Qwen3_5GatedFlashAttention, flash_attn_version=3),
    "fa4": functools.partial(Qwen3_5GatedFlashAttention, flash_attn_version=4),
}


def _get_gated_attention(config: Qwen3_5TextConfig) -> nn.Module:
    attn_config = Qwen3_5MoeGatedAttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
    )

    attn_impl = normalize_qwen3_5_attn_implementation(config._attn_implementation)
    config._attn_implementation = attn_impl

    if attn_impl not in QWEN35_ATTN_IMPL2CLASS:
        supported = list(QWEN35_ATTN_IMPL2CLASS.keys())
        raise ValueError(
            f"Qwen3.5 attention does not support '{config._attn_implementation}'. "
            f"Supported implementations: {supported}."
        )

    return QWEN35_ATTN_IMPL2CLASS[attn_impl](attn_config)


def _create_rotary_emb(config: Qwen3_5TextConfig) -> Qwen3_5MoeRotaryEmbedding:
    return Qwen3_5MoeRotaryEmbedding(config)


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = _get_gated_attention(config)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )
        self.mlp = MLP(mlp_config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cu_seqlens=cu_seqlens)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5PreTrainedModel(PreTrainedModelPrimeRL, HFQwen3_5PreTrainedModel):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": Qwen3_5DecoderLayer,
    }

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        attn_impl = normalize_qwen3_5_attn_implementation(attn_implementation or "sdpa")
        if attn_impl not in QWEN35_ATTN_IMPL2CLASS:
            supported = list(QWEN35_ATTN_IMPL2CLASS.keys())
            raise ValueError(
                f"Qwen3.5 attention does not support '{attn_implementation}'. Supported implementations: {supported}."
            )
        return attn_impl

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


class Qwen3_5Model(Qwen3_5PreTrainedModel):
    def __init__(self, config: Qwen3_5TextConfig):
        config._attn_implementation = normalize_qwen3_5_attn_implementation(config._attn_implementation)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _create_rotary_emb(config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        flash_attn_enabled = self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4")
        if flash_attn_enabled:
            cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _checkpoint_conversion_mapping = {}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = Qwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Union[torch.Tensor, None] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None, "use_cache is not supported for custom qwen3_5 for now"
        assert past_key_values is None, "past_key_values is not supported for custom qwen3_5 for now"

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            elif input_ids is not None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        lm_rope = self.model.rotary_emb
        if hasattr(lm_rope, "rope_init_fn"):
            inv_freq, lm_rope.attention_scaling = lm_rope.rope_init_fn(lm_rope.config, lm_rope.inv_freq.device)
            lm_rope.inv_freq.copy_(inv_freq)


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5GatedFlashAttention",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
]
