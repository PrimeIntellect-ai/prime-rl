from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig

from .afmoe_attn import AFMOE_ATTN_IMPL2CLASS, AfmoeAttentionConfig
from .configuration_afmoe import AfmoeConfig
from .converting_afmoe import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)


def _create_rotary_emb(config: AfmoeConfig) -> RotaryEmbedding:
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
    else:
        rope_type = "default"

    rotary_config = RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type=rope_type,
        model_config=config,
    )
    return RotaryEmbedding(rotary_config)


def _get_afmoe_attention(config: AfmoeConfig, layer_idx: int) -> nn.Module:
    is_local = config.layer_types[layer_idx] == "sliding_attention"

    attn_config = AfmoeAttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        is_local_attention=is_local,
        sliding_window=config.sliding_window if is_local else None,
        attention_dropout=config.attention_dropout,
    )

    attn_impl = config._attn_implementation
    if attn_impl == "eager":
        attn_impl = "sdpa"

    if attn_impl not in AFMOE_ATTN_IMPL2CLASS:
        supported = list(AFMOE_ATTN_IMPL2CLASS.keys())
        raise ValueError(
            f"AFMoE attention does not support '{config._attn_implementation}'. "
            f"Supported implementations: {supported}. "
            f"Note: flash_attention is not supported for AFMoE due to sliding window + RoPE constraints."
        )

    return AFMOE_ATTN_IMPL2CLASS[attn_impl](attn_config)


class AfmoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = _get_afmoe_attention(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        self.pre_mlp_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_mlp_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        self.moe_enabled = layer_idx >= config.num_dense_layers
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )
        moe_args = MoEArgs(
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            score_func=config.score_func,
            route_norm=config.route_norm,
            route_scale=config.route_scale,
            score_before_experts=getattr(config, "score_before_experts", False),
            top_k=config.num_experts_per_tok,
            use_grouped_mm=getattr(config, "use_grouped_mm", True),
            load_balance_coeff=getattr(config, "load_balance_coeff", None),
        )
        if self.moe_enabled:
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
            self.mlp = MLP(mlp_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AfmoePreTrainedModel(PreTrainedModelPrimeRL):
    config_class = AfmoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["AfmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_mlp_layernorm",
        "q_norm",
        "k_norm",
        "norm",
    ]
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": AfmoeDecoderLayer,
    }
    supports_gradient_checkpointing = True

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in module_name for module_name in state_dict.keys())

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> bool:
        return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict


class AfmoeModel(AfmoePreTrainedModel):
    _no_split_modules = ["AfmoeDecoderLayer"]

    def __init__(self, config: AfmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                AfmoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = _create_rotary_emb(config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        if self.config.mup_enabled:
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            mask = causal_mask_mapping[decoder_layer.attention_type]

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


class AfmoeForCausalLM(AfmoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = AfmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_type_ids: Optional[torch.Tensor] = None,  # will be ignored
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        assert use_cache is None, "use_cache is not supported for custom afmoe for now"
        assert past_key_values is None, "past_key_values is not supported for custom afmoe for now"

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)


        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = [
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
]
