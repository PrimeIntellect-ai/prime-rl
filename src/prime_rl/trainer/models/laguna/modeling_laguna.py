from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import maybe_autocast

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
from prime_rl.trainer.models.laguna.converting_laguna import (
    convert_hf_layer_to_prime,
    convert_hf_to_prime,
    convert_prime_layer_to_hf,
    convert_prime_to_hf,
)
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention, SDPAAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import FeedForward, MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids


class LagunaRotaryEmbedding(nn.Module):
    def __init__(self, config: LagunaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = list(dict.fromkeys(config.layer_types))
        self.rope_type = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: LagunaConfig | None = None,
        device: Optional[torch.device] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        rope_params = config.rope_parameters[layer_type]
        base = rope_params["rope_theta"]
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _laguna_attention_config(config: LagunaConfig, num_heads: int) -> AttentionConfig:
    return AttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=config.num_key_value_heads,
        is_causal=True,
        attention_bias=config.attention_bias,
        use_qk_norm=True,
        rms_norm_eps=config.rms_norm_eps,
        qk_norm_type="per_head",
    )


class LagunaFlashAttention(FlashAttention):
    def __init__(self, config: LagunaConfig, layer_idx: int, num_heads: int, flash_attn_version: int = 2):
        super().__init__(_laguna_attention_config(config, num_heads), flash_attn_version=flash_attn_version)
        self.num_heads = num_heads
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None
        self.g_proj = nn.Linear(config.hidden_size, num_heads, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        input_shape = hidden_states.shape[:-1]
        attn_output = attn_output.view(*input_shape, self.num_heads, self.head_dim)
        gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)
        attn_output = (attn_output * gate.unsqueeze(-1)).view(*input_shape, -1)
        return self.o_proj(attn_output), None


class LagunaSDPAAttention(SDPAAttention):
    def __init__(self, config: LagunaConfig, layer_idx: int, num_heads: int):
        super().__init__(_laguna_attention_config(config, num_heads))
        self.num_heads = num_heads
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None
        self.g_proj = nn.Linear(config.hidden_size, num_heads, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        dropout_p = 0.0 if not self.training else self.attention_dropout
        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=attention_mask is None,
        )
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(hidden_states, position_embeddings)
        attn_output = self._attention_core(query_states, key_states, value_states, attention_mask=attention_mask)
        input_shape = hidden_states.shape[:-1]
        attn_output = attn_output.view(*input_shape, self.num_heads, self.head_dim)
        gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)
        attn_output = (attn_output * gate.unsqueeze(-1)).view(*input_shape, -1)
        return self.o_proj(attn_output), None


def _get_laguna_attention(config: LagunaConfig, layer_idx: int):
    attn_impl = config._attn_implementation
    if attn_impl == "eager":
        attn_impl = "sdpa"
    num_heads = config.num_attention_heads_per_layer[layer_idx]
    match attn_impl:
        case "flash_attention_2":
            return LagunaFlashAttention(config, layer_idx, num_heads, flash_attn_version=2)
        case "flash_attention_3":
            return LagunaFlashAttention(config, layer_idx, num_heads, flash_attn_version=3)
        case "fa4":
            return LagunaFlashAttention(config, layer_idx, num_heads, flash_attn_version=4)
        case "sdpa":
            return LagunaSDPAAttention(config, layer_idx, num_heads)
        case _:
            raise ValueError(f"Laguna attention does not support '{config._attn_implementation}'.")


class LagunaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LagunaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.mlp_layer_type = config.mlp_layer_types[layer_idx]
        self.self_attn = _get_laguna_attention(config, layer_idx)

        if self.mlp_layer_type == "sparse":
            moe_args = MoEArgs(
                num_experts=config.num_experts,
                num_shared_experts=0,
                score_func="sigmoid",
                route_norm=True,
                route_scale=config.moe_routed_scaling_factor,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                use_grouped_mm=config.use_grouped_mm,
                load_balance_coeff=config.load_balance_coeff,
                fp8=getattr(config, "fp8", False),
            )
            if config.moe_router_logit_softcapping:
                raise NotImplementedError("Laguna router logit softcapping is not supported by PrimeRL MoE yet.")
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
            self.shared_expert = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.shared_expert_intermediate_size,
            )
        else:
            mlp_config = MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_act,
                bias=False,
            )
            self.mlp = MLP(mlp_config)
            self.shared_expert = None

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(mlp_input, routed_experts=routed_experts)
        if self.shared_expert is not None:
            bs, slen, dim = hidden_states.shape
            shared_output = self.shared_expert(mlp_input.view(-1, dim)).view(bs, slen, dim)
            hidden_states = hidden_states + shared_output
        hidden_states = residual + hidden_states
        return hidden_states


class LagunaPreTrainedModel(PreTrainedModelPrimeRL):
    config: LagunaConfig
    config_class = LagunaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LagunaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LagunaDecoderLayer,
    }

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "mlp.experts.0.gate_proj.weight" in name or "mlp.experts.gate_up_proj" in name or "mlp.gate.weight" in name
            for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in name for name in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_prime_to_hf(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_prime(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_prime_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_prime(state_dict, layer_idx)
        return state_dict


class LagunaModel(LagunaPreTrainedModel):
    def __init__(self, config: LagunaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LagunaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = LagunaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4"):
            cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
            causal_mask_mapping = dict.fromkeys(set(self.config.layer_types), None)
        else:
            cu_seqlens = None
            max_seqlen = None
            if isinstance(attention_mask, dict):
                causal_mask_mapping = attention_mask
            else:
                mask_kwargs = {
                    "config": self.config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": None,
                    "position_ids": position_ids,
                }
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                }

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            routed_experts_layer = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings[self.config.layer_types[layer_idx]],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                routed_experts=routed_experts_layer,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)


class LagunaForCausalLM(LagunaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: LagunaConfig):
        super().__init__(config)
        self.model = LagunaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        assert use_cache is None, "use_cache is not supported for custom Laguna"
        assert past_key_values is None, "past_key_values is not supported for custom Laguna"

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self) -> None:
        rotary_emb = self.model.rotary_emb
        for layer_type in rotary_emb.layer_types:
            rope_init_fn = rotary_emb.compute_default_rope_parameters
            if rotary_emb.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rotary_emb.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(
                rotary_emb.config,
                getattr(rotary_emb, f"{layer_type}_inv_freq").device,
                layer_type=layer_type,
            )
            getattr(rotary_emb, f"{layer_type}_inv_freq").copy_(inv_freq)
            getattr(rotary_emb, f"{layer_type}_original_inv_freq").copy_(inv_freq)
            setattr(rotary_emb, f"{layer_type}_attention_scaling", attention_scaling)

        for module in self.modules():
            if isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()
                if module.expert_bias is not None:
                    module.expert_bias.zero_()


__all__ = [
    "LagunaForCausalLM",
    "LagunaModel",
    "LagunaPreTrainedModel",
]
