import functools
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prime_rl.trainer.models.afmoe.configuration_afmoe import AfmoeConfig
from prime_rl.trainer.models.afmoe.converting_afmoe import conversion_chain
from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.layers.attn import (
    flash_attn_2_varlen_op,
    flash_attn_3_varlen_op,
    flash_attn_4_varlen_op,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, apply_rotary_pos_emb
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


@dataclass
class AfmoeAttentionConfig:
    """Configuration for AFMoE attention layers."""

    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    is_local_attention: bool
    sliding_window: int | None = None


class AfmoeAttentionBase(nn.Module):
    def __init__(self, config: AfmoeAttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.is_local_attention
        self.sliding_window = config.sliding_window if config.is_local_attention else None

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)

        # QK normalization
        self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))


class AfmoeFlashAttention(AfmoeAttentionBase):
    """AFMoE attention using Flash Attention varlen functions."""

    _funcs = {
        2: flash_attn_2_varlen_op,
        3: flash_attn_3_varlen_op,
        4: flash_attn_4_varlen_op,
    }

    def __init__(self, config: AfmoeAttentionConfig, flash_attn_version: int = 4):
        super().__init__(config)
        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, max_seqlen):
        """Run the flash attention kernel. q/k/v are [total_tokens, heads, dim]."""
        kwargs: dict = {"causal": True}
        if self.sliding_window is not None:
            kwargs["window_size"] = (self.sliding_window - 1, 0)
        if self._flash_attn_version == 4:
            out, _ = self.func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, **kwargs)
        else:
            out = self.func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.is_local_attention:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        attn_output = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)
        attn_output = attn_output.contiguous().view(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate_states)
        return self.o_proj(attn_output), None


AFMOE_ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(AfmoeFlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(AfmoeFlashAttention, flash_attn_version=3),
    "flash_attention_4": functools.partial(AfmoeFlashAttention, flash_attn_version=4),
}


def _get_afmoe_attention(config: AfmoeConfig, layer_idx: int) -> nn.Module:
    is_local = config.layer_types[layer_idx] == "sliding_attention"
    attn_config = AfmoeAttentionConfig(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        is_local_attention=is_local,
        sliding_window=config.sliding_window if is_local else None,
    )
    return AFMOE_ATTN_IMPL2CLASS[config.attn_implementation](attn_config)


class AfmoeDecoderLayer(nn.Module):
    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = _get_afmoe_attention(config, layer_idx)

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        self.pre_mlp_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_mlp_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        self.moe_enabled = layer_idx >= config.num_dense_layers
        moe_args = MoEArgs(
            num_experts=config.num_experts,
            expert_type="gated",
            activation=config.hidden_act,
            score_func=config.score_func,
            route_norm=config.route_norm,
            route_scale=config.route_scale,
            score_before_experts=config.score_before_experts,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=config.load_balance_coeff,
        )
        if self.moe_enabled:
            shared_expert = None
            if config.num_shared_experts > 0:
                shared_expert = FeedForward(
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size * config.num_shared_experts,
                    expert_type=moe_args.expert_type,
                    activation=moe_args.activation,
                )
            self.mlp = MoE.from_args(
                moe_args,
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                shared_expert=shared_expert,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                activation=config.hidden_act,
            )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return residual + hidden_states


class AfmoeModel(nn.Module):
    def __init__(self, config: AfmoeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [AfmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = RotaryEmbedding(config.rope_parameters, config.head_dim, config.max_position_embeddings)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        routed_experts: Tensor | None = None,
    ) -> Tensor:
        """``routed_experts`` (``[batch, seq, num_layers, top_k]``) replays the inference router's choices."""
        hidden_states = self.embed_tokens(input_ids)
        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=hidden_states.device),
            total_tokens=None if seq_lens_are_pre_shard else hidden_states.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        if self.config.mup_enabled:
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, cu_seqlens, max_seqlen, routed_experts=layer_routed_experts
            )
        return self.norm(hidden_states)


class AfmoeForCausalLM(PrimeModel):
    def __init__(self, config: AfmoeConfig):
        super().__init__(config)
        self.model = AfmoeModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config: AfmoeConfig):
        return conversion_chain(config)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
        sampling_mask: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> PrimeLmOutput:
        hidden_states = self.model(input_ids, position_ids, seq_lens, seq_lens_are_pre_shard, routed_experts)
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)

    def init_buffers_post_meta(self) -> None:
        self.model.rotary_emb.reset_parameters()
