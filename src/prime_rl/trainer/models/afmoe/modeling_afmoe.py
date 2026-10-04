import torch
from torch import Tensor, nn

from prime_rl.trainer.models.afmoe.configuration_afmoe import AfmoeConfig
from prime_rl.trainer.models.afmoe.converting_afmoe import conversion_chain
from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class AfmoeFlashAttention(FlashAttention):
    """``FlashAttention`` with a sigmoid output gate. Rotary embeddings and the sliding window apply to local layers only."""

    def __init__(self, config: AfmoeConfig, layer_idx: int, flash_attn_version: int):
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=False,
            use_qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
        )
        super().__init__(attn_config, flash_attn_version=flash_attn_version)
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> tuple[Tensor, None]:
        if not self.is_local_attention:
            position_embeddings = None
        attn_output = self.attend(hidden_states, position_embeddings, cu_seqlens, max_seqlen)
        attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(attn_output), None


_FLASH_ATTN_VERSIONS = {"flash_attention_2": 2, "flash_attention_3": 3, "flash_attention_4": 4}


class AfmoeDecoderLayer(nn.Module):
    def __init__(self, config: AfmoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = AfmoeFlashAttention(
            config, layer_idx, flash_attn_version=_FLASH_ATTN_VERSIONS[config.attn_implementation]
        )

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
