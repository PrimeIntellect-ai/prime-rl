import torch
from torch import Tensor, nn

from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.gpt_oss.attention import GptOssAttention
from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss.converting_gpt_oss import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx)

        router = TokenChoiceTopKRouter(
            dim=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            score_func="topk_softmax",
            route_norm=False,
            route_scale=1.0,
            gate_bias=True,
        )
        experts = GroupedExperts(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_experts=config.num_local_experts,
            expert_type="gated",
            activation="clamped_swiglu",
            bias=True,
        )
        experts.init_weights(config.initializer_range)
        self.mlp = MoE(
            router=router,
            experts=experts,
            shared_expert=None,
            score_before_experts=False,
            load_balance_coeff=None,
        )
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        return residual + hidden_states


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, cu_seqlens, max_seqlen, routed_experts=layer_routed_experts
            )
        return self.norm(hidden_states)


class GptOssForCausalLM(PrimeModel):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__(config)
        self.model = GptOssModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config: GptOssConfig):
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


__all__ = [
    "GptOssDecoderLayer",
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssRMSNorm",
]
