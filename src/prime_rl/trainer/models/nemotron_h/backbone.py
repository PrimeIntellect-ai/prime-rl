import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.nemotron_h.base import NemotronHPreTrainedModel
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.mamba import NemotronHMamba2
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class NemotronHMoE(MoE):
    def __init__(
        self,
        *,
        hidden_size: int,
        latent_size: int,
        projection_bias: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fc1_latent_proj = nn.Linear(hidden_size, latent_size, bias=projection_bias)
        self.fc2_latent_proj = nn.Linear(latent_size, hidden_size, bias=projection_bias)

    def prepare_expert_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc1_latent_proj(hidden_states)

    def prepare_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2_latent_proj(hidden_states)


class NemotronHDecoderLayer(nn.Module):
    def __init__(self, config: NemotronHConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layer_norm_epsilon))

        if self.layer_type == "mamba":
            self.mamba = NemotronHMamba2(config)
        elif self.layer_type == "attention":
            self.self_attn = ATTN_IMPL2CLASS[config._attn_implementation](
                AttentionConfig(
                    hidden_size=config.hidden_size,
                    head_dim=config.head_dim,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    is_causal=True,
                    attention_bias=config.attention_bias,
                    output_bias=config.attention_bias,
                    use_qk_norm=False,
                    rms_norm_eps=config.layer_norm_epsilon,
                )
            )
        elif self.layer_type == "moe":
            router = TokenChoiceTopKRouter(
                dim=config.hidden_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                score_func="sigmoid",
                route_norm=config.norm_topk_prob,
                route_scale=config.routed_scaling_factor,
                selection_bias=True,
                topk_sorted=False,
            )
            expert_size = config.moe_latent_size or config.hidden_size
            experts = GroupedExperts(
                dim=expert_size,
                hidden_dim=config.moe_intermediate_size,
                num_experts=config.n_routed_experts,
                expert_type="non_gated",
                activation=config.mlp_hidden_act,
                bias=config.mlp_bias,
            )
            experts.init_weights(config.initializer_range)
            shared_expert = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.moe_shared_expert_intermediate_size,
                expert_type="non_gated",
                activation=config.mlp_hidden_act,
                bias=config.mlp_bias,
            )
            moe_kwargs = {
                "router": router,
                "experts": experts,
                "shared_expert": shared_expert,
                "score_before_experts": False,
                "load_balance_coeff": config.load_balance_coeff,
            }
            if config.moe_latent_size is None:
                self.mlp = MoE(**moe_kwargs)
            else:
                self.mlp = NemotronHMoE(
                    hidden_size=config.hidden_size,
                    latent_size=config.moe_latent_size,
                    projection_bias=config.mlp_bias,
                    **moe_kwargs,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        if self.layer_type == "mamba":
            hidden_states = self.mamba(hidden_states, cu_seqlens)
        elif self.layer_type == "attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        return residual + hidden_states


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            NemotronHDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.layer_norm_epsilon))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=hidden_states.device),
            total_tokens=None if seq_lens_are_pre_shard else hidden_states.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                routed_experts=layer_routed_experts,
            )
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))
