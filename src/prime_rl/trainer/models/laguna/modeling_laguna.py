import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
from prime_rl.trainer.models.laguna.converting_laguna import conversion_chain
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


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
        is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if is_local_attention else None
        # Attention output gating, matching vLLM's Laguna implementation:
        #   True / "per-head" (Laguna XS.2, S): one gate per head, broadcast across head_dim
        #   "per-element"     (Laguna M):       one gate per (head, head_dim) channel
        #   False:                              no gating
        self.gating = bool(config.gating)
        self.gate_per_head = config.gating is True or config.gating == "per-head"
        if self.gating:
            gate_size = num_heads if self.gate_per_head else num_heads * self.head_dim
            self.g_proj = nn.Linear(config.hidden_size, gate_size, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> tuple[Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        attn_output = self.attend(hidden_states, position_embeddings, cu_seqlens, max_seqlen)
        attn_output = attn_output.view(*input_shape, self.num_heads, self.head_dim)
        if self.gating:
            gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)
            # per-head gates broadcast across head_dim; per-element gates are already
            # one-per-channel and line up with the [..., num_heads, head_dim] view.
            gate = gate.unsqueeze(-1) if self.gate_per_head else gate.view(*attn_output.shape)
            attn_output = attn_output * gate
        attn_output = attn_output.view(*input_shape, -1)
        return self.o_proj(attn_output), None


_FLASH_ATTN_VERSIONS = {"flash_attention_2": 2, "flash_attention_3": 3, "flash_attention_4": 4}


class LagunaDecoderLayer(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = LagunaFlashAttention(
            config,
            layer_idx,
            num_heads=config.num_attention_heads_per_layer[layer_idx],
            flash_attn_version=_FLASH_ATTN_VERSIONS[config.attn_implementation],
        )

        if config.mlp_layer_types[layer_idx] == "sparse":
            moe_args = MoEArgs(
                num_experts=config.num_experts,
                expert_type="gated",
                activation=config.hidden_act,
                score_func="sigmoid",
                route_norm=True,
                route_scale=config.moe_routed_scaling_factor,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                load_balance_coeff=config.load_balance_coeff,
            )
            if config.moe_router_logit_softcapping:
                raise NotImplementedError("Laguna router logit softcapping is not supported by PrimeRL MoE yet.")
            self.mlp = MoE.from_args(
                moe_args,
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                shared_expert=FeedForward(
                    dim=config.hidden_size,
                    hidden_dim=config.shared_expert_intermediate_size,
                    expert_type=moe_args.expert_type,
                    activation=moe_args.activation,
                ),
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                activation=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        return residual + hidden_states


class LagunaModel(nn.Module):
    def __init__(self, config: LagunaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([LagunaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        # One rotary embedding per attention layer type.
        self.rotary_emb = nn.ModuleDict(
            {
                layer_type: RotaryEmbedding(rope, config.head_dim, config.max_position_embeddings)
                for layer_type, rope in config.rope_parameters.items()
            }
        )

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
        position_embeddings = {
            layer_type: rotary_emb(hidden_states, position_ids) for layer_type, rotary_emb in self.rotary_emb.items()
        }

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings[decoder_layer.layer_type],
                cu_seqlens,
                max_seqlen,
                routed_experts=layer_routed_experts,
            )
        return self.norm(hidden_states)


class LagunaForCausalLM(PrimeModel):
    def __init__(self, config: LagunaConfig):
        super().__init__(config)
        self.model = LagunaModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "mlp.experts.0.gate_proj.weight" in name or "mlp.experts.gate_up_proj" in name or "mlp.gate.weight" in name
            for name in state_dict
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config: LagunaConfig):
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
        for rotary_emb in self.model.rotary_emb.values():
            rotary_emb.reset_parameters()

        for module in self.modules():
            if isinstance(module, MoE) and module.tokens_per_expert.device.type != "meta":
                module.tokens_per_expert.zero_()
                if module.router.selection_bias is not None:
                    module.router.selection_bias.zero_()


__all__ = ["LagunaForCausalLM", "LagunaModel"]
