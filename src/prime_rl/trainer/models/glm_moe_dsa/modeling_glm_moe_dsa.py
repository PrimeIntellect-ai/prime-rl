import warnings

import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.trainer.models.base import PrimeModel
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import conversion_chain
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding
from prime_rl.utils.cp import CPContext


def _sparse_mla_attention_args(config: GlmMoeDsaConfig, layer_idx: int) -> SparseMlaAttentionArgs:
    return SparseMlaAttentionArgs(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_head_dim=config.qk_head_dim,
        v_head_dim=config.v_head_dim,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        use_index_cache=config.use_index_cache,
        skip_topk=config.skips_topk(layer_idx),
    )


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GlmMoeDsaAttention(_sparse_mla_attention_args(config, layer_idx))

        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            expert_type="gated",
            activation=config.hidden_act,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=1e-3,
        )
        if layer_idx >= config.first_k_dense_replace:
            shared_expert = None
            if config.n_shared_experts > 0:
                shared_expert = FeedForward(
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size * config.n_shared_experts,
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

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        ks: Tensor,
        ke: Tensor,
        cached_indices: Tensor | None = None,
        routed_experts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cached_indices = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            ks=ks,
            ke=ke,
            cached_indices=cached_indices,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
        return residual + hidden_states, cached_indices


class GlmMoeDsaModel(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.use_index_cache = config.use_index_cache
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = RotaryEmbedding(
            config.rope_parameters, config.qk_rope_head_dim, config.max_position_embeddings
        )
        self.cp_context = CPContext()

    def _gather_position_ids_for_cp(self, position_ids: Tensor) -> Tensor:
        gathered_position_ids = [torch.empty_like(position_ids) for _ in range(self.cp_context.cp_world_size)]
        dist.all_gather(gathered_position_ids, position_ids.contiguous(), group=self.cp_context.cp_group)
        return torch.cat(gathered_position_ids, dim=1)

    def forward(self, input_ids: Tensor, position_ids: Tensor, routed_experts: Tensor | None = None) -> Tensor:
        """Document boundaries derive from ``position_ids``: sparse MLA builds its varlen indices from them.

        ``routed_experts`` (``[batch, seq, num_layers, top_k]``) replays the inference router's choices.
        """
        hidden_states = self.embed_tokens(input_ids)

        cp_rank, cp_world_size = self.cp_context.cp_rank, self.cp_context.cp_world_size
        if self.cp_context.cp_enabled:
            position_ids_full = self._gather_position_ids_for_cp(position_ids)
        else:
            position_ids_full = position_ids

        flat_position_ids = position_ids_full.view(-1)
        S_full = flat_position_ids.shape[0]
        ks_full = torch.arange(S_full, dtype=torch.int32, device=flat_position_ids.device) - flat_position_ids.to(
            torch.int32
        )
        ke_full = torch.arange(1, S_full + 1, dtype=torch.int32, device=flat_position_ids.device)

        # Position embeddings are computed over the full sequence: K uses cos/sin[0:S_full]
        # while Q uses the local CP slice. ks/ke are computed in K's global coordinate
        # system and then sharded to the local Q range so the indexer's per-token
        # causal/varlen mask aligns with the gathered K.
        position_embeddings = self.rotary_emb(hidden_states, position_ids_full)

        if cp_world_size > 1:
            s_local = S_full // cp_world_size
            ks = ks_full[cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()
            ke = ke_full[cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()
        else:
            ks, ke = ks_full, ke_full

        cached_indices = None
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx, :] if routed_experts is not None else None
            hidden_states, next_cached_indices = decoder_layer(
                hidden_states,
                position_embeddings,
                ks,
                ke,
                cached_indices=cached_indices,
                routed_experts=layer_routed_experts,
            )
            cached_indices = next_cached_indices if self.use_index_cache else None

        return self.norm(hidden_states)


class GlmMoeDsaForCausalLM(PrimeModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.lm_head = VanillaOutputLinear(config.hidden_size, config.vocab_size)

        warnings.warn("GlmMoeDsaForCausalLM is experimental, higher trainer<->inference KL mismatch may be observed.")
        warnings.warn("`model.attn` is ignored, GlmMoeDsa uses only sparse attention.")

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith("mlp.router.selection_bias")

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Per-expert keys, or the fused expert keys of newer HF checkpoints.
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config: GlmMoeDsaConfig):
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
        # Sparse MLA derives document boundaries from position_ids, so seq_lens is unused.
        hidden_states = self.model(input_ids, position_ids, routed_experts)
        return self.lm_head(hidden_states, labels, temperature=temperature, sampling_mask=sampling_mask)

    def init_buffers_post_meta(self) -> None:
        self.model.rotary_emb.reset_parameters()
