"""GLM-5.3 Flash text model.

Layers alternate Kimi Delta Attention with sparse MLA (three KDA layers, then one sparse MLA
layer), every layer is wrapped in manifold-constrained hyper-connections over `hc_mult` residual
streams, and all but the first `first_k_dense_replace` layers use a clamped-SwiGLU MoE. The
streams are averaged back into one before the final norm. The vision tower and the MTP layer of
the checkpoint are not modelled.
"""

import torch
from torch import Tensor, nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput

from prime_rl.configs.trainer import DSABackend
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4HyperConnection
from prime_rl.trainer.models.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
from prime_rl.trainer.models.glm5_next.converting_glm5_next import (
    conversion_chain,
    is_hf_state_dict,
    is_prime_state_dict,
)
from prime_rl.trainer.models.glm5_next.linear_attention import KimiDeltaAttention
from prime_rl.trainer.models.glm5_next.sparse_attention import PoolLayout, SparseMlaAttention
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, TokenSupport, VanillaOutputLinear
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


def _check_supported(config: Glm5NextTextConfig) -> None:
    assert config.hidden_act == "silu", f"hidden_act={config.hidden_act!r} is not supported"
    assert config.scoring_func == "sigmoid" and config.n_group == 1 and config.topk_group == 1
    assert config.qk_rope_head_dim == 0, "GLM-5.3 Flash attention uses no RoPE"
    assert config.num_key_value_heads == config.num_attention_heads
    assert config.index_kpool_compress and config.index_kpool_always_select_tail
    assert all(indexer_type == "full" for indexer_type in config.indexer_types), "shared indexers are not supported"


class Glm5NextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int, dsa_backend: DSABackend) -> None:
        super().__init__()
        # Token mixer: Kimi Delta Attention (linear) in three of every four layers, sparse MLA in the rest.
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.self_attn = KimiDeltaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.linear_num_heads,
                head_dim=config.linear_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                gate_lower_bound=config.linear_lower_bound,
                norm_eps=config.rms_norm_eps,
            )
        # transformers renames the checkpoint's `deepseek_sparse_attention` to `indexed_attention` on load.
        elif self.layer_type == "indexed_attention":
            self.self_attn = SparseMlaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_head_dim=config.qk_head_dim,
                v_head_dim=config.v_head_dim,
                norm_eps=config.rms_norm_eps,
                index_num_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                index_token_budget=config.index_topk,
                index_pool_size=config.index_kpool,
                dsa_backend=dsa_backend,
            )
        else:
            raise ValueError(f"Unsupported GLM-5.3 Flash layer type {self.layer_type!r}")

        # Channel mixer: MoE everywhere except the first `first_k_dense_replace` layers, which are dense.
        activation = ClampedSilu(config.swiglu_limit)
        if config.mlp_layer_types[layer_idx] == "sparse":
            router = TokenChoiceTopKRouter(
                dim=config.hidden_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                score_func="sigmoid",
                route_norm=config.norm_topk_prob,
                route_scale=config.routed_scaling_factor,
                selection_bias=True,
            )
            experts = GroupedExperts(
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                num_experts=config.n_routed_experts,
                activation=activation,
            )
            shared_expert = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size * config.n_shared_experts,
                activation=activation,
            )
            self.mlp = MoE(
                router=router,
                experts=experts,
                shared_expert=shared_expert,
                # HF scales each expert's output by its routing weight after `down_proj`.
                score_before_experts=False,
                load_balance_coeff=1e-3,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                activation=activation,
            )

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        # mHC replaces regular residual: a sublayer reads a mix of the `hc_mult` streams and writes its output back into all of them
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        mhc_states: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        pool_layout: PoolLayout,
        routed_experts: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # mhc_states: [batch, seq, hc_mult, hidden]. `collapsed` is the single-stream sublayer input;
        # `post` and `comb` are the weights for writing the output back and mixing the streams.
        post, comb, collapsed = self.attn_hc(mhc_states)
        attention_input = self.input_layernorm(collapsed)
        if self.layer_type == "linear_attention":
            # KDA resets its recurrent state at each packed-document boundary.
            attn_output = self.self_attn(attention_input, cu_seqlens)
        else:
            # Sparse MLA attends to the pools the indexer selects within the query's own document.
            attn_output = self.self_attn(attention_input, pool_layout)
        mhc_states = self.attn_hc.update_states(post, comb, attn_output, mhc_states)

        post, comb, collapsed = self.ffn_hc(mhc_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), routed_experts=routed_experts)
        return self.ffn_hc.update_states(post, comb, mlp_output, mhc_states)


class Glm5NextPreTrainedModel(PreTrainedModelPrimeRL):
    config_class = Glm5NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm5NextDecoderLayer"]
    _can_compile_fullgraph = False

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        # KDA, the indexer and sparse MLA all-gather what they need over the CP group themselves.
        return None

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith(
            ("hc.base", "hc.scale", "self_attn.A_log", "self_attn.dt_bias", "mlp.router.selection_bias")
        )

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        init_std = getattr(self.config, "text_config", self.config).initializer_range
        if isinstance(module, DeepseekV4HyperConnection):
            module.init_weights(init_std)
        elif isinstance(module, MoE):
            module.init_weights(init_std, module.tokens_per_expert.device)


class Glm5NextTextModel(Glm5NextPreTrainedModel):
    config_class = Glm5NextTextConfig

    def __init__(self, config: Glm5NextTextConfig, dsa_backend: DSABackend = "tilelang") -> None:
        super().__init__(config)
        _check_supported(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Glm5NextDecoderLayer(config, layer_idx, dsa_backend) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        """
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Unused: the model has no positional encoding, and document boundaries come from `seq_lens`.
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row. KDA, the indexer's pools and sparse MLA's
            causal ranges all restart at each document boundary.
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds the whole row's boundaries; required under context parallelism.
        """
        assert seq_lens_are_pre_shard == self.cp_context.cp_enabled, (
            f"seq_lens_are_pre_shard={seq_lens_are_pre_shard} disagrees with cp_world_size={self.cp_context.cp_world_size}"
        )
        inputs_embeds = self.embed_tokens(input_ids)
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(
            seq_lens.to(inputs_embeds.device),
            total_tokens=inputs_embeds.shape[1] * self.cp_context.cp_world_size,
        )
        # A hint, not a constraint: a one-document row (2-entry cu_seqlens) gets its own graph.
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
        pool_layout = PoolLayout.build(
            cu_seqlens,
            inputs_embeds.shape[1],
            self.cp_context.cp_rank,
            self.cp_context.cp_world_size,
            self.config.index_kpool,
        )

        mhc_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_routed_experts = routed_experts[:, :, layer_idx].contiguous() if routed_experts is not None else None
            mhc_states = decoder_layer(mhc_states, cu_seqlens, pool_layout, routed_experts=layer_routed_experts)
        return BaseModelOutput(last_hidden_state=self.norm(mhc_states.mean(dim=2)))


class Glm5NextModel(nn.Module):
    def __init__(self, config: Glm5NextConfig, dsa_backend: DSABackend = "tilelang") -> None:
        super().__init__()
        self.language_model = Glm5NextTextModel(config.text_config, dsa_backend)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutput:
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )


class Glm5NextForCausalLM(Glm5NextPreTrainedModel):
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Glm5NextConfig | Glm5NextTextConfig) -> None:
        super().__init__(config)
        text_config = getattr(config, "text_config", config)
        # get_model sets the trainer's `model.dsa_backend` on the top-level config.
        dsa_backend = getattr(config, "dsa_backend", "tilelang")
        self.model = (
            Glm5NextModel(config, dsa_backend)
            if hasattr(config, "text_config")
            else Glm5NextTextModel(config, dsa_backend)
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = VanillaOutputLinear(text_config.hidden_size, text_config.vocab_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        temperature: torch.Tensor | None = None,
        support: TokenSupport | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        return self.lm_head(
            outputs.last_hidden_state[:, slice_indices],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
            support=support,
        )

    def init_buffers_post_meta(self) -> None:
        for module in self.modules():
            if isinstance(module, MoE):
                module.tokens_per_expert.zero_()
                module.routing_confidence_sum.zero_()
                if module.router.selection_bias is not None:
                    module.router.selection_bias.zero_()


__all__ = [
    "Glm5NextDecoderLayer",
    "Glm5NextForCausalLM",
    "Glm5NextModel",
    "Glm5NextPreTrainedModel",
    "Glm5NextTextModel",
]
