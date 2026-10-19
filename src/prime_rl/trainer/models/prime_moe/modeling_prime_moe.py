"""The pretraining MoE: GLM-4.5's architecture (prime-rl's `glm4_moe`) with Gated Residual connections and null
experts. Module names follow `glm4_moe`, so checkpoints share its state-dict layout."""

from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import conversion_chain
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig
from prime_rl.trainer.models.prime_moe.configuration_prime_moe import PrimeMoeConfig
from prime_rl.trainer.models.prime_moe.hyper_connection import HyperConnection
from prime_rl.trainer.models.prime_moe.moe import PrimeMoE, PrimeRouter
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


class PrimeMoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PrimeMoeConfig, layer_idx: int):
        super().__init__()
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
            attention_bias=config.attention_bias,
            use_qk_norm=config.use_qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.self_attn = ATTN_IMPL2CLASS[config._attn_implementation](attn_config)

        if layer_idx >= config.first_k_dense_replace:
            shared_expert = None
            if config.n_shared_experts > 0:
                shared_expert = FeedForward(
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size * config.n_shared_experts,
                    expert_type="gated",
                    activation=config.hidden_act,
                )
            self.mlp = PrimeMoE(
                router=PrimeRouter(
                    dim=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_null_experts=config.n_null_experts,
                    top_k=config.num_experts_per_tok,
                    route_norm=config.norm_topk_prob,
                    route_scale=config.routed_scaling_factor,
                ),
                experts=GroupedExperts(
                    dim=config.hidden_size,
                    hidden_dim=config.moe_intermediate_size,
                    num_experts=config.n_routed_experts,
                    expert_type="gated",
                    activation=config.hidden_act,
                ),
                shared_expert=shared_expert,
            )
        else:
            self.mlp = FeedForward(
                dim=config.hidden_size, hidden_dim=config.intermediate_size, activation=config.hidden_act
            )

        self.gated_residual = config.hc_count > 1
        if self.gated_residual:
            connection_args = {
                "hidden_size": config.hidden_size,
                "stream_count": config.hc_count,
                "low_rank": config.hc_lowrank,
                "norm_eps": config.rms_norm_eps,
            }
            self.attn_hyper_connection = HyperConnection(**connection_args)
            self.mlp_hyper_connection = HyperConnection(**connection_args)
        else:
            self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
            self.post_attention_layernorm = RMSNorm(
                RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if self.gated_residual:
            block_input, residual_state = self.attn_hyper_connection.mix(hidden_states)
            block_output, _ = self.self_attn(
                hidden_states=block_input,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hidden_states = self.attn_hyper_connection.combine(block_output, residual_state)
            block_input, residual_state = self.mlp_hyper_connection.mix(hidden_states)
            block_output = self.mlp(block_input)
            return self.mlp_hyper_connection.combine(block_output, residual_state)

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
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class PrimeMoePreTrainedModel(PreTrainedModelPrimeRL):
    config: PrimeMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PrimeMoeDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return name.endswith("mlp.router.selection_bias")

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config):
        # Same expert layout as glm4_moe.
        return conversion_chain(config)


class PrimeMoeModel(PrimeMoePreTrainedModel):
    def __init__(self, config: PrimeMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PrimeMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.hc_count > 1:
            self.hyper_connection_mixer = HyperConnection(
                hidden_size=config.hidden_size,
                stream_count=config.hc_count,
                low_rank=config.hc_lowrank,
                norm_eps=config.rms_norm_eps,
                with_residual_injection=False,
            )
        else:
            self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        if isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"
        self.rotary_emb = RotaryEmbedding(
            RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings, rope_type=rope_type, model_config=config
            )
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        # Gated Residual: every residual stream starts from the token embedding.
        hidden_states = inputs_embeds.repeat(1, 1, self.config.hc_count)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, position_embeddings=position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

        if self.config.hc_count > 1:
            hidden_states, _ = self.hyper_connection_mixer(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class PrimeMoeForCausalLM(PrimeMoePreTrainedModel):
    def __init__(self, config: PrimeMoeConfig):
        super().__init__(config)
        self.model = PrimeMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        """``labels`` and ``temperature`` go to prime-rl's wrapped LM head, which then returns per-token logprobs
        and entropy; without them it returns logits."""
        assert use_cache is None and past_key_values is None, "prime_moe does not support KV caching"
        if position_ids is None:
            tokens = inputs_embeds if inputs_embeds is not None else input_ids
            position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        rotary_emb = self.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)
