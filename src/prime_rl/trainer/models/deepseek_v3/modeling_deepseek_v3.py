"""PrimeRL implementation of DeepSeek V3 (671B MoE).

DeepSeek V3 is a Mixture-of-Experts model with:
- Multi-head Latent Attention (MLA)
- Group-based expert routing with 256 routed experts + 1 shared expert
- First k layers are dense (configurable)
"""

from typing import Optional
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import auto_docstring, logging
from transformers.models.deepseek_v3 import modeling_deepseek_v3

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)
from prime_rl.trainer.models.deepseek_v3.attention_deepseek_v3 import (
    DeepSeekAttentionCore,
)

from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import (
    FeedForward,
    LatentMoE,
    NemotronHRouter,
    NonGatedGroupedExperts,
    TokenReorderer,
)
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    RotaryEmbeddingConfig,
)
from prime_rl.trainer.models.layers.moe import (
    MoE,
    MoEArgs,
    TokenChoiceTopKRouter,
    _selected_probability_mass_sum,
)
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig

from prime_rl.trainer.models.deepseek_v3.converting_deepseek_v3 import (
    convert_prime_to_hf,
    convert_hf_to_prime,
)
import math
from typing import Literal

from prime_rl.trainer import perf
from prime_rl.trainer.perf import PerfCounter, PretrainedConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids


from prime_rl.utils.logger import get_logger


class DeepSeekV3TopKRouter(TokenChoiceTopKRouter):
    """
    Why: Added 'n_group' for router selection.
    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__(
            dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=1,
        )

        self.num_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.route_scale = config.routed_scaling_factor

        assert self.num_experts // self.n_group >= 2

        if config.load_balance_coeff is not None and config.load_balance_coeff > 0:
            self.use_expert_bias = True
        else:
            self.use_expert_bias = False

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # scores shape (bs*slen, num_experts)
        assert (
            routed_experts is None or routed_experts.shape[-1] == self.top_k
        ), f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.use_expert_bias:
            scores_for_choice = scores + expert_bias
        else:
            scores_for_choice = scores

        if self.n_group > 1:

            group_scores = (
                scores_for_choice.view(
                    -1, self.n_group, self.num_experts // self.n_group
                )
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )

            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, self.num_experts // self.n_group)
                .reshape(-1, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if routed_experts is not None:
            top_scores = scores.gather(dim=1, index=routed_experts)
            selected_experts_indices = routed_experts
        elif self.use_expert_bias:
            # note: expert_bias was added before selecting scores
            _, selected_experts_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=1
            )

        routing_confidence_sum = _selected_probability_mass_sum(
            scores, top_scores, self.score_func
        )

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator

        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            routing_confidence_sum,
        )


class DeepSeekV3MoE(MoE):
    """
    Why: patching router to include 'n_groups'
    """

    def __init__(self, config: DeepseekV3Config):
        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=1.0,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=config.use_grouped_mm,
            load_balance_coeff=config.load_balance_coeff,
        )
        super().__init__(
            moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size
        )
        self.router = DeepSeekV3TopKRouter(config)


class DeepseekV3MLADecoderLayer(GradientCheckpointingLayer):
    """DeepSeek V3 layer with Multi-head Latent Attention.

    Uses MLA with compressed KV cache via q_lora_rank and kv_lora_rank.
    """

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV3Attention(config)

        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )

        if self._is_moe_layer(config, layer_idx):
            self.mlp = DeepSeekV3MoE(config)
        else:
            self.mlp = MLP(mlp_config)

        self.input_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
        self.post_attention_layernorm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

    @staticmethod
    def _is_moe_layer(config: DeepseekV3Config, layer_idx: int) -> bool:
        return layer_idx >= config.first_k_dense_replace

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
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
        hidden_states = residual + hidden_states

        return hidden_states


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV3Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeek V3.

    Uses compressed latent representations for efficient KV caching:
    - Queries are compressed to q_lora_rank
    - Key/values are compressed to kv_lora_rank
    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.rope_interleave = config.rope_interleave
        self.rope_theta = config.rope_theta

        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.q_lora_rank is None or self.q_lora_rank > 0

        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads

        self._attention = DeepSeekAttentionCore(config)

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                config.num_attention_heads * config.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = RMSNorm(
                RMSNormConfig(hidden_size=self.q_lora_rank, eps=config.rms_norm_eps)
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                config.num_attention_heads * config.qk_head_dim,
                bias=False,
            )

        if self.kv_lora_rank is not None:
            self.kv_a_proj_with_mqa = nn.Linear(
                config.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=config.attention_bias,
            )
            self.kv_a_layernorm = RMSNorm(
                RMSNormConfig(hidden_size=self.kv_lora_rank, eps=config.rms_norm_eps)
            )
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )
        else:
            raise NotImplementedError()

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if config.rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, _ = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.view(bsz, seq_len, self.num_attention_heads, self.qk_head_dim)

        q_nope, q_rot = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if self.kv_lora_rank is not None:
            kv = self.kv_a_proj_with_mqa(hidden_states)
            k_pass, k_rot = torch.split(
                kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass))
            k_pass = k_pass.view(
                bsz,
                seq_len,
                self.num_key_value_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )

            k_nope, v = torch.split(
                k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k_rot = k_rot.view(bsz, seq_len, 1, self.qk_rope_head_dim)
        else:
            raise NotImplementedError()
            # k_nope = hidden_states.view(bsz, seq_len, self.num_key_value_heads, self.qk_nope_head_dim)
            # k_cache = None
            # k_pe = hidden_states[..., -self.qk_rope_head_dim:]

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.rope_interleave:
                q_rot, k_rot = self._apply_rotary_pos_emb_interleave(
                    q_rot, k_rot, cos, sin
                )
            else:
                q_rot, k_rot = self._apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            k_rot = k_rot.expand(
                bsz, seq_len, self.num_key_value_heads, self.qk_rope_head_dim
            )

        # q,k,v = (bs, sl, nkv, d)
        q = torch.cat([q_nope, q_rot], dim=-1)
        k = torch.cat([k_nope, k_rot], dim=-1)

        attn_output = self._attention._attention_core(
            q, k, v, cu_seqlens, max_seqlen, scale=self.scaling
        )

        attn_output = attn_output.reshape(
            bsz, seq_len, self.num_attention_heads * self.v_head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _apply_rotary_pos_emb_interleave(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


@auto_docstring
class DeepseekV3PreTrainedModel(PreTrainedModelPrimeRL):
    config: DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV3MLADecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (NonGatedGroupedExperts, NemotronHRouter)):
            module.init_weights(std)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:

        # hf format
        cond1 = any(
            True if re.search(r"model.layers.\d+.mlp.gate_proj", name) else False
            for name in state_dict
        )

        # in hf-saftensors experts might be splitted
        cond2 = any(
            (
                True
                if re.search(r"model.layers.\d+.mlp.experts.0.up_proj.weight", name)
                else False
            )
            for name in state_dict
        )

        return any([cond1, cond2])

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        cond1 = any(
            True if re.search(r"model.layers.\d+.mlp.experts.w1", name) else False
            for name in state_dict
        )
        return any([cond1])

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_prime_to_hf(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_prime(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(
        cls, state_dict: dict[str, Tensor], layer_idx: int
    ) -> dict[str, Tensor]:
        from prime_rl.trainer.models.deepseek_v3.converting_deepseek_v3 import (
            convert_prime_layer_to_hf,
        )

        convert_prime_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(
        cls, state_dict: dict[str, Tensor], layer_idx: int
    ) -> dict[str, Tensor]:
        from prime_rl.trainer.models.deepseek_v3.converting_deepseek_v3 import (
            convert_hf_layer_to_prime,
        )

        convert_hf_layer_to_prime(state_dict, layer_idx)
        return state_dict


class DeepseekV3Model(DeepseekV3PreTrainedModel):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV3MLADecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )

        self.rotary_emb = RotaryEmbedding(
            config=RotaryEmbeddingConfig(
                max_position_embeddings=config.max_position_embeddings,
                rope_type=config.rope_type,
                model_config=config,
            ),
        )

        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config._attn_implementation in (
            "flash_attention_2",
            "flash_attention_3",
            "fa4",
        ):
            cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            max_seqlen = None
            cu_seqlens = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, GenerationMixin):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PrimeLmOutput:
        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ).unsqueeze(0)
            else:
                position_ids = torch.arange(
                    input_ids.shape[1], device=input_ids.device
                ).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        pass


__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3Model",
    "DeepseekV3PreTrainedModel",
]
