# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from prime_rl.trainer.models.glm4_moe_lite.converting_glm4_moe_lite import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    """Apply interleaved rotary position embeddings (DeepSeek-V3 style)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply standard rotary position embeddings."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Glm4MoeLiteMLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with LoRA-style KV compression.

    This is the attention mechanism used by GLM-4.7-Flash and DeepSeek-V3.
    """

    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int, flash_attn_version: int = 2):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.rope_interleave = config.rope_interleave

        self.scaling = self.qk_head_dim ** (-0.5)

        # Q projection with LoRA decomposition
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV projection with LoRA decomposition
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=self.kv_lora_rank, eps=config.rms_norm_eps))
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.flash_attn_version = flash_attn_version
        if flash_attn_version == 3:
            self.func = flash_attn_3_varlen_func
        else:
            self.func = flash_attn_varlen_func

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Q projection
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q_states = q_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass))
        k_pass = k_pass.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(
            1, 2
        )
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Apply RoPE
        cos, sin = position_embeddings
        if self.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # Transpose for flash attention: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Pad value states if using flash attention and dimensions don't match
        need_padding = self.qk_head_dim != self.v_head_dim
        if need_padding:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        out = self.func(
            query_states[0],
            key_states[0],
            value_states[0],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )
        if isinstance(out, tuple):
            out = out[0]

        if need_padding:
            out = out[..., : self.v_head_dim]

        out = out.contiguous()
        attn_output = out.view(1, out.shape[0], -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Glm4MoeLiteDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Determine flash attention version
        flash_attn_version = 2
        if config._attn_implementation == "flash_attention_3":
            flash_attn_version = 3

        self.self_attn = Glm4MoeLiteMLAAttention(config, layer_idx, flash_attn_version)

        # MoE or dense MLP
        if config.mlp_layer_types[layer_idx] == "sparse":
            moe_args = MoEArgs(
                num_experts=config.n_routed_experts,
                num_shared_experts=config.n_shared_experts,
                score_func="sigmoid",
                route_norm=config.norm_topk_prob,
                route_scale=config.routed_scaling_factor,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                load_balance_coeff=1e-3,
                use_grouped_mm=config.use_grouped_mm,
            )
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
            mlp_config = MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                gate_act=config.hidden_act,
                bias=False,
            )
            self.mlp = MLP(mlp_config)

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring
class Glm4MoeLitePreTrainedModel(PreTrainedModelPrimeRL):
    config: Glm4MoeLiteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Glm4MoeLiteDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Glm4MoeLiteDecoderLayer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict contains MoE layers in HuggingFace format."""
        return any("mlp.experts.1.up_proj" in module_name for module_name in state_dict.keys())

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """Check if the state dict contains MoE layers in PrimeRL training format."""
        return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert MoE weights from PrimeRL training format to HuggingFace format in-place."""
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert MoE weights from HuggingFace format to PrimeRL training format in-place."""
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's MoE weights from PrimeRL format to HuggingFace format in-place."""
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """Convert a single layer's MoE weights from HuggingFace format to PrimeRL format in-place."""
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict


@auto_docstring
class Glm4MoeLiteModel(Glm4MoeLitePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.47.*"]

    def __init__(self, config: Glm4MoeLiteConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm4MoeLiteDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"
        rotary_config = RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
        self.rotary_emb = RotaryEmbedding(rotary_config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3"):
            flat_position_ids = position_ids.view(-1)
            seqlens = torch.cat(
                [
                    flat_position_ids[0:1],
                    flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
                    flat_position_ids[-1:] + 1,
                ]
            )
            max_seqlen = seqlens.max().item()
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            max_seqlen = None
            cu_seqlens = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class Glm4MoeLiteForCausalLM(Glm4MoeLitePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm4MoeLiteModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
            If not provided, the wrapped LM head returns logits only.
        temperature (`torch.Tensor`, *optional*):
            Temperature used for the logprobs/entropy computation when `labels` are provided.
        """
        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        # HF standard transformer model
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["Glm4MoeLiteConfig", "Glm4MoeLitePreTrainedModel", "Glm4MoeLiteModel", "Glm4MoeLiteForCausalLM"]
