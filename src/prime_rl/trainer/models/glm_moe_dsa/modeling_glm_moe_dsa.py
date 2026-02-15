# coding=utf-8
# Copyright 2026 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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

import functools
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_to_hf_moe,
)
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import (
    RotaryEmbedding,
    RotaryEmbeddingConfig,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)

# flash-attention-2
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore

# flash-attention-3
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore

# tilelang sparse MLA kernels (vendored)
try:
    from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
    from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface
except ImportError:
    sparse_mla_fwd_interface = None  # type: ignore
    sparse_mla_bwd = None  # type: ignore


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class _SparseMLA(torch.autograd.Function):
    """Autograd wrapper for tilelang sparse MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, out, indices, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, out, indices, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(q, kv, out, do.contiguous(), indices, lse, sm_scale=ctx.sm_scale)
        return dq, dkv, None, None


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class Indexer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        if config.q_lora_rank is None:
            raise ValueError("Sparse indexer requires q_lora_rank to be set")

        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=config.attention_bias)
        self.k_norm = LayerNorm(dim=self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_head, bias=False)

    @torch.no_grad()
    def compute_sparse_indices(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        cu_seqlens: torch.Tensor,
        index_topk: int,
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[1]
        assert index_topk % 64 == 0, f"index_topk must be divisible by 64 (block_I), got {index_topk}"

        q_idx = self.wq_b(q_latent[0]).view(total_tokens, self.n_head, self.head_dim)
        k_idx = self.k_norm(self.wk(hidden_states[0]))
        w = self.weights_proj(hidden_states[0])

        q_combined = (q_idx * w.unsqueeze(-1)).sum(dim=1)

        num_seqs = cu_seqlens.shape[0] - 1
        indices = torch.zeros(total_tokens, index_topk, dtype=torch.int32, device=hidden_states.device)

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            scores = q_combined[start:end] @ k_idx[start:end].T
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal_mask, float("-inf"))

            actual_topk = min(index_topk, seq_len)
            _, topk_idx = torch.topk(scores, actual_topk, dim=-1)

            if actual_topk < index_topk:
                padding = topk_idx[:, :1].expand(-1, index_topk - actual_topk)
                topk_idx = torch.cat([topk_idx, padding], dim=-1)

            indices[start:end] = (topk_idx + start).to(torch.int32)

        return indices.view(1, total_tokens, 1, index_topk)


class _MLABase(nn.Module):
    """Base class for MLA (Multi-head Latent Attention) with shared projection logic."""

    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.rope_interleave = config.rope_interleave

        # Query projection (LoRA path)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # Key-Value compressed projection (MQA-style compression + rope key)
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

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.attention_bias)

        # Indexer: selects top-k tokens per query for sparse attention
        self.indexer = Indexer(config)

        # Attention scaling
        self.scaling = self.qk_head_dim ** (-0.5)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default") if isinstance(rope_parameters, dict) else "default"
        if rope_type != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into query, key, value with MLA decomposition.

        Returns (query_states, key_states, value_states) all in shape
        (batch, num_heads, seq_len, head_dim).
        """
        batch_size, seq_length = hidden_states.shape[:2]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        kv_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # Query path
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q_states = q_states.view(query_shape).transpose(1, 2)
        q_nope, q_rope = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV path: compress then up-project
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_upproj = self.kv_b_proj(self.kv_a_layernorm(k_compressed)).view(kv_shape).transpose(1, 2)
        k_nope, value_states = torch.split(k_upproj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # k_rope is shared across heads (MQA for the rope portion)
        k_rope = k_rope.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Apply RoPE
        cos, sin = position_embeddings
        if self.rope_interleave:
            q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos, sin)
        else:
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # Expand rope key to all heads
        k_rope = k_rope.expand(*k_nope.shape[:-1], -1)

        # Concatenate nope + rope parts
        query_states = torch.cat((q_nope, q_rope), dim=-1)
        key_states = torch.cat((k_nope, k_rope), dim=-1)

        return query_states, key_states, value_states

    @torch.no_grad()
    def _compute_sparse_indices(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        return self.indexer.compute_sparse_indices(hidden_states, q_latent, cu_seqlens, self.config.index_topk)


class GlmMoeDsaFlashAttention(_MLABase):
    """MLA attention using Flash Attention."""

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: GlmMoeDsaConfig, flash_attn_version: int = 2):
        super().__init__(config)
        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)

        # FA2 requires q_head_dim == v_head_dim; pad value if they differ
        needs_padding = self.qk_head_dim != self.v_head_dim
        if needs_padding:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # Flash attention expects (total_tokens, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        args = [
            query_states[0],
            key_states[0],
            value_states[0],
            cu_seqlens,
            cu_seqlens,
        ]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen])

        out = self._flash_attn_call(*args, causal=True)
        if isinstance(out, tuple):
            out = out[0]

        if needs_padding:
            out = out[..., : self.v_head_dim]

        out = out.contiguous()
        attn_output = out.view(1, out.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class GlmMoeDsaSDPAAttention(_MLABase):
    """MLA attention using PyTorch SDPA."""

    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)

        # SDPA needs matching last dims for Q @ K^T; pad value if they differ
        needs_padding = self.qk_head_dim != self.v_head_dim
        if needs_padding:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # SDPA with GQA: repeat KV heads to match query head count
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        out = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)

        if needs_padding:
            out = out[..., : self.v_head_dim]

        out = out.transpose(1, 2).contiguous()
        attn_output = out.view(out.shape[0], out.shape[1], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class GlmMoeDsaSparseAttention(_MLABase):
    """MLA with Dynamic Sparse Attention via tilelang kernels.

    Uses the absorption trick to operate in the compressed KV latent space:
      sparse_q  = cat(Q_nope @ W_kv_b_k_nope^T, Q_rope)   →  [B, S, H, kv_lora_rank + rope_dim]
      sparse_kv = cat(kv_a_layernorm(k_compressed), k_rope) →  [B, S, 1, kv_lora_rank + rope_dim]

    The tilelang sparse_mla_fwd/bwd kernels attend only to top-k tokens per query
    (selected by the bf16 indexer). Output is un-absorbed back to v_head_dim via
    the V portion of kv_b_proj weight.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, total_tokens, _ = hidden_states.shape

        # Q path
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_full = self.q_b_proj(q_latent).view(batch_size, total_tokens, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Compressed KV (no kv_b_proj up-projection)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed_normed = self.kv_a_layernorm(k_compressed)

        # RoPE — needs [batch, heads, seq, dim] layout
        q_rope_r = q_rope.transpose(1, 2)
        k_rope_r = k_rope.unsqueeze(1)
        cos, sin = position_embeddings
        if self.rope_interleave:
            q_rope_r, k_rope_r = apply_rotary_pos_emb_interleave(q_rope_r, k_rope_r, cos, sin)
        else:
            q_rope_r, k_rope_r = apply_rotary_pos_emb(q_rope_r, k_rope_r, cos, sin)
        q_rope = q_rope_r.transpose(1, 2)  # [B, total, H, rope_dim]
        k_rope = k_rope_r.squeeze(1)  # [B, total, rope_dim]

        # Indexer
        indices = self._compute_sparse_indices(hidden_states, q_latent, cu_seqlens)

        # Absorption: Q_nope @ W_kv_b_k_nope^T → [B, S, H, kv_lora_rank]
        kv_b_w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = kv_b_w[:, : self.qk_nope_head_dim, :]  # [H, nope, kv_lora_rank]
        w_v = kv_b_w[:, self.qk_nope_head_dim :, :]  # [H, v_head_dim, kv_lora_rank]

        q_absorbed = torch.einsum("bshd,hdk->bshk", q_nope, w_k_nope)

        # Build sparse tensors for tilelang
        sparse_q = torch.cat([q_absorbed, q_rope], dim=-1)  # [B, S, H, 576]
        sparse_kv = torch.cat([k_compressed_normed, k_rope], dim=-1).unsqueeze(2)  # [B, S, 1, 576]

        # Sparse attention via tilelang
        out = _SparseMLA.apply(sparse_q, sparse_kv, indices, self.scaling)  # [B, S, H, kv_lora_rank]

        # Un-absorb: out @ W_v^T → [B, S, H, v_head_dim]
        out = torch.einsum("bshk,hdk->bshd", out, w_v)

        out = out.reshape(batch_size, total_tokens, -1)
        return self.o_proj(out), None


_MLA_ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(GlmMoeDsaFlashAttention, flash_attn_version=2),
    "sdpa": GlmMoeDsaSDPAAttention,
    "flash_attention_3": functools.partial(GlmMoeDsaFlashAttention, flash_attn_version=3),
    "fa4": functools.partial(GlmMoeDsaFlashAttention, flash_attn_version=4),
    "sparse": GlmMoeDsaSparseAttention,
}


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _MLA_ATTN_IMPL2CLASS[config._attn_implementation](config)

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
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
        )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
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


@auto_docstring
class GlmMoeDsaPreTrainedModel(PreTrainedModelPrimeRL):
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.1.up_proj" in name or "mlp.experts.gate_up_proj" in name for name in state_dict.keys())

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict


@auto_docstring
class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        requested_attn_impl = config._attn_implementation
        if requested_attn_impl == "sparse":
            config._attn_implementation = "sdpa"

        super().__init__(config)

        if requested_attn_impl == "sparse":
            config._attn_implementation = requested_attn_impl

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default") if isinstance(rope_parameters, dict) else "default"
        rotary_config = RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
        self.rotary_emb = RotaryEmbedding(rotary_config)
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
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if self.config._attn_implementation in ("flash_attention_2", "flash_attention_3", "fa4", "sparse"):
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
class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        requested_attn_impl = config._attn_implementation
        if requested_attn_impl == "sparse":
            config._attn_implementation = "sdpa"

        super().__init__(config)

        self.model = GlmMoeDsaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if requested_attn_impl == "sparse":
            config._attn_implementation = requested_attn_impl

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        """
        assert use_cache is None, "use_cache is not supported for custom glm_moe_dsa for now"
        assert past_key_values is None, "past_key_values is not supported for custom glm_moe_dsa for now"

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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.inv_freq.copy_(inv_freq)


__all__ = ["GlmMoeDsaConfig", "GlmMoeDsaPreTrainedModel", "GlmMoeDsaModel", "GlmMoeDsaForCausalLM"]
