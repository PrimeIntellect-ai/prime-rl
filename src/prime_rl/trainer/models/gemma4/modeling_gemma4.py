# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
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
"""Prime-RL Gemma4 text-only modeling.

Started as a verbatim copy of ``transformers.models.gemma4.modeling_gemma4`` (text
portion only), then edited in-place for prime-rl conventions. The attention
module is ours — it dispatches to prime-rl's ``FlashAttention``/``SDPAAttention``
primitives instead of HF's ``ALL_ATTENTION_FUNCTIONS`` registry, so we own the
(sliding-window + packed-varlen) code path.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

# Flash-attn imports (same dispatch prime-rl uses elsewhere).
try:
    from flash_attn import flash_attn_varlen_func as flash_attn_2_varlen_func
except ImportError:
    flash_attn_2_varlen_func = None  # type: ignore

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gemma4.configuration_gemma4 import Gemma4Config
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Norm, rotary, MLP, experts, router — vendored verbatim from HF.
# ---------------------------------------------------------------------------


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma4TextRotaryEmbedding(nn.Module):
    """Per-layer-type RoPE.

    Gemma4 uses different RoPE parameters (and possibly different head_dim via
    ``global_head_dim``) for sliding vs full attention, so this module keeps a
    separate inv_freq per layer_type.
    """

    def __init__(self, config: Gemma4Config, device: Optional[torch.device] = None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[str, Callable] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            if rope_params is None:
                continue
            rope_type = rope_params["rope_type"]
            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self._default_rope_parameters
            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            curr_inv_freq, curr_attention_scaling = rope_init_fn(config, **kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def _default_rope_parameters(
        config: Gemma4Config | None = None,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma4TextMLP(nn.Module):
    """Dense MLP that runs in parallel to the MoE block at each decoder layer."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4TextExperts(nn.Module):
    """MoE experts with packed (gate+up) projection. Mirrors HF's layout."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Accumulate per-(token, top_k_slot) contributions in a 3D buffer, then
        # reduce over the top_k dim with torch.sum's fp32 accumulator. Matches
        # HF's grouped_mm / batched_mm forwards and avoids index_add_'s in-place
        # bf16 accumulation (which loses ~1e-3 relative precision over 8 terms).
        num_tokens, hidden_dim = hidden_states.shape
        num_top_k = top_k_index.shape[-1]
        weighted_out = torch.zeros(
            (num_tokens, num_top_k, hidden_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            # (token_idx, top_k_pos) pairs are disjoint across experts by construction.
            weighted_out[token_idx, top_k_pos, :] = current_hidden_states.to(weighted_out.dtype)
        return weighted_out.sum(dim=1).to(hidden_states.dtype)


class Gemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps
        self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        router_probabilities = F.softmax(expert_scores, dim=-1)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index


# ---------------------------------------------------------------------------
# Custom attention — the only block that deviates from HF.
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA. Input shape: [B, Hkv, S, D]."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv, n_rep, slen, head_dim)
        .reshape(batch, num_kv * n_rep, slen, head_dim)
    )


class Gemma4Attention(nn.Module):
    """Gemma4 attention with prime-rl-owned kernel dispatch.

    Differences from HF's ``Gemma4TextAttention``:
      - Dispatches to ``flash_attn_varlen_func`` (FA2/FA3) or ``F.scaled_dot_product_attention``
        directly, rather than HF's ``ALL_ATTENTION_FUNCTIONS`` registry.
      - Accepts ``cu_seqlens``/``max_seqlen`` for packed-sequence training (FA path).
      - Does not implement KV cache sharing / past_key_values (training only).

    Weight layout matches HF exactly so checkpoints load unchanged.
    """

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Gemma4 has per-layer-type head_dim: sliding uses ``head_dim``, full uses
        # ``global_head_dim`` (when set).
        self.head_dim = (
            config.global_head_dim
            if (not self.is_sliding and getattr(config, "global_head_dim", None))
            else config.head_dim
        )

        # "Alternative attention" layers (full_attention with attention_k_eq_v)
        # have no v_proj; V reuses K.
        self.use_alternative_attention = getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        num_kv_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_heads // num_kv_heads
        # Gemma4's effective scaling is absorbed into q_norm / k_norm, so the
        # attention kernel itself runs with scale=1.0.
        self.scaling = 1.0

        # Projections
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = (
            None
            if self.use_alternative_attention
            else nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

        # Per-head Q/K/V norms. V is norm-without-scale.
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        self.attention_dropout = config.attention_dropout
        self._attn_impl = config._attn_implementation

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape
        hidden_shape = (B, S, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        if self.v_proj is not None:
            v = self.v_proj(hidden_states).view(hidden_shape)
        else:
            v = self.k_proj(hidden_states).view(hidden_shape)

        # Per-head norms happen in full-res [B, S, H, D] layout.
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Rotary — HF applies with unsqueeze_dim=2 on [B, S, H, D] layouts.
        cos, sin = position_embeddings
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)

        return q, k, v  # each [B, S, H, D]

    def _flash_attn_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Varlen flash-attn call. q/k/v shape: [1, S, H, D]."""
        assert q.shape[0] == 1, "packed-varlen attention expects batch dim 1"
        fn = flash_attn_3_varlen_func if self._attn_impl == "flash_attention_3" else flash_attn_2_varlen_func
        if fn is None:
            raise RuntimeError(f"attn impl {self._attn_impl} requested but not installed")

        kwargs: dict = {"causal": True}
        if self.sliding_window is not None:
            kwargs["window_size"] = (self.sliding_window - 1, 0)
        kwargs["softmax_scale"] = self.scaling

        # Shape: [total, H, D]
        q_flat = q[0]
        k_flat = k[0]
        v_flat = v[0]
        out = fn(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.view(1, out.shape[0], -1)

    def _sdpa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """SDPA path. q/k/v shape: [B, S, H, D].

        Builds the correct attention mask for the current layer type:
          - causal (lower-triangular)
          - + sliding window (self.sliding_window) if this is a sliding layer
          - + block-diagonal segmentation if ``cu_seqlens`` is given (packed
            training), so tokens in segment B don't attend to segment A.
        """
        # SDPA wants [B, H, S, D].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        dropout_p = self.attention_dropout if self.training else 0.0
        S = q.size(-2)

        # Always build an explicit mask so SDPA always takes the same kernel
        # path HF does (is_causal=False + attn_mask=…). Using the is_causal=True
        # shortcut for the unpacked full-attention case would dispatch to a
        # different SDPA kernel and produce slightly different bf16 numerics.
        row = torch.arange(S, device=q.device)[:, None]
        col = torch.arange(S, device=q.device)[None, :]
        allow = row >= col  # causal
        if self.is_sliding:
            allow = allow & (row - col < self.sliding_window)
        if cu_seqlens is not None:
            # Assign a segment id per position, then allow only within-segment pairs.
            segment_id = torch.zeros(S, dtype=torch.long, device=q.device)
            bounds = cu_seqlens[1:-1].to(torch.long) if cu_seqlens.numel() > 2 else cu_seqlens.new_empty(0)
            for b in bounds.tolist():
                segment_id[b:] += 1
            allow = allow & (segment_id[:, None] == segment_id[None, :])

        attn_mask = torch.zeros((S, S), dtype=q.dtype, device=q.device).masked_fill(~allow, float("-inf"))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask[None, None, :, :],
            dropout_p=dropout_p,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> tuple[torch.Tensor, None]:
        q, k, v = self._project_qkv(hidden_states, position_embeddings)

        if self._attn_impl in ("flash_attention_2", "flash_attention_3") and cu_seqlens is not None:
            out = self._flash_attn_forward(q, k, v, cu_seqlens, max_seqlen)
        else:
            out = self._sdpa_forward(q, k, v, cu_seqlens=cu_seqlens)

        out = self.o_proj(out)
        return out, None


# ---------------------------------------------------------------------------
# Decoder layer, embedding, model, ForCausalLM.
# ---------------------------------------------------------------------------


class Gemma4TextDecoderLayer(GradientCheckpointingLayer):
    """Gemma4 decoder layer: MLP runs in parallel to MoE, optional per-layer input gate."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Gemma4Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """Input embedding scaled by ``embed_scale`` on lookup (Gemma4 convention)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


@auto_docstring
class Gemma4PreTrainedModel(PreTrainedModelPrimeRL):
    config: Gemma4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma4TextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("experts.gate_up_proj" in name for name in state_dict.keys())

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return cls.is_hf_state_dict(state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict


@auto_docstring
class Gemma4Model(Gemma4PreTrainedModel):
    """Gemma4 text model. Training-only: no KV cache, no past_key_values."""

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                padding_idx=None,
                embed_scale=self.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.register_buffer(
                "per_layer_model_projection_scale",
                torch.tensor(config.hidden_size**-0.5),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(self.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self.post_init()

    def get_per_layer_inputs(
        self,
        input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert self.hidden_size_per_layer_input, "per-layer inputs only defined when configured"
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
    ) -> torch.Tensor:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        # Varlen cu_seqlens for packed training when FA is the kernel.
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

        # Pre-compute RoPE tables per unique layer_type.
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(inputs_embeds, position_ids, layer_type)

        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.model = Gemma4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Union[torch.Tensor, None] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        if position_ids is None:
            ref = inputs_embeds if inputs_embeds is not None else input_ids
            position_ids = torch.arange(ref.shape[1], device=ref.device).unsqueeze(0)

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
        """Re-initialize non-persistent buffers that ``.to_empty()`` leaves as garbage.

        ``torch.nn.Module.to_empty()`` allocates empty storage for every buffer,
        including ones registered with ``persistent=False`` (which aren't in the
        state_dict and so don't get filled by ``load_state_dict``). We have to
        restore them by hand.
        """
        cfg = self.config

        # 1. Scaled-word-embedding ``embed_scale`` buffers (persistent=False).
        self.model.embed_tokens.embed_scale.copy_(torch.tensor(cfg.hidden_size**0.5))
        if getattr(self.model, "embed_tokens_per_layer", None) is not None:
            self.model.embed_tokens_per_layer.embed_scale.copy_(torch.tensor(cfg.hidden_size_per_layer_input**0.5))

        # 2. Per-layer projection scales used when hidden_size_per_layer_input is set.
        if self.model.hidden_size_per_layer_input:
            self.model.per_layer_model_projection_scale.copy_(torch.tensor(cfg.hidden_size**-0.5))
            self.model.per_layer_input_scale.copy_(torch.rsqrt(torch.tensor(2.0)))

        # 3. Per-decoder ``layer_scalar`` buffers (HF ships as ones).
        for layer in self.model.layers:
            layer.layer_scalar.copy_(torch.ones_like(layer.layer_scalar))

        # 4. RoPE inv_freq buffers. Re-register as fp32 regardless of current
        # buffer dtype: HF keeps inv_freq in fp32 even when params are bf16, so
        # any ``.to(bfloat16)`` on the whole model that downcast the buffer must
        # be reverted here. Storing in bf16 perturbs cos/sin enough to drift Q/K
        # after RoPE, which compounds layer-over-layer (layer 0 ~3e-3 rel).
        rope = self.model.rotary_emb
        for layer_type in rope.layer_types:
            if not hasattr(rope, f"{layer_type}_inv_freq"):
                continue
            init_fn = rope.rope_init_fns.get(layer_type)
            if init_fn is None:
                continue
            device = getattr(rope, f"{layer_type}_inv_freq").device
            kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope.rope_type[layer_type] == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, attention_scaling = init_fn(rope.config, **kwargs)
            rope.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            rope.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(rope, f"{layer_type}_attention_scaling", attention_scaling)


__all__ = [
    "Gemma4ForCausalLM",
    "Gemma4Model",
    "Gemma4PreTrainedModel",
]
