"""Inkling 2 (text tower) for PrimeRL training.

Reference implementations:
  * transformers 5.23 ``transformers/models/inkling2/modeling_inkling2.py``
  * transformers 5.24 ``transformers/models/inkling2_loop/modeling_inkling2_loop.py``
  * the vLLM port (vllm/models/inkling2)

Structure per layer: block attention residual -> attention (sliding window or
global with a hidden-conditioned relative bias, per-token sinks, output gate,
q/rel L2 clips, unweighted value norm, log scaling on global layers) -> short
convolution -> block attention residual -> MoE (384 experts, top-8 sigmoid
router with a selection bias, softcapped SwiGLU) -> short convolution.

Looped checkpoints (Inkling 2 Mini) run the loop core several times with shared
weights. Every pass restarts from the state that entered the core; after the
first pass, each core layer also reads the previous pass's blocks (the message)
and gates it into its attention and MoE inputs.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL, TextModelOutput, count_active_params
from prime_rl.trainer.models.inkling2.configuration_inkling2 import (
    Inkling2Config,
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
    Inkling2TextConfig,
)
from prime_rl.trainer.models.inkling2.converting_inkling2 import conversion_chain
from prime_rl.trainer.models.layers.activations import SoftcappedSwiglu
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

# Eager flex_attention materializes the full [heads, seq, seq] score matrix through the vmap
# reference path (48 x 8192 x 8192 fp32 = 24 GiB at 8K). The compiled kernel fuses score_mod
# into the block-sparse loop: measured peak 0.32 GiB at 8K and 0.56 GiB at 16K for a 48-head,
# 8-KV-head attention with the relative-bias score_mod. Shapes are static (padding to seq_len).
flex_attention = torch.compile(_flex_attention, dynamic=False)


class Inkling2UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class Inkling2L2Clip(nn.Module):
    """Scales each head vector down to ``threshold`` when its L2 norm is larger."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


class Inkling2ScaledL2Clip(nn.Module):
    """``Inkling2L2Clip`` applied after a learned per-(head, dim) scale."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * self.scale.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


def _doc_ids_from_cu_seqlens(cu_seqlens: Tensor, seq_len: int) -> Tensor:
    """Document index of every packed position, shape ``[1, seq_len]``."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=cu_seqlens.device, dtype=torch.int32), lengths
    ).unsqueeze(0)


def _shifted_within_documents(x: Tensor, doc_ids: Tensor, lag: int) -> Tensor:
    """``x[t - lag]`` where ``t - lag`` is in the same document, else zeros."""
    if lag == 0:
        return x
    shifted = torch.zeros_like(x)
    shifted[:, lag:] = x[:, :-lag]
    same_doc = torch.zeros_like(doc_ids, dtype=torch.bool)
    same_doc[:, lag:] = doc_ids[:, lag:] == doc_ids[:, :-lag]
    return shifted * same_doc.unsqueeze(-1)


class Inkling2ShortConvolution(nn.Module):
    """Static depthwise causal short convolution with a renormalized residual.

    Implemented as a lag-accumulation (one shift-and-add per tap) so it stays a
    tensor program under ``torch.compile`` and cannot leak across packed
    documents. Computed in fp32 like the reference.
    """

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(kernel_size + 1)

    def forward(self, hidden_states: Tensor, cu_seqlens: Tensor, doc_ids: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        weight = self.conv1d.weight.float().view(-1, self.kernel_size)
        out = torch.zeros_like(x)
        for lag in range(self.kernel_size):
            # PyTorch's conv1d is a cross-correlation: tap `j` weighs lag `K-1-j`.
            out = out + _shifted_within_documents(x, doc_ids, lag) * weight[:, self.kernel_size - 1 - lag]
        return ((out + x) * self.scale).to(dtype)


class Inkling2DynamicShortConvolution(nn.Module):
    """Short causal conv whose taps are static per-channel weights plus per-token
    weights projected from the same hidden states, shared within each KV head."""

    def __init__(self, hidden_size: int, input_size: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.static_weight = nn.Parameter(torch.zeros(kernel_size, hidden_size))
        self.weight_proj = nn.Linear(input_size, num_heads * kernel_size, bias=False)
        self.scale = 1.0 / math.sqrt(2 * kernel_size + 1)

    def forward(self, hidden_states: Tensor, dynamic_input: Tensor, cu_seqlens: Tensor, doc_ids: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        dynamic = (
            self.weight_proj(dynamic_input).float().view(x.shape[0], x.shape[1], self.num_heads, 1, self.kernel_size)
        )
        static = self.static_weight.float().T.view(self.num_heads, self.head_dim, self.kernel_size)
        taps = (static + dynamic).reshape(x.shape[0], x.shape[1], -1, self.kernel_size)  # [B, S, C, K]
        out = torch.zeros_like(x)
        for lag in range(self.kernel_size):
            out = out + _shifted_within_documents(x, doc_ids, lag) * taps[..., lag]
        return ((out + x) * self.scale).to(dtype)


class Inkling2Attention(nn.Module):
    """Sliding-window or global attention with relative bias, sinks and output gate."""

    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.scaling = self.head_dim**-0.5
        self.d_rel = config.d_rel

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.r_proj = nn.Linear(hidden, self.num_heads * self.d_rel, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.q_l2_clip = Inkling2ScaledL2Clip(self.num_heads, self.head_dim, config.qk_l2_clip_threshold)
        self.k_l2_clip = Inkling2L2Clip(config.qk_l2_clip_threshold)
        self.rel_l2_clip = Inkling2ScaledL2Clip(self.num_heads, config.d_rel, config.rel_l2_clip_threshold)
        self.rel_logits_proj = nn.Parameter(torch.empty(config.d_rel, self.rel_extent))
        self.v_norm = Inkling2UnweightedRMSNorm(eps=config.rms_norm_eps)
        if self.is_sliding:
            kv_size = self.num_kv_heads * self.head_dim
            self.k_sconv = Inkling2ShortConvolution(kv_size, config.sconv_kernel_size)
            self.v_sconv = Inkling2ShortConvolution(kv_size, config.sconv_kernel_size)
        else:
            kv_size = self.num_kv_heads * self.head_dim
            self.k_sconv = Inkling2DynamicShortConvolution(kv_size, hidden, self.num_kv_heads, config.sconv_kernel_size)
            self.v_sconv = Inkling2DynamicShortConvolution(kv_size, hidden, self.num_kv_heads, config.sconv_kernel_size)

        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha

    def forward(
        self,
        hidden_states: Tensor,
        key_hidden_states: Tensor,
        value_hidden_states: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        doc_ids: Tensor,
        block_mask: BlockMask,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        B, S, _ = hidden_states.shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        if self.is_sliding:
            key_states = self.k_sconv(self.k_proj(key_hidden_states), cu_seqlens, doc_ids)
            value_states = self.v_sconv(self.v_proj(value_hidden_states), cu_seqlens, doc_ids)
        else:
            key_states = self.k_sconv(self.k_proj(key_hidden_states), key_hidden_states, cu_seqlens, doc_ids)
            value_states = self.v_sconv(self.v_proj(value_hidden_states), value_hidden_states, cu_seqlens, doc_ids)
        relative_states = self.r_proj(hidden_states)
        gate = self.g_proj(hidden_states)
        sinks = self.sink_proj(hidden_states).transpose(1, 2)  # [B, H, S]

        query_states = self.q_l2_clip(query_states.view(hidden_shape))
        key_states = self.k_l2_clip(key_states.view(hidden_shape))
        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states.float()).to(value_states.dtype)
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))

        if not self.is_sliding:
            tau = 1.0 + self.log_scaling_alpha * torch.log(
                ((position_ids + 1).float() / self.log_scaling_n_floor).clamp(min=1.0)
            )
            tau = tau[..., None, None]
            query_states = (query_states.float() * tau).to(query_states.dtype)
            relative_states = (relative_states.float() * tau).to(relative_states.dtype)

        # Hidden-conditioned relative bias: one profile per (query, head, distance).
        rel_logits = torch.matmul(relative_states, self.rel_logits_proj)  # [B, S, H, rel_extent]

        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        out, aux = flex_attention(
            q,
            k,
            v,
            score_mod=_relative_bias_score_mod(rel_logits, self.rel_extent),
            block_mask=block_mask,
            scale=self.scaling,
            enable_gqa=self.num_heads != self.num_kv_heads,
            return_aux=AuxRequest(lse=True),
        )
        attn_output = (out * torch.sigmoid(aux.lse - sinks.float()).unsqueeze(-1)).to(out.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        return self.o_proj(attn_output)


def _relative_bias_score_mod(rel_logits: Tensor, rel_extent: int):
    def score_mod(score, b, h, q_idx, kv_idx):
        dist = q_idx - kv_idx
        return score + rel_logits[b, q_idx, h, dist.clamp(0, rel_extent - 1)] * (dist < rel_extent)

    return score_mod


class Inkling2BlockAttentionResidual(nn.Module):
    """Pseudo-query attention over the block states (the previous block outputs)
    plus the current partial state."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, query: Tensor, block_states: Tensor, partial: Tensor) -> Tensor:
        """``query``: ``[num_queries, H]``; returns ``[B, S, num_queries, H]``."""
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float()
        stacked = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + self.eps)
        q = query.float().reshape(-1, self.hidden_size).expand(*stacked.shape[:-2], -1, -1)
        scores = torch.matmul(q, stacked.transpose(-1, -2)) * self.hidden_size**-0.5
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, stacked)


class Inkling2AttnRes(nn.Module):
    """The per-layer queries of the block attention residuals."""

    def __init__(self, config: Inkling2TextConfig):
        super().__init__()
        local_ids = set(config.local_layer_ids)
        # The release stores these in FP32; parameters use the model dtype (FSDP requires
        # one storage dtype) and the conversion chain casts them back to FP32 for HF export.
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(config.hidden_size) if (i + 1) in local_ids else torch.zeros(3, config.hidden_size),
                requires_grad=True,
            )
            for i in range(config.num_hidden_layers - 1)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(config.hidden_size), requires_grad=True) for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=True)


class Inkling2LoopController(nn.Module):
    """Gates a message of the previous loop pass into one sublayer input, in fp32:
    ``(1 - tanh(z @ forget.T)) * x + tanh(z @ gain.T) * m`` with ``z = [rms(x), rms(m)] @ down.T``."""

    def __init__(self, hidden_size: int, rank: int, eps: float):
        super().__init__()
        self.norm = Inkling2UnweightedRMSNorm(eps)
        self.down_weight = nn.Parameter(torch.empty(rank, 2 * hidden_size))
        self.gain_weight = nn.Parameter(torch.empty(hidden_size, rank))
        self.forget_weight = nn.Parameter(torch.empty(hidden_size, rank))

    def forward(self, hidden_states: Tensor, message: Tensor) -> Tensor:
        hidden_states, message = hidden_states.float(), message.float()
        joined = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)
        # The down projection runs in the weight dtype like serving; the gates stay in fp32.
        z = F.linear(joined.to(self.down_weight.dtype), self.down_weight).float()
        forget = torch.tanh(F.linear(z, self.forget_weight.float()))
        gain = torch.tanh(F.linear(z, self.gain_weight.float()))
        return (1 - forget) * hidden_states + gain * message


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        hidden = config.hidden_size

        self.self_attn = Inkling2Attention(config, layer_idx)
        if self.is_sliding:
            self.input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
        else:
            self.q_input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
            self.k_input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
            self.v_input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
        self.post_attention_convolution = Inkling2ShortConvolution(hidden, config.sconv_kernel_size)
        self.post_mlp_convolution = Inkling2ShortConvolution(hidden, config.sconv_kernel_size)
        self.attention_input = Inkling2BlockAttentionResidual(hidden, config.rms_norm_eps) if layer_idx > 0 else None
        self.mlp_input = Inkling2BlockAttentionResidual(hidden, config.rms_norm_eps)
        if layer_idx in config.loop_core_layers:
            self.message_input = Inkling2BlockAttentionResidual(hidden, config.rms_norm_eps)
            self.controllers = nn.ModuleDict(
                {
                    name: Inkling2LoopController(hidden, config.loop_gate_rank, config.rms_norm_eps)
                    for name in ("attn", "k", "v", "mlp")
                }
            )

        if (config.mlp_gate_softcap, config.mlp_up_softcap) != (
            SoftcappedSwiglu.gate_softcap,
            SoftcappedSwiglu.up_softcap,
        ):
            raise ValueError(
                f"Inkling 2 expert softcaps {config.mlp_gate_softcap}/{config.mlp_up_softcap} are not supported; "
                f"the expert activation uses {SoftcappedSwiglu.gate_softcap}/{SoftcappedSwiglu.up_softcap}"
            )
        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            expert_type="gated",
            activation="softcapped_swiglu",
            score_func="sigmoid",
            route_norm=True,
            route_scale=config.route_scale,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=1e-3,
        )
        self.mlp = MoE.from_args(moe_args, dim=hidden, hidden_dim=config.moe_intermediate_size, shared_expert=None)

    def forward(
        self,
        hidden_states: Tensor,
        block_states: Tensor,
        attn_query: Tensor | None,
        mlp_query: Tensor,
        position_ids: Tensor,
        cu_seqlens: Tensor,
        doc_ids: Tensor,
        block_masks: dict[str, BlockMask],
        routed_experts: Tensor | None = None,
        bank: Tensor | None = None,
        message_queries: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """``bank`` (``[B, S, n, H]``) and ``message_queries`` (``[4, H]``) are given to a loop core
        layer after the first pass: the blocks the previous pass sealed inside the core followed by
        its last partial block, and this layer's Q, K, V and MoE message queries."""
        dtype = hidden_states.dtype
        block_residual = hidden_states
        messages = None
        if bank is not None:
            messages = self.message_input(message_queries, bank[..., :-1, :], bank[..., -1, :]).unbind(-2)

        if self.attention_input is not None:
            q_hidden = self.attention_input(attn_query, block_states, hidden_states)
            if self.is_sliding:
                hidden_states = q_hidden[..., 0, :]
            else:
                q_hidden, k_hidden, v_hidden = q_hidden.unbind(-2)

        if self.is_sliding and messages is None:
            attn_input = self.input_layernorm(hidden_states).to(dtype)
            attn_out = self.self_attn(
                attn_input,
                attn_input,
                attn_input,
                position_ids,
                cu_seqlens,
                doc_ids,
                block_masks["sliding"],
            )
        else:
            if self.is_sliding:
                # A sliding layer has one input and one norm; its Q, K and V inputs differ only by their messages.
                q_hidden = k_hidden = v_hidden = hidden_states
                q_norm = k_norm = v_norm = self.input_layernorm
            else:
                q_norm, k_norm, v_norm = self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm
            if messages is not None:
                q_hidden = self.controllers["attn"](q_hidden, messages[0])
                k_hidden = self.controllers["k"](k_hidden, messages[1])
                v_hidden = self.controllers["v"](v_hidden, messages[2])
            attn_out = self.self_attn(
                q_norm(q_hidden).to(dtype),
                k_norm(k_hidden).to(dtype),
                v_norm(v_hidden).to(dtype),
                position_ids,
                cu_seqlens,
                doc_ids,
                block_masks["sliding" if self.is_sliding else "full"],
            )
        attn_out = self.post_attention_convolution(attn_out, cu_seqlens, doc_ids)

        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
            residual = attn_out
        else:
            residual = (block_residual.float() + attn_out.float()).to(dtype)

        mlp_in = self.mlp_input(mlp_query, block_states, residual)[..., 0, :]
        if messages is not None:
            mlp_in = self.controllers["mlp"](mlp_in, messages[3])
        mlp_out = self.mlp(self.post_attention_layernorm(mlp_in).to(dtype), routed_experts=routed_experts)
        mlp_out = self.post_mlp_convolution(mlp_out, cu_seqlens, doc_ids)
        return ((residual.float() + mlp_out.float()).to(dtype), block_states)


def _build_block_masks(
    doc_ids: Tensor,
    sliding_window: int | None,
) -> dict[str, BlockMask]:
    B, S = doc_ids.shape

    def make(window: int | None):
        def mask_mod(b, h, q_idx, kv_idx):
            same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
            causal = q_idx >= kv_idx
            if window is None:
                return same_doc & causal
            return same_doc & causal & ((q_idx - kv_idx) < window)

        # The mask is head-independent (the head-dependent relative bias lives in
        # score_mod), so build it once for a single head: the dense [B, S, S]
        # intermediate and its int64 copy stay O(S^2) instead of O(H * S^2), and
        # flex_attention broadcasts the BlockMask over the query heads.
        return create_block_mask(mask_mod, B, None, S, S, device=doc_ids.device)

    return {"sliding": make(sliding_window), "full": make(None)}


class Inkling2PreTrainedModel(PreTrainedModelPrimeRL):
    config: Inkling2Config

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, MoE):
            buffer_device = module.tokens_per_expert.device
            if buffer_device.type == "meta":
                buffer_device = torch.device("cpu")
            module.init_weights(std, buffer_device=buffer_device)
        elif isinstance(module, Inkling2AttnRes):
            for param in module.parameters():
                nn.init.zeros_(param)
        elif isinstance(module, Inkling2LoopModel):
            nn.init.zeros_(module.middle_cycle_cross_pass_queries)
            nn.init.zeros_(module.middle_cycle_cross_pass_kv_queries)
        elif isinstance(module, Inkling2LoopController):
            for param in module.parameters():
                nn.init.trunc_normal_(param, mean=0.0, std=std)
        elif isinstance(module, Inkling2ScaledL2Clip):
            nn.init.ones_(module.scale)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Inkling2DynamicShortConvolution):
            nn.init.zeros_(module.static_weight)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, Inkling2Attention):
            nn.init.normal_(module.rel_logits_proj, mean=0.0, std=std)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(name.startswith("model.llm.") for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return "attn_res." in name or "middle_cycle_cross_pass_" in name or name.endswith("mlp.router.selection_bias")

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        return "Inkling 2 short convolutions need a halo exchange across context-parallel shards; CP > 1 is not supported yet"


class Inkling2Model(Inkling2PreTrainedModel):
    def __init__(self, config: Inkling2TextConfig):
        super().__init__(config)
        self.build(config)
        self.post_init()

    def build(self, config: Inkling2TextConfig) -> None:
        """Create the modules and parameters; `post_init` initializes them right after."""
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = RMSNorm(RMSNormConfig(config.hidden_size, config.rms_norm_eps))
        self.layers = nn.ModuleList([Inkling2DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(config.hidden_size, config.rms_norm_eps))
        self.attn_res = Inkling2AttnRes(config)
        self.final_input = Inkling2BlockAttentionResidual(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        routed_experts: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> TextModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.embed_norm(inputs_embeds)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        doc_ids = _doc_ids_from_cu_seqlens(cu_seqlens, inputs_embeds.shape[1])

        block_masks = _build_block_masks(doc_ids, self.config.sliding_window_size)

        hidden_states = inputs_embeds
        B, S, _ = hidden_states.shape
        block_states = hidden_states.new_zeros(B, S, 0, self.config.hidden_size)

        if routed_experts is not None and routed_experts.shape[2] != self.config.num_layer_visits:
            raise ValueError(
                f"routed_experts has {routed_experts.shape[2]} layer slots, but the model routes "
                f"{self.config.num_layer_visits} times per token (once per layer visit)"
            )
        hidden_states, block_states = self._run_layers(
            hidden_states, block_states, routed_experts, (position_ids, cu_seqlens, doc_ids, block_masks)
        )

        hidden_states = self.final_input(self.attn_res.output_query, block_states, hidden_states)[..., 0, :]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return TextModelOutput(last_hidden_state=hidden_states)

    def _run_layer(
        self,
        layer_idx: int,
        visit: int,
        hidden_states: Tensor,
        block_states: Tensor,
        routed_experts: Tensor | None,
        layer_args: tuple,
        bank: Tensor | None = None,
        message_queries: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Runs one layer. ``visit`` counts the layer calls of a forward and indexes ``routed_experts``."""
        # Every call passes the same positional arguments, so a compiled layer is not guarded on keyword names.
        return self.layers[layer_idx](
            hidden_states,
            block_states,
            self.attn_res.attn_queries[layer_idx - 1] if layer_idx > 0 else None,
            self.attn_res.mlp_queries[layer_idx],
            *layer_args,
            # A compiled layer takes the routes of its visit as a tensor of their own, not a strided view.
            routed_experts[:, :, visit].contiguous() if routed_experts is not None else None,
            bank,
            message_queries,
        )

    def _run_layers(
        self, hidden_states: Tensor, block_states: Tensor, routed_experts: Tensor | None, layer_args: tuple
    ) -> tuple[Tensor, Tensor]:
        for layer_idx in range(len(self.layers)):
            hidden_states, block_states = self._run_layer(
                layer_idx, layer_idx, hidden_states, block_states, routed_experts, layer_args
            )
        return hidden_states, block_states


class Inkling2LoopModel(Inkling2Model):
    def build(self, config: Inkling2LoopTextConfig) -> None:
        # The trainer can truncate the layers after the config is built (`debug.num_layers`).
        config.validate_loop()
        super().build(config)
        # Queries of the loop messages, named as in the checkpoint: non-layer weights are broadcast to
        # inference under the trainer's names. Rows 2j and 2j+1 belong to core layer j, as its Q and MoE
        # queries in the first table and its K and V queries in the second. Like the other block-residual
        # queries, the release stores them in FP32 and training keeps them in the model dtype.
        num_queries = 2 * len(config.loop_core_layers)
        self.middle_cycle_cross_pass_queries = nn.Parameter(torch.zeros(num_queries, config.hidden_size))
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(torch.zeros(num_queries, config.hidden_size))

    def _run_layers(
        self, hidden_states: Tensor, block_states: Tensor, routed_experts: Tensor | None, layer_args: tuple
    ) -> tuple[Tensor, Tensor]:
        core, num_passes = self.config.loop_core_layers, self.config.loop_num_passes
        for layer_idx in range(core.start):
            hidden_states, block_states = self._run_layer(
                layer_idx, layer_idx, hidden_states, block_states, routed_experts, layer_args
            )

        restart_hidden_states, restart_block_states = hidden_states, block_states
        num_restart_blocks = restart_block_states.shape[-2]
        queries, kv_queries = self.middle_cycle_cross_pass_queries, self.middle_cycle_cross_pass_kv_queries
        # Q, K, V and MoE queries of each core layer, stacked into new tensors: a compiled layer must
        # not receive views of the two tables.
        message_queries = [
            torch.stack((queries[2 * j], kv_queries[2 * j], kv_queries[2 * j + 1], queries[2 * j + 1]))
            for j in range(len(core))
        ]
        bank = None
        for pass_idx in range(num_passes):
            if pass_idx > 0:
                # The first core layer seals the restart partial as a block; the blocks after it and
                # the last partial are what this pass produced inside the core.
                bank = torch.cat([block_states[..., num_restart_blocks + 1 :, :], hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states, block_states = restart_hidden_states, restart_block_states
            for j, layer_idx in enumerate(core):
                hidden_states, block_states = self._run_layer(
                    layer_idx,
                    self.config.layer_visit_ids(layer_idx)[pass_idx],
                    hidden_states,
                    block_states,
                    routed_experts,
                    layer_args,
                    bank,
                    message_queries[j] if bank is not None else None,
                )

        for layer_idx in range(core.stop, len(self.layers)):
            (visit,) = self.config.layer_visit_ids(layer_idx)
            hidden_states, block_states = self._run_layer(
                layer_idx, visit, hidden_states, block_states, routed_experts, layer_args
            )
        return hidden_states, block_states


class SoftcapLinear(nn.Linear):
    """``nn.Linear`` that carries the final-logit softcap into the injected LM head."""

    def __init__(self, in_features: int, out_features: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap


class Inkling2ForCausalLM(Inkling2PreTrainedModel):
    text_config_class = Inkling2TextConfig
    text_model_class = Inkling2Model

    def __init__(self, config: Inkling2Config | Inkling2TextConfig):
        text_config = getattr(config, "text_config", None)
        if isinstance(text_config, dict):
            text_config = self.text_config_class(**text_config)
        if text_config is None:
            text_config = config
        super().__init__(text_config)
        self.model = self.text_model_class(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = SoftcapLinear(
            text_config.hidden_size, text_config.vocab_size, text_config.final_logit_softcapping
        )
        self.lm_head.unpadded_vocab_size = text_config.unpadded_vocab_size
        self.router_aux_loss_coef = 0.0
        self.num_experts = text_config.n_routed_experts
        self.num_experts_per_tok = text_config.num_experts_per_tok
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        logits_to_keep: Union[int, Tensor] = 0,
        temperature: Tensor | None = None,
        routed_experts: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
        )

    def init_buffers_post_meta(self) -> None:
        for module in self.modules():
            if isinstance(module, MoE):
                if module.tokens_per_expert.device.type != "meta":
                    module.tokens_per_expert.zero_()
                    module.routing_confidence_sum.zero_()
                    if module.router.selection_bias is not None:
                        module.router.selection_bias.zero_()


class Inkling2LoopForCausalLM(Inkling2ForCausalLM):
    config: Inkling2LoopConfig
    text_config_class = Inkling2LoopTextConfig
    text_model_class = Inkling2LoopModel

    def num_active_params(self) -> int:
        """A token goes through the loop core once per pass, and through its controllers once per pass after the first."""
        extra_passes = self.config.loop_num_passes - 1
        core_params, controller_params = [], []
        for layer_idx in self.config.loop_core_layers:
            for name, param in self.model.layers[layer_idx].named_parameters():
                (controller_params if ".controllers." in f".{name}" else core_params).append(param)
        return (
            count_active_params(self)
            + extra_passes * count_active_params(self, core_params)
            + (extra_passes - 1) * count_active_params(self, controller_params)
        )


__all__ = [
    "Inkling2Config",
    "Inkling2ForCausalLM",
    "Inkling2LoopConfig",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopModel",
    "Inkling2LoopTextConfig",
    "Inkling2Model",
    "Inkling2PreTrainedModel",
    "Inkling2TextConfig",
]
