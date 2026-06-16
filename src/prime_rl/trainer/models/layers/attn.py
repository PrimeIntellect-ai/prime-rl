import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .norms import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb

# flash-attention-2
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn import flash_attn_func as fa2_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore
    fa2_func = None  # type: ignore

# flash-attention-3
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
    from flash_attn_interface import flash_attn_func as fa3_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore
    fa3_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
    from flash_attn.cute import flash_attn_func as fa4_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore
    fa4_func = None


@dataclass
class AttentionConfig:
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    is_causal: bool
    attention_bias: bool
    use_qk_norm: bool
    rms_norm_eps: float
    qk_norm_type: Literal["per_head", "per_layer"] = "per_head"
    output_bias: bool = False
    scaling: float | None = None

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5


# TODO: Does torch compile support config._attn_implementation forking?
# If so, we can combine FlashAttention and SDPAAttention into one class
# Otherwise, do ABC or something to make the signatures match


class FlashAttentionCore(nn.Module):
    """Plain Flash Attention."""

    _funcs_varlen = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    _funcs = {
        2: fa2_func,
        3: fa3_func,
        4: fa4_func,
    }

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__()
        self.scaling = config.scaling
        self.is_causal = config.is_causal

        self._flash_attn_version = flash_attn_version
        self.att_core_func = self._funcs[flash_attn_version]
        self.func = self._funcs_varlen[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def _fa_installed(self):
        """Checks that flash attention is installed."""
        return self._funcs_varlen[self._flash_attn_version] is not None

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
    ):
        """Run the varlen flash attention kernel. q/k/v are [total_tokens, heads, dim].

        When running Ring attention or Ulysses this method will be patched.
        """

        kwargs: dict = {"causal": True, "softmax_scale": softmax_scale}
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            kwargs["window_size"] = (sliding_window - 1, 0)
        if self._flash_attn_version == 4:
            # FA4's flash_attn_varlen_func has qv as the 4th positional arg,
            # so cu_seqlens must be passed as keyword args to avoid misalignment.
            kwargs["cu_seqlens_q"] = cu_seqlens
            kwargs["cu_seqlens_k"] = cu_seqlens
            out = self._flash_attn_call(q, k, v, **kwargs)
        else:
            out = self._flash_attn_call(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs
            )
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: int | None = None,
        max_seqlen: int | None = None,
        softmax_scale: int | None = None,
    ) -> torch.Tensor:
        # q,k,v - [bs, sl, nh, hdim]

        if softmax_scale is None:
            softmax_scale = self.scaling

        bs = query_states.shape[0]
        if cu_seqlens is None:
            # Non-varlen: q/k/v are [bs, sl, nh, hdim]; FA returns same shape.
            out = self.att_core_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                softmax_scale=softmax_scale,
            )
            if isinstance(out, tuple):
                # 'fa4' returns tuple
                out = out[0]
            return out.view(out.shape[0], out.shape[1], -1)
        elif bs == 1:
            # Varlen: FA expects [total_tokens, nh, hdim] and returns the same.
            out = self._compute_attention(
                query_states[0],
                key_states[0],
                value_states[0],
                cu_seqlens,
                max_seqlen,
                softmax_scale=softmax_scale,
            )
            if isinstance(out, tuple):
                # 'fa4' returns tuple
                out = out[0]
            # Reshape back to [1, total_tokens, hidden] so o_proj sees a 3D tensor.
            return out.contiguous().view(1, out.shape[0], -1)

        raise NotImplementedError("varlen attention with bs > 1 is not supported")


class SDPAAttentionCore(nn.Module):
    """Plain SDPA Attention."""

    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.scaling
        self.is_causal = config.is_causal

    def _attention_core(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        softmax_scale: int | None = None,
    ) -> torch.Tensor:

        if softmax_scale is None:
            softmax_scale = self.scaling

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=softmax_scale,
            is_causal=self.is_causal,
        )
        out = out.transpose(1, 2).contiguous()
        return out.view(out.shape[0], out.shape[1], -1)


class FlashAttention(FlashAttentionCore):
    """Flash Attention with Q,K,V"""

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__(config, flash_attn_version)
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.output_bias,
        )
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                self.q_norm = RMSNorm(
                    RMSNormConfig(
                        hidden_size=config.num_attention_heads * self.head_dim,
                        eps=config.rms_norm_eps,
                    )
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(
                        hidden_size=config.num_key_value_heads * self.head_dim,
                        eps=config.rms_norm_eps,
                    )
                )
            else:
                self.q_norm = RMSNorm(
                    RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps)
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps)
                )

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm and self.qk_norm_type == "per_layer":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.use_qk_norm and self.qk_norm_type == "per_head":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # TODO: Can we optimize the rotary application instead of double transpose?
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        return query_states, key_states, value_states

    def output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(
            hidden_states, position_embeddings
        )

        attn_output = self._attention_core(
            query_states,
            key_states,
            value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_output = self.output_proj(attn_output)
        return attn_output, None


class SDPAAttention(SDPAAttentionCore):
    """SDPA Attention"""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.output_bias,
        )
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                self.q_norm = RMSNorm(
                    RMSNormConfig(
                        hidden_size=config.num_attention_heads * self.head_dim,
                        eps=config.rms_norm_eps,
                    )
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(
                        hidden_size=config.num_key_value_heads * self.head_dim,
                        eps=config.rms_norm_eps,
                    )
                )
            else:
                self.q_norm = RMSNorm(
                    RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps)
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps)
                )

    def attn_projections(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm and self.qk_norm_type == "per_layer":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.use_qk_norm and self.qk_norm_type == "per_head":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        return query_states, key_states, value_states

    def output_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self.attn_projections(
            hidden_states, position_embeddings
        )

        attn_output = self._attention_core(query_states, key_states, value_states)
        attn_output = self.output_proj(attn_output)
        return attn_output, None


ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(FlashAttention, flash_attn_version=2),
    "sdpa": SDPAAttention,
    "flash_attention_3": functools.partial(FlashAttention, flash_attn_version=3),
    "fa4": functools.partial(FlashAttention, flash_attn_version=4),
}


def substitute_ring_attn(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
    attn_impl: str = "flash_attention_2",
) -> None:
    """Patch _compute_attention on FlashAttention variants to use ring attention."""
    from ring_flash_attn import llama3_flash_attn_varlen_func

    from .ring_attn import ring_fa3_varlen_func, ring_fa4_varlen_func

    if attn_impl == "fa4":
        ring_func = ring_fa4_varlen_func
    elif attn_impl == "flash_attention_3":
        ring_func = ring_fa3_varlen_func
    else:
        ring_func = llama3_flash_attn_varlen_func

    def _ring_compute_attention(
        self, q, k, v, cu_seqlens, max_seqlen, softmax_scale=None
    ):
        from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

        window_size = (-1, -1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            window_size = (sliding_window - 1, 0)

        out = ring_func(
            q,
            k,
            v,
            cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
            cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
            max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
            max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
            local_k_slice=DATA_PARAMS["local_k_slice"],
            causal=True,
            window_size=window_size,
            group=process_group,
            heads_k_stride=heads_k_stride,
            softmax_scale=softmax_scale,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    FlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    AfmoeFlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedFlashAttention,
    )

    Qwen3_5MoeGatedFlashAttention._compute_attention = _ring_compute_attention
