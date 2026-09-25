import functools
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from prime_rl.trainer.models.fusions import fuse_qkv_projections

from .norms import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb

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

# flash-attention-4
# flash_attn_4_varlen_func stays the raw kernel: gpt_oss and afmoe import it from here.
try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore
    _flash_attn_fwd = None  # type: ignore
    _flash_attn_bwd = None  # type: ignore


# FA4's flash_attn_varlen_func is a Python autograd.Function that Dynamo cannot trace. Inside a
# checkpointed block that graph break drops the whole block to eager, so FA4's kernels are exposed
# as opaque custom ops with FA2/FA3's varlen signature and dispatched through FlashAttention._funcs.
# That keeps _compute_attention version-free and lets selective AC save the op output.
@torch.library.custom_op("prime_rl_attn::flash_attn_4_varlen", mutates_args=())
def flash_attn_4_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        return_lse=True,
    )
    return out, lse


@flash_attn_4_varlen.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size_left, window_size_right):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl_attn::flash_attn_4_varlen_backward", mutates_args=())
def flash_attn_4_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return dq, dk, dv


@flash_attn_4_varlen_backward.register_fake
def _(
    dout,
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
    window_size_left,
    window_size_right,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _flash_attn_4_varlen_setup_context(ctx, inputs, output) -> None:
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size_left, window_size_right = (
        inputs
    )
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right


def _flash_attn_4_varlen_autograd(ctx, dout: torch.Tensor, _dlse: torch.Tensor | None):
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    dq, dk, dv = flash_attn_4_varlen_backward(
        dout.contiguous(),
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.causal,
        ctx.window_size_left,
        ctx.window_size_right,
    )
    return dq, dk, dv, None, None, None, None, None, None, None


flash_attn_4_varlen.register_autograd(_flash_attn_4_varlen_autograd, setup_context=_flash_attn_4_varlen_setup_context)


def _flash_attn_4_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """FA4 with the same call convention as FA2/FA3's flash_attn_varlen_func, for _funcs dispatch."""
    window_size_left, window_size_right = window_size
    out, _ = flash_attn_4_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        None if window_size_left == -1 else window_size_left,
        None if window_size_right == -1 else window_size_right,
    )
    return out


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


# TODO: Does torch compile support config._attn_implementation forking?
# If so, we can combine FlashAttention variants into one class
# Otherwise, do ABC or something to make the signatures match


class FlashAttention(nn.Module):
    """Flash Attention"""

    supported_fusions = {"qkv": fuse_qkv_projections}

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: _flash_attn_4_varlen_func if _flash_attn_fwd is not None else None,
    }

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.qkv_sizes = (
            config.num_attention_heads * self.head_dim,
            config.num_key_value_heads * self.head_dim,
            config.num_key_value_heads * self.head_dim,
        )
        self.q_proj = nn.Linear(config.hidden_size, self.qkv_sizes[0], bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.qkv_sizes[1], bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.qkv_sizes[2], bias=config.attention_bias)
        self.register_module("qkv_proj", None)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.output_bias)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                self.q_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
                )
                self.k_norm = RMSNorm(
                    RMSNormConfig(hidden_size=config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
                )
            else:
                self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
                self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query, key and value projections, from one packed GEMM when qkv is fused."""
        if self.qkv_proj is None:
            return self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        return self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)

    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, max_seqlen):
        """Run the flash attention kernel. q/k/v are [total_tokens, heads, dim]."""
        kwargs: dict = {"causal": True}
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            kwargs["window_size"] = (sliding_window - 1, 0)
        return self.func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.project_qkv(hidden_states)

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
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary application instead of double transpose?
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        out = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)
        attn_output = out.contiguous().view(1, out.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(FlashAttention, flash_attn_version=2),
    "flash_attention_3": functools.partial(FlashAttention, flash_attn_version=3),
    "flash_attention_4": functools.partial(FlashAttention, flash_attn_version=4),
}


def substitute_ring_attn(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
    attn_impl: str = "flash_attention_2",
) -> None:
    """Patch _compute_attention on FlashAttention variants to use ring attention."""
    from .ring_attn import ring_varlen_attention

    def _ring_compute_attention(self, q, k, v, cu_seqlens, max_seqlen):
        from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

        window_size = (-1, -1)
        sliding_window = getattr(self, "sliding_window", None)
        if sliding_window is not None:
            window_size = (sliding_window - 1, 0)

        out = ring_varlen_attention(
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
            attention_backend=attn_impl,
        )
        return out

    FlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.afmoe.modeling_afmoe import AfmoeFlashAttention

    AfmoeFlashAttention._compute_attention = _ring_compute_attention

    from prime_rl.trainer.models.gpt_oss.attention import substitute_gpt_oss_ring_attention

    substitute_gpt_oss_ring_attention(process_group, heads_k_stride)
