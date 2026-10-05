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
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward, _flash_attn_varlen_forward
except ImportError:
    flash_attn_varlen_func = None  # type: ignore
    _flash_attn_varlen_backward = _flash_attn_varlen_forward = None  # type: ignore

# flash-attention-3
try:
    from flash_attn_interface import _flash_attn_backward as _flash_attn_3_bwd
    from flash_attn_interface import _flash_attn_forward as _flash_attn_3_fwd
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore
    _flash_attn_3_bwd = _flash_attn_3_fwd = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore
    _flash_attn_bwd = _flash_attn_fwd = None  # type: ignore


# Each FlashAttention version's flash_attn_varlen_func is a Python autograd.Function that Dynamo
# cannot trace (FA2 and FA3 also reject the same cu_seqlens tensor passed as both cu_seqlens_q and
# cu_seqlens_k). Inside a checkpointed block that graph break drops the whole block to eager, so
# each version is exposed as an opaque custom op instead, which also lets selective activation
# checkpointing save its output.
@torch.library.custom_op("prime_rl_attn::flash_attn_4_varlen", mutates_args=())
def _flash_attn_4_varlen(
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


@_flash_attn_4_varlen.register_fake
def _(q, k, v, *args):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl_attn::flash_attn_4_varlen_backward", mutates_args=())
def _flash_attn_4_varlen_backward(
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


@_flash_attn_4_varlen_backward.register_fake
def _(dout, q, k, v, *args):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _flash_attn_4_varlen_setup_context(ctx, inputs, output) -> None:
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size_left, window_size_right = (
        inputs
    )
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen = (max_seqlen_q, max_seqlen_k)
    ctx.causal = causal
    ctx.window_size = (window_size_left, window_size_right)


def _flash_attn_4_varlen_autograd(ctx, dout: torch.Tensor, _dlse: torch.Tensor | None):
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    dq, dk, dv = _flash_attn_4_varlen_backward(
        dout.contiguous(), q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, *ctx.max_seqlen, ctx.causal, *ctx.window_size
    )
    return dq, dk, dv, None, None, None, None, None, None, None


_flash_attn_4_varlen.register_autograd(_flash_attn_4_varlen_autograd, setup_context=_flash_attn_4_varlen_setup_context)


def flash_attn_4_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for FA4's ``flash_attn_varlen_func`` as called by ``FlashAttention``, routed through the custom op."""
    return _flash_attn_4_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, window_size[0], window_size[1]
    )


@torch.library.custom_op("prime_rl_attn::flash_attn_3_varlen", mutates_args=())
def _flash_attn_3_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _flash_attn_3_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return out, lse


@_flash_attn_3_varlen.register_fake
def _(q, k, v, *args):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl_attn::flash_attn_3_varlen_backward", mutates_args=())
def _flash_attn_3_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_3_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return dq, dk, dv


@_flash_attn_3_varlen_backward.register_fake
def _(dout, q, k, v, *args):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# FA2's public flash_attn_varlen_func pads the head dim to a multiple of 8 around its kernels; these ops call
# the kernels directly, so they pad the same way.
def pad_head_dim_to_multiple_of_8_for_fa2(t: torch.Tensor) -> torch.Tensor:
    head_dim = t.shape[-1]
    if head_dim % 8 == 0:
        return t
    return torch.nn.functional.pad(t, [0, 8 - head_dim % 8])


@torch.library.custom_op("prime_rl_attn::flash_attn_2_varlen", mutates_args=())
def _flash_attn_2_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    q, k, v = (pad_head_dim_to_multiple_of_8_for_fa2(t) for t in (q, k, v))
    out_padded, lse, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return out_padded[..., :head_dim], lse


# FA2's kernel allocates its outputs with empty_like on the padded inputs, so the fakes keep their strides and padding.
@_flash_attn_2_varlen.register_fake
def _(q, k, v, *args):
    head_dim = q.shape[-1]
    out = torch.empty_like(pad_head_dim_to_multiple_of_8_for_fa2(q))[..., :head_dim]
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl_attn::flash_attn_2_varlen_backward", mutates_args=())
def _flash_attn_2_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    dout, q, k, v, out = (pad_head_dim_to_multiple_of_8_for_fa2(t) for t in (dout, q, k, v, out))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        0.0,
        None,
        False,
    )
    return dq[..., :head_dim], dk[..., :head_dim], dv[..., :head_dim]


@_flash_attn_2_varlen_backward.register_fake
def _(dout, q, k, v, *args):
    head_dim = q.shape[-1]
    return tuple(torch.empty_like(pad_head_dim_to_multiple_of_8_for_fa2(t))[..., :head_dim] for t in (q, k, v))


def _flash_attn_varlen_setup_context(ctx, inputs, output) -> None:
    q, k, v, cu_seqlens_q, cu_seqlens_k, *scalar_args = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.scalar_args = scalar_args


def _flash_attn_varlen_autograd(backward_op, ctx, dout: torch.Tensor, _dlse: torch.Tensor | None):
    dq, dk, dv = backward_op(dout.contiguous(), *ctx.saved_tensors, *ctx.scalar_args)
    return dq, dk, dv, None, None, None, None, None, None, None, None


_flash_attn_2_varlen.register_autograd(
    functools.partial(_flash_attn_varlen_autograd, _flash_attn_2_varlen_backward),
    setup_context=_flash_attn_varlen_setup_context,
)
_flash_attn_3_varlen.register_autograd(
    functools.partial(_flash_attn_varlen_autograd, _flash_attn_3_varlen_backward),
    setup_context=_flash_attn_varlen_setup_context,
)


def flash_attn_2_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Drop-in for FA2's ``flash_attn_varlen_func``, routed through the custom op."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, _ = _flash_attn_2_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, *window_size
    )
    return out


def flash_attn_3_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """Drop-in for FA3's ``flash_attn_varlen_func``, routed through the custom op."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, _ = _flash_attn_3_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, *window_size
    )
    return out


def quadratic_attention_flops_per_token(
    *, num_heads: int, qk_head_dim: int, v_head_dim: int, seq_len: int, sliding_window: int | None = None
) -> int:
    """Training FLOPs per token of softmax attention, as torchtitan's `quadratic_attention_flops_per_token`.

    Two matmuls per head and attended token (q @ K^T over `qk_head_dim`, scores @ V over `v_head_dim`),
    2 FLOPs per multiply-add, 3x for forward plus backward. Causal sparsity and the flash-attention
    recompute are not counted.
    """
    attended_tokens = seq_len if sliding_window is None else min(seq_len, sliding_window)
    return 6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens


def delta_rule_flops_per_token(*, num_heads: int, key_head_dim: int, v_head_dim: int) -> int:
    """Training FLOPs per token of a gated delta-rule recurrence, as torchtitan's `delta_rule_flops_per_token`.

    Reading, updating and querying the `[key_head_dim, v_head_dim]` state are three
    `key_head_dim * v_head_dim` products per head.
    """
    return 6 * 3 * num_heads * key_head_dim * v_head_dim


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
        2: flash_attn_2_varlen_op,
        3: flash_attn_3_varlen_op,
        4: flash_attn_4_varlen_op,
    }

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
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

    def attention_flops_per_token(self, seq_len: int) -> int:
        return quadratic_attention_flops_per_token(
            num_heads=self.num_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            seq_len=seq_len,
            sliding_window=getattr(self, "sliding_window", None),
        )

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
        if self._flash_attn_version == 4:
            # FA4's flash_attn_varlen_func has qv as the 4th positional arg,
            # so cu_seqlens must be passed as keyword args to avoid misalignment.
            kwargs["cu_seqlens_q"] = cu_seqlens
            kwargs["cu_seqlens_k"] = cu_seqlens
            out, _ = self.func(q, k, v, **kwargs)
        else:
            out = self.func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)
        return out

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
