import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .rms_norm import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None


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


# TODO: Does torch compile support config._attn_implementation forking?
# If so, we can combine FlashAttention and SDPAAttention into one class
# Otherwise, do ABC or something to make the signatures match


class FlashAttention(nn.Module):
    """Flash Attention"""

    def __init__(self, config: AttentionConfig, flash_attn_version: int = 2):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
            self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

        self.func = flash_attn_3_varlen_func if flash_attn_version == 3 else flash_attn_varlen_func

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary application instead of double transpose?
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
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

        out = out.contiguous()
        attn_output = out.view(1, out.shape[0], -1)
        attn_weights = None

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SDPAAttention(nn.Module):
    """SDPA Attention"""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
            self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states: torch.Tensor = self.q_proj(hidden_states).view(hidden_shape)
        key_states: torch.Tensor = self.k_proj(hidden_states).view(hidden_shape)
        value_states: torch.Tensor = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary application instead of double transpose?
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        out = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
        out = out.transpose(1, 2).contiguous()  # .view(out.shape[0], out.shape[1], -1)
        attn_output = out.view(out.shape[0], out.shape[1], -1)
        attn_weights = None

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


ATTN_IMPL2CLASS = {
    "flash_attention_2": functools.partial(FlashAttention, flash_attn_version=2),
    "sdpa": SDPAAttention,
    "flash_attention_3": functools.partial(FlashAttention, flash_attn_version=3),
}


def substitute_prime_rl_flash_attn(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
    attn_impl: str = "flash_attention_2",
) -> None:
    from ring_flash_attn import llama3_flash_attn_varlen_func
    from ring_flash_attn.utils import AllGatherComm, get_default_args

    def ring_fa3_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        local_k_slice: slice,
        causal: bool,
        heads_k_stride: int,
        group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        def fa3_forward(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            softmax_scale: float,
            causal: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            from flash_attn_interface import _flash_attn_forward

            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "cu_seqlens_q": cu_seqlens_q,
                    "cu_seqlens_k": cu_seqlens_k,
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                }
            )
            out, lse, _, _ = _flash_attn_forward(**params)
            return out, lse

        class RingFA3Varlen(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v):
                softmax_scale = q.shape[-1] ** (-0.5)
                out_list = []
                lse_list = []

                nheads = q.shape[1]
                total_k, nheads_k, head_dim = k.shape
                if nheads_k % heads_k_stride != 0:
                    raise ValueError("nheads_k must be divisible by heads_k_stride")

                world_size = torch.distributed.get_world_size(group)
                kv_buffer = torch.empty(
                    (2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device
                )
                kv_buffer_copy = torch.empty_like(kv_buffer)
                comm = AllGatherComm(group)

                comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride].contiguous())
                comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride].contiguous())

                for i in range(0, nheads_k, heads_k_stride):
                    comm.wait()
                    kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

                    if i < nheads_k - heads_k_stride:
                        left = i + heads_k_stride
                        right = left + heads_k_stride
                        comm.all_gather(kv_buffer_copy[0], k[:, left:right].contiguous())
                        comm.all_gather(kv_buffer_copy[1], v[:, left:right].contiguous())

                    q_i = q[:, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
                    k_i = kv_buffer[0][local_k_slice]
                    v_i = kv_buffer[1][local_k_slice]
                    out_i, lse_i = fa3_forward(
                        q=q_i,
                        k=k_i,
                        v=v_i,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                    out_list.append(out_i)
                    lse_list.append(lse_i)

                out = torch.cat(out_list, dim=1)
                lse = torch.cat(lse_list, dim=-2)

                ctx.save_for_backward(q, k, v, out, lse)
                ctx.softmax_scale = softmax_scale
                return out

            @staticmethod
            def backward(ctx, dout):
                from flash_attn_interface import _flash_attn_backward

                q, k, v, out, softmax_lse = ctx.saved_tensors
                nheads = q.shape[1]
                total_k, nheads_k, head_dim = k.shape

                world_size = torch.distributed.get_world_size(group)
                kv_buffer = torch.empty(
                    (2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device
                )
                kv_buffer_copy = torch.empty_like(kv_buffer)
                dkv_buffer = torch.empty(
                    (2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device
                )

                if heads_k_stride != nheads_k:
                    kv_contiguous_buffer = torch.empty(
                        (2, total_k, heads_k_stride, head_dim), dtype=k.dtype, device=k.device
                    )

                dq = torch.empty_like(q)
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)

                comm = AllGatherComm(group)
                comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride].contiguous())
                comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride].contiguous())

                for i in range(0, nheads_k, heads_k_stride):
                    dkv_buffer.zero_()
                    q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
                    q_i = q[:, q_slice]
                    dout_i = dout[:, q_slice]
                    out_i = out[:, q_slice]
                    dq_i = dq[:, q_slice]
                    lse_i = softmax_lse[q_slice].contiguous()

                    comm.wait()
                    kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
                    if i < nheads_k - heads_k_stride:
                        left = i + heads_k_stride
                        right = left + heads_k_stride
                        comm.all_gather(kv_buffer_copy[0], k[:, left:right].contiguous())
                        comm.all_gather(kv_buffer_copy[1], v[:, left:right].contiguous())

                    k_i = kv_buffer[0][local_k_slice]
                    v_i = kv_buffer[1][local_k_slice]
                    dk_i = dkv_buffer[0][local_k_slice]
                    dv_i = dkv_buffer[1][local_k_slice]

                    params = get_default_args(_flash_attn_backward).copy()
                    params.update(
                        {
                            "dout": dout_i,
                            "q": q_i,
                            "k": k_i,
                            "v": v_i,
                            "out": out_i,
                            "softmax_lse": lse_i,
                            "cu_seqlens_q": cu_seqlens_q,
                            "cu_seqlens_k": cu_seqlens_k,
                            "max_seqlen_q": max_seqlen_q,
                            "max_seqlen_k": max_seqlen_k,
                            "dq": dq_i,
                            "dk": dk_i,
                            "dv": dv_i,
                            "softmax_scale": ctx.softmax_scale,
                            "causal": causal,
                        }
                    )
                    _flash_attn_backward(**params)

                    if heads_k_stride != nheads_k:
                        dk_i = kv_contiguous_buffer[0]
                        dv_i = kv_contiguous_buffer[1]
                    else:
                        dk_i = dk
                        dv_i = dv

                    torch.distributed.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=group)
                    torch.distributed.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=group)
                    if heads_k_stride != nheads_k:
                        dk[:, i : i + heads_k_stride] = dk_i
                        dv[:, i : i + heads_k_stride] = dv_i

                return dq, dk, dv

        return RingFA3Varlen.apply(q, k, v)

    use_fa3 = attn_impl == "flash_attention_3"
    ring_func = ring_fa3_varlen_func if use_fa3 else llama3_flash_attn_varlen_func

    class RingFlashAttention(FlashAttention):
        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            cu_seqlens: torch.LongTensor | None = None,
            max_seqlen: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

            if self.use_qk_norm:  # main diff from Llama
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

            cu_seqlens_q = DATA_PARAMS["cu_seqlens_q"]
            cu_seqlens_k = DATA_PARAMS["cu_seqlens_k"]
            max_seqlen_q = DATA_PARAMS["max_seqlen_q"]
            max_seqlen_k = DATA_PARAMS["max_seqlen_k"]
            local_k_slice = DATA_PARAMS["local_k_slice"]

            # TODO: Can we optimize the rotary application instead of double transpose?
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            out = ring_func(
                query_states[0],
                key_states[0],
                value_states[0],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                local_k_slice=local_k_slice,
                causal=True,
                group=process_group,
                heads_k_stride=heads_k_stride,
            )
            if isinstance(out, tuple):
                out = out[0]
            out = out.contiguous()
            attn_output = out.view(1, out.shape[0], -1)
            attn_weights = None

            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    FlashAttention.forward = RingFlashAttention.forward
