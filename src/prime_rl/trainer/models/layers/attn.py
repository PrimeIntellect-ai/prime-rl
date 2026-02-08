import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .rms_norm import RMSNorm, RMSNormConfig
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

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore


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

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

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

        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)
        self._kv_prefix_num_tokens = 0
        self._kv_prefix_init: Literal["normal", "zeros"] = "normal"
        self._kv_prefix_init_std = 0.02
        self.kv_prefix_key: nn.Parameter | None = None
        self.kv_prefix_value: nn.Parameter | None = None

    def kv_prefix_num_tokens(self) -> int:
        return self._kv_prefix_num_tokens

    def enable_kv_prefix(
        self,
        num_tokens: int,
        init: Literal["normal", "zeros"] = "normal",
        init_std: float = 0.02,
    ) -> None:
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if init_std < 0:
            raise ValueError(f"init_std must be >= 0, got {init_std}")

        self._kv_prefix_num_tokens = num_tokens
        self._kv_prefix_init = init
        self._kv_prefix_init_std = init_std

        num_key_value_heads = self.k_proj.out_features // self.head_dim
        self.kv_prefix_key = nn.Parameter(
            torch.empty(
                num_key_value_heads,
                num_tokens,
                self.head_dim,
                device=self.k_proj.weight.device,
                dtype=self.k_proj.weight.dtype,
            )
        )
        self.kv_prefix_value = nn.Parameter(
            torch.empty(
                num_key_value_heads,
                num_tokens,
                self.head_dim,
                device=self.v_proj.weight.device,
                dtype=self.v_proj.weight.dtype,
            )
        )

        if self.kv_prefix_key.device.type != "meta":
            self._init_kv_prefix_parameters()

    def _init_kv_prefix_parameters(self, generator: torch.Generator | None = None) -> None:
        if self._kv_prefix_num_tokens == 0:
            return
        assert self.kv_prefix_key is not None
        assert self.kv_prefix_value is not None

        if self._kv_prefix_init == "zeros":
            nn.init.zeros_(self.kv_prefix_key)
            nn.init.zeros_(self.kv_prefix_value)
            return

        nn.init.normal_(self.kv_prefix_key, mean=0.0, std=self._kv_prefix_init_std, generator=generator)
        nn.init.normal_(self.kv_prefix_value, mean=0.0, std=self._kv_prefix_init_std, generator=generator)

    def _prepend_prefix_for_packed_sequences(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens_q: torch.LongTensor,
        max_seqlen_q: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, int]:
        assert self.kv_prefix_key is not None
        assert self.kv_prefix_value is not None
        assert self._kv_prefix_num_tokens > 0

        prefix_key = self.kv_prefix_key.transpose(0, 1).to(dtype=key_states.dtype)
        prefix_value = self.kv_prefix_value.transpose(0, 1).to(dtype=value_states.dtype)

        merged_keys = []
        merged_values = []
        for seq_idx in range(cu_seqlens_q.shape[0] - 1):
            seq_start = int(cu_seqlens_q[seq_idx].item())
            seq_end = int(cu_seqlens_q[seq_idx + 1].item())
            merged_keys.append(prefix_key)
            merged_keys.append(key_states[seq_start:seq_end])
            merged_values.append(prefix_value)
            merged_values.append(value_states[seq_start:seq_end])

        prefixed_keys = torch.cat(merged_keys, dim=0).contiguous()
        prefixed_values = torch.cat(merged_values, dim=0).contiguous()

        seq_prefix_offsets = torch.arange(
            cu_seqlens_q.shape[0],
            device=cu_seqlens_q.device,
            dtype=cu_seqlens_q.dtype,
        ) * self._kv_prefix_num_tokens
        cu_seqlens_k = cu_seqlens_q + seq_prefix_offsets
        max_seqlen_k = max_seqlen_q + self._kv_prefix_num_tokens
        return prefixed_keys, prefixed_values, cu_seqlens_k, max_seqlen_k

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
        key_states_0 = key_states[0]
        value_states_0 = value_states[0]
        cu_seqlens_k = cu_seqlens
        max_seqlen_k = max_seqlen

        if self._kv_prefix_num_tokens > 0:
            if cu_seqlens is None or max_seqlen is None:
                raise ValueError("KV-prefix tuning requires packed sequence metadata (cu_seqlens/max_seqlen).")
            key_states_0, value_states_0, cu_seqlens_k, max_seqlen_k = self._prepend_prefix_for_packed_sequences(
                key_states_0,
                value_states_0,
                cu_seqlens,
                max_seqlen,
            )

        args = [
            query_states[0],
            key_states_0,
            value_states_0,
            cu_seqlens,
            cu_seqlens_k,
        ]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen_k])

        out = self._flash_attn_call(
            *args,
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
    "fa4": functools.partial(FlashAttention, flash_attn_version=4),
}


def substitute_prime_rl_flash_attn(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
    attn_impl: str = "flash_attention_2",
) -> None:
    from ring_flash_attn import llama3_flash_attn_varlen_func

    from .ring_attn import ring_fa3_varlen_func

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
