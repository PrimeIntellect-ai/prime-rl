"""GPT-OSS attention and context-parallel execution.

Learned sink logits are model parameters, not an attention-backend option. GPT-OSS
therefore owns its FA4 call and shards each sink with the corresponding query heads
in its Ulysses path.
"""

import torch
from torch import nn

from prime_rl.trainer.models.fusions import fuse_qkv_projections
from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from prime_rl.trainer.models.layers.attn import flash_attn_4_varlen_func, quadratic_attention_flops
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb


class GptOssAttention(nn.Module):
    supported_fusions = {"qkv": fuse_qkv_projections}

    def __init__(self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.qkv_sizes = (self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features)
        self.register_module("qkv_proj", None)
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        nn.init.normal_(self.sinks, mean=0.0, std=config.initializer_range)
        self.flash_attn = flash_attn_4_varlen_func

    def attention_flops(self, seq_len: int) -> int:
        return quadratic_attention_flops(
            num_heads=self.num_attention_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            seq_len=seq_len,
            attended_tokens=min(seq_len, self.sliding_window or seq_len),
        )

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (None, None)
        output, _ = self.flash_attn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=True,
            window_size=window_size,
            learnable_sink=self.sinks,
        )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if self.qkv_proj is None:
            query, key, value = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        else:
            query, key, value = self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(
            query.transpose(1, 2),
            key.transpose(1, 2),
            cos,
            sin,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        output = self.compute_attention(query[0], key[0], value[0], cu_seqlens, max_seqlen)
        return self.o_proj(output.contiguous().view(batch_size, sequence_length, -1))


def substitute_gpt_oss_ulysses_attention(process_group: torch.distributed.ProcessGroup) -> None:
    from prime_rl.trainer.models.layers.ulysses_attn import ULYSSES_PARAMS, ulysses_flash_attn_varlen_func

    cp_size = torch.distributed.get_world_size(process_group)

    def ulysses_attention(self, query, key, value, _cu_seqlens, _max_seqlen):
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        return ulysses_flash_attn_varlen_func(
            self.flash_attn,
            query,
            key,
            value,
            cu_seqlens_q=ULYSSES_PARAMS["cu_seqlens"],
            cu_seqlens_k=ULYSSES_PARAMS["cu_seqlens"],
            max_seqlen_q=ULYSSES_PARAMS["max_seqlen"],
            max_seqlen_k=ULYSSES_PARAMS["max_seqlen"],
            causal=True,
            cp_group=process_group,
            cp_size=cp_size,
            flash_attn_version=4,
            window_size=window_size,
            softmax_scale=self.scaling,
            learnable_sink=self.sinks,
        )

    GptOssAttention.compute_attention = ulysses_attention


__all__ = [
    "GptOssAttention",
    "substitute_gpt_oss_ulysses_attention",
]
