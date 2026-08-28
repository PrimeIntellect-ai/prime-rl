"""GPT-OSS attention and context-parallel execution.

Learned sink logits are model parameters, not an attention-backend option. GPT-OSS
therefore owns its FA4 call and shards each sink with the corresponding query heads
in its ring and Ulysses paths.
"""

import torch
from torch import nn

from prime_rl.trainer.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int) -> None:
        super().__init__()
        if config.attention_dropout != 0:
            raise ValueError("The custom GPT-OSS implementation does not support attention dropout")

        from flash_attn.cute import flash_attn_varlen_func

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
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        nn.init.normal_(self.sinks, mean=0.0, std=config.initializer_range)
        self.flash_attn = torch._dynamo.disable(flash_attn_varlen_func)

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (None, None)
        output = self.flash_attn(
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
        return output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError(f"Custom GPT-OSS expects one packed row, got batch size {batch_size}")

        query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

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


def substitute_gpt_oss_ring_attention(
    process_group: torch.distributed.ProcessGroup,
    heads_k_stride: int,
) -> None:
    from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

    from prime_rl.trainer.distributed.collectives import all_gather

    def ring_attention(self, query, key, value, _cu_seqlens, _max_seqlen):
        key_value_groups = self.num_attention_heads // self.num_key_value_heads
        local_k_slice = DATA_PARAMS["local_k_slice"]
        window_size = (self.sliding_window - 1, 0) if self.sliding_window is not None else (None, None)
        outputs = []

        for key_head_start in range(0, self.num_key_value_heads, heads_k_stride):
            key_head_stop = min(key_head_start + heads_k_stride, self.num_key_value_heads)
            query_head_start = key_head_start * key_value_groups
            query_head_stop = key_head_stop * key_value_groups

            gathered_key = all_gather(key[:, key_head_start:key_head_stop], 0, process_group)[local_k_slice]
            gathered_value = all_gather(value[:, key_head_start:key_head_stop], 0, process_group)[local_k_slice]
            output = self.flash_attn(
                query[:, query_head_start:query_head_stop],
                gathered_key,
                gathered_value,
                cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
                cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
                max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
                max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
                softmax_scale=self.scaling,
                causal=True,
                window_size=window_size,
                learnable_sink=self.sinks[query_head_start:query_head_stop],
            )
            outputs.append(output[0] if isinstance(output, tuple) else output)

        return torch.cat(outputs, dim=1)

    GptOssAttention.compute_attention = ring_attention


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
    "substitute_gpt_oss_ring_attention",
    "substitute_gpt_oss_ulysses_attention",
]
