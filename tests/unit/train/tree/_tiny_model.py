import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | BlockMask) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        if isinstance(attn_mask, BlockMask):
            hidden_states = flex_attention(
                query,
                key,
                value,
                block_mask=attn_mask,
                scale=1 / math.sqrt(self.head_dim),
            )
            hidden_states = hidden_states.transpose(1, 2).contiguous()
        else:
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            hidden_states = torch.matmul(attn, value).transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)
        hidden_states = residual + self.o_proj(hidden_states)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_position_embeddings: int = 256,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([TinyTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | BlockMask | None = None,
    ) -> torch.Tensor:
        hidden_states = self.token_embed(input_ids) + self.position_embed(position_ids)
        if attn_mask is None:
            seq_len = input_ids.shape[1]
            attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device).tril()
        if isinstance(attn_mask, torch.Tensor) and attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif isinstance(attn_mask, torch.Tensor) and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
        return self.lm_head(self.norm(hidden_states))


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device).tril()
