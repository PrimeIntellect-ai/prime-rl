import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb


class PrimeFlashAttention(FlashAttention):
    """FlashAttention with two optional changes to the attention output, in this order:

    - ``exclusive`` (XSA; Zhai 2026, "Exclusive Self Attention"): remove from each head's output its component along the
      token's own (normalized) value vector, so attention carries only context the residual path does not already have.
    - ``output_gate`` (Qiu et al. 2025, "Gated Attention for Large Language Models", their best variant; Qwen3-Next):
      multiply the output elementwise by ``sigmoid(g_proj(x))``, computed from the block input.

    The block input can also be a (query, key, value) triple of separate inputs (Gated Residual reads); the gate then
    reads the query input. Separate inputs need unfused q, k and v projections.
    """

    def __init__(self, config: AttentionConfig, flash_attn_version: int, *, exclusive: bool, output_gate: bool):
        super().__init__(config, flash_attn_version=flash_attn_version)
        self.exclusive = exclusive
        self.g_proj = nn.Linear(config.hidden_size, self.qkv_sizes[0], bias=False) if output_gate else None

    def forward(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        if isinstance(hidden_states, tuple):
            assert self.qkv_proj is None, "separate query, key and value inputs need model.fusions without qkv"
            query_input, key_input, value_input = hidden_states
            query_states, key_states, value_states = (
                self.q_proj(query_input),
                self.k_proj(key_input),
                self.v_proj(value_input),
            )
            hidden_states = query_input
        else:
            query_states, key_states, value_states = self.project_qkv(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.use_qk_norm and self.qk_norm_type == "per_layer":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)
        if self.use_qk_norm and self.qk_norm_type == "per_head":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states.transpose(1, 2), key_states.transpose(1, 2), cos, sin
            )
            query_states, key_states = query_states.transpose(1, 2), key_states.transpose(1, 2)

        out = self._compute_attention(query_states[0], key_states[0], value_states[0], cu_seqlens, max_seqlen)
        if self.exclusive:
            # Each query head's own value is its key/value group's.
            values = F.normalize(value_states[0], dim=-1).repeat_interleave(self.num_key_value_groups, dim=1)
            out = out - (out * values).sum(dim=-1, keepdim=True) * values
        attn_output = out.contiguous().view(1, out.shape[0], -1)
        if self.g_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(attn_output), None
