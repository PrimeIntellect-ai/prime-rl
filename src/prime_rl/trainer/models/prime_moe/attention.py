import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention


class PrimeFlashAttention(FlashAttention):
    """FlashAttention with two optional changes to the attention output, in this order:

    - ``exclusive`` (XSA; Zhai 2026, "Exclusive Self Attention"): remove from each head's output its component along the
      token's own (normalized) value vector, so attention carries only context the residual path does not already have.
    - ``output_gate`` (Qiu et al. 2025, "Gated Attention for Large Language Models", their best variant; Qwen3-Next):
      multiply the output elementwise by ``sigmoid(g_proj(x))``, computed from the block input.
    """

    def __init__(self, config: AttentionConfig, flash_attn_version: int, *, exclusive: bool, output_gate: bool):
        super().__init__(config, flash_attn_version=flash_attn_version)
        self.exclusive = exclusive
        self.g_proj = nn.Linear(config.hidden_size, self.qkv_sizes[0], bias=False) if output_gate else None

    def project_output(self, out: torch.Tensor, values: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.exclusive:
            # Each query head's own value is its key/value group's.
            values = F.normalize(values, dim=-1).repeat_interleave(self.num_key_value_groups, dim=1)
            out = out - (out * values).sum(dim=-1, keepdim=True) * values
        attn_output = out.contiguous().view(1, out.shape[0], -1)
        if self.g_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(attn_output)
