import torch
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from torch import nn

from prime_rl.trainer.models.layers import fla_ops
from prime_rl.trainer.models.layers.attn import delta_rule_flops
from prime_rl.utils.cp import CPContext


class GatedDeltaNet(nn.Module):
    """Qwen3.5-style Gated DeltaNet linear attention (FLA kernels)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        norm_eps: float,
        activation: str,
        output_gate_activation: str,
    ) -> None:
        super().__init__()
        self.num_value_heads = num_value_heads
        self.num_key_heads = num_key_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = key_head_dim * num_key_heads
        self.value_dim = value_head_dim * num_value_heads
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=conv_kernel_size,
            groups=conv_dim,
            padding=conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.empty(num_value_heads))
        self.A_log = nn.Parameter(torch.empty(num_value_heads))
        self.norm = FusedRMSNormGated(value_head_dim, eps=norm_eps, activation=output_gate_activation)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.cp_context = CPContext()
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.ones_(self.dt_bias)
        # The lower bound keeps log(A) finite.
        self.A_log.uniform_(0.01, 16).log_()

    def attention_flops(self, seq_len: int) -> int:
        return delta_rule_flops(
            num_heads=self.num_value_heads,
            key_head_dim=self.key_head_dim,
            v_head_dim=self.value_head_dim,
            seq_len=seq_len,
        )

    def forward(self, hidden_states: torch.Tensor, fla_varlen: fla_ops.FlaVarlen) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        output_gate = self.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, self.num_value_heads, self.value_head_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        mixed_qkv = fla_ops.causal_conv1d(
            mixed_qkv, self.conv1d.weight.squeeze(1), self.activation, fla_varlen, self.cp_context
        )

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_value_heads, self.value_head_dim)

        heads_per_key = self.num_value_heads // self.num_key_heads
        if heads_per_key > 1:
            query = query.repeat_interleave(heads_per_key, dim=2)
            key = key.repeat_interleave(heads_per_key, dim=2)

        core_output = fla_ops.chunk_gated_delta_rule(query, key, value, decay, beta, fla_varlen, self.cp_context)

        core_output = fla_ops.rms_norm_gated(
            core_output.reshape(-1, self.value_head_dim),
            output_gate.reshape(-1, self.value_head_dim),
            self.norm.weight,
            self.norm.eps,
            self.norm.activation,
        ).reshape(batch_size, sequence_length, self.value_dim)
        return self.out_proj(core_output)


__all__ = ["GatedDeltaNet"]
