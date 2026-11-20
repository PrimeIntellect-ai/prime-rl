import math

import torch
from fla.modules import FusedRMSNormGated
from torch import nn

from prime_rl.trainer.models.layers import fla_ops


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention with GLM-5.3's safe (lower-bounded) forget gate.

    The checkpoint keeps one depthwise convolution per projection; they run as a single
    convolution over the concatenated q/k/v channels, which is the same computation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.q_conv1d = self._depthwise_conv()
        self.k_conv1d = self._depthwise_conv()
        self.v_conv1d = self._depthwise_conv()

        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = nn.Parameter(torch.empty(num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size))
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # fla's KimiDeltaAttention init with a gate lower bound: dt is log-uniform in [1e-3, 0.1] and dt_bias is
        # its inverse softplus, computed in float32 and rounded to the parameter's dtype once.
        nn.init.zeros_(self.A_log)
        dt = torch.empty_like(self.dt_bias, dtype=torch.float32)
        dt.uniform_(math.log(1e-3), math.log(0.1)).exp_().clamp_(min=1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _depthwise_conv(self) -> nn.Conv1d:
        return nn.Conv1d(
            self.projection_size,
            self.projection_size,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, fla_varlen: fla_ops.FlaVarlen) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)), dim=-1
        )
        conv_weight = torch.cat((self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight)).squeeze(1)
        mixed_qkv = fla_ops.causal_conv1d(mixed_qkv, conv_weight, "silu", fla_varlen)

        head_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query, key, value = (part.reshape(head_shape) for part in mixed_qkv.chunk(3, dim=-1))
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(head_shape)
        beta = self.b_proj(hidden_states)

        core_output = fla_ops.chunk_kda(
            query, key, value, forget_gate, beta, self.A_log, self.dt_bias, self.gate_lower_bound, fla_varlen
        )

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        core_output = fla_ops.rms_norm_gated(
            core_output.reshape(-1, self.head_dim),
            output_gate.reshape(-1, self.head_dim),
            self.o_norm.weight,
            self.o_norm.eps,
            self.o_norm.activation,
        )
        return self.o_proj(core_output.reshape(batch_size, sequence_length, self.projection_size))


__all__ = ["KimiDeltaAttention"]
