import torch
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from fla.modules.conv import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import nn
from torch.distributed import ProcessGroup

# FLA's context carries a process group that Dynamo cannot trace through the convolution.
causal_conv1d_with_context_parallelism = torch.compiler.disable(causal_conv1d)


class GatedDeltaNet(nn.Module):
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
    ) -> None:
        super().__init__()
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_kernel_size = conv_kernel_size

        self.in_proj_qkv = nn.Linear(hidden_size, 2 * self.key_dim + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_value_heads, bias=False)

        conv_dim = 2 * self.key_dim + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=conv_kernel_size,
            groups=conv_dim,
            padding=conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        A = torch.empty(num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(value_head_dim, eps=norm_eps, activation="sigmoid")
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.context_parallel_group: ProcessGroup | None = None

    def set_context_parallel_group(self, process_group: ProcessGroup) -> None:
        self.context_parallel_group = process_group

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        output_gate = self.in_proj_z(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_value_heads,
            self.value_head_dim,
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        context = None
        if self.context_parallel_group is not None:
            context = build_cp_context(
                cu_seqlens=cu_seqlens.to(device=hidden_states.device, dtype=torch.int32),
                group=self.context_parallel_group,
                conv1d_kernel_size=self.conv_kernel_size,
            )

        convolution = {
            "x": mixed_qkv,
            "weight": self.conv1d.weight.squeeze(1),
            "bias": self.conv1d.bias,
            "activation": "silu",
        }
        if context is None:
            mixed_qkv, _ = causal_conv1d(**convolution, cu_seqlens=cu_seqlens)
        else:
            mixed_qkv, _ = causal_conv1d_with_context_parallelism(**convolution, cp_context=context)

        query, key, value = mixed_qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_value_heads, self.value_head_dim)

        heads_per_key = self.num_value_heads // self.num_key_heads
        if heads_per_key > 1:
            query = query.repeat_interleave(heads_per_key, dim=2)
            key = key.repeat_interleave(heads_per_key, dim=2)

        delta_rule = {
            "q": query,
            "k": key,
            "v": value,
            "g": decay,
            "beta": beta,
            "use_qk_l2norm_in_kernel": True,
            "cu_seqlens": context.cu_seqlens if context is not None else cu_seqlens,
        }
        if context is None:
            core_output, _ = chunk_gated_delta_rule(
                **delta_rule,
                initial_state=None,
                output_final_state=False,
            )
        else:
            core_output, _ = chunk_gated_delta_rule(**delta_rule, cp_context=context)

        core_output = self.norm(
            core_output.reshape(-1, self.value_head_dim),
            output_gate.reshape(-1, self.value_head_dim),
        )
        return self.out_proj(core_output.reshape(batch_size, sequence_length, self.value_dim))


__all__ = ["GatedDeltaNet"]
