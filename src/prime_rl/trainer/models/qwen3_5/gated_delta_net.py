import torch
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from fla.modules.conv import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import nn

from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

# FLA's CP convolution exchanges boundary state inside the kernel call, which
# Dynamo cannot trace; keep that call eager while compiling the surrounding layer.
causal_conv1d_with_context_parallelism = torch.compiler.disable(causal_conv1d)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.num_value_heads = config.linear_num_value_heads
        self.num_key_heads = config.linear_num_key_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.key_head_dim * self.num_key_heads
        self.value_dim = self.value_head_dim * self.num_value_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads).uniform_(0, 16).log_())
        self.norm = FusedRMSNormGated(
            self.value_head_dim,
            eps=config.rms_norm_eps,
            activation=config.output_gate_type,
        )
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(config.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_value_heads, bias=False)
        self.context_parallel_group = None
        self.context_parallel_world_size = 1

    def set_context_parallel_attributes(self, process_group, world_size: int) -> None:
        self.context_parallel_group = process_group
        self.context_parallel_world_size = world_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        *,
        cu_seqlens_are_pre_shard: bool,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        output_gate = self.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, self.num_value_heads, self.value_head_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        context = None
        if self.context_parallel_group is not None:
            if not cu_seqlens_are_pre_shard:
                raise ValueError("Qwen3.5 context parallelism requires pre-shard sequence boundaries")
            context = build_cp_context(
                cu_seqlens=cu_seqlens.to(device=hidden_states.device, dtype=torch.int32),
                group=self.context_parallel_group,
                conv1d_kernel_size=self.conv_kernel_size,
            )

        conv_kwargs = {
            "x": mixed_qkv,
            "weight": self.conv1d.weight.squeeze(1),
            "bias": self.conv1d.bias,
            "activation": self.activation,
        }
        if context is None:
            mixed_qkv, _ = causal_conv1d(**conv_kwargs, cu_seqlens=cu_seqlens)
        else:
            mixed_qkv, _ = causal_conv1d_with_context_parallelism(**conv_kwargs, cp_context=context)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_value_heads, self.value_head_dim)

        heads_per_key = self.num_value_heads // self.num_key_heads
        if heads_per_key > 1:
            query = query.repeat_interleave(heads_per_key, dim=2)
            key = key.repeat_interleave(heads_per_key, dim=2)

        delta_kwargs = {
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
                **delta_kwargs,
                initial_state=None,
                output_final_state=False,
            )
        else:
            core_output, _ = chunk_gated_delta_rule(**delta_kwargs, cp_context=context)

        core_output = self.norm(
            core_output.reshape(-1, self.value_head_dim),
            output_gate.reshape(-1, self.value_head_dim),
        ).reshape(batch_size, sequence_length, self.value_dim)
        return self.out_proj(core_output)


__all__ = ["Qwen3_5GatedDeltaNet"]
