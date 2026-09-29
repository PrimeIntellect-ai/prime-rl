import torch
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from fla.modules.conv import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch import nn

from prime_rl.trainer.models.layers.ulysses_attn import head_to_sequence_parallel, sequence_to_head_parallel
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from prime_rl.utils.cp import CPContext, gather_for_cp
from prime_rl.utils.sequence import CPPartition

# Dynamo lowers all-gather to concatenation, then fails to copy the result into
# FLA's stacked output buffer. Keep CP convolution eager until this is fixed:
# https://github.com/pytorch/pytorch/issues/155632
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
        self.cp_context = CPContext()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        cp_total_tokens: int | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        # FLA caches metadata by tensor identity; checkpoint replay must execute the same operations.
        cu_seqlens = cu_seqlens.clone()
        mixed_qkv = self.in_proj_qkv(hidden_states)
        output_gate = self.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, self.num_value_heads, self.value_head_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        partition = None
        head_parallel = False
        num_key_heads, num_value_heads = self.num_key_heads, self.num_value_heads
        key_dim, value_dim = self.key_dim, self.value_dim
        conv_weight = self.conv1d.weight.squeeze(1)
        if self.cp_context.cp_enabled and cp_total_tokens is not None:
            partition = CPPartition(cp_total_tokens, self.cp_context.cp_world_size)
            group, degree, rank = self.cp_context.cp_group, self.cp_context.cp_world_size, self.cp_context.cp_rank
            head_parallel = self.num_value_heads % degree == 0
            if head_parallel:
                heads_per_key = self.num_value_heads // self.num_key_heads
                q, k, v = mixed_qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
                q = (
                    q.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
                    .repeat_interleave(heads_per_key, dim=2)
                    .flatten(2)
                )
                k = (
                    k.reshape(batch_size, sequence_length, self.num_key_heads, self.key_head_dim)
                    .repeat_interleave(heads_per_key, dim=2)
                    .flatten(2)
                )
                mixed_qkv = torch.cat(
                    [sequence_to_head_parallel(x, group, degree, cp_total_tokens) for x in (q, k, v)], dim=-1
                )
                num_key_heads = num_value_heads = self.num_value_heads // degree
                key_dim = num_key_heads * self.key_head_dim
                value_dim = num_value_heads * self.value_head_dim
                q_weight, k_weight, v_weight = conv_weight.split((self.key_dim, self.key_dim, self.value_dim))
                head_slice = slice(rank * num_value_heads, (rank + 1) * num_value_heads)
                weights = []
                for weight, head_dim, repeats in (
                    (q_weight, self.key_head_dim, heads_per_key),
                    (k_weight, self.key_head_dim, heads_per_key),
                    (v_weight, self.value_head_dim, 1),
                ):
                    weight = weight.reshape(-1, head_dim, self.conv_kernel_size).repeat_interleave(repeats, dim=0)
                    weights.append(weight[head_slice].flatten(0, 1))
                conv_weight = torch.cat(weights)
                beta = sequence_to_head_parallel(beta, group, degree, cp_total_tokens)
                decay = sequence_to_head_parallel(decay, group, degree, cp_total_tokens)
                output_gate = sequence_to_head_parallel(output_gate.flatten(2), group, degree, cp_total_tokens).reshape(
                    batch_size, cp_total_tokens, num_value_heads, self.value_head_dim
                )
            else:
                mixed_qkv = gather_for_cp(mixed_qkv, group, cp_total_tokens)
                beta = gather_for_cp(beta, group, cp_total_tokens)
                decay = gather_for_cp(decay, group, cp_total_tokens)
            sequence_length = cp_total_tokens

        context = None
        if self.cp_context.cp_enabled and partition is None:
            context = build_cp_context(
                cu_seqlens=cu_seqlens.to(device=hidden_states.device, dtype=torch.int32),
                group=self.cp_context.cp_group,
                conv1d_kernel_size=self.conv_kernel_size,
            )

        if sequence_length == 0:
            anchor = mixed_qkv.sum() + beta.sum() + decay.sum() + conv_weight.sum()
            core_output = mixed_qkv[..., :value_dim].reshape(
                batch_size, 0, num_value_heads, self.value_head_dim
            ) + anchor.to(mixed_qkv.dtype)
        else:
            convolution = causal_conv1d_with_context_parallelism if context is not None else causal_conv1d
            mixed_qkv, _ = convolution(
                x=mixed_qkv,
                weight=conv_weight,
                bias=self.conv1d.bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
                cp_context=context,
            )

            query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
            query = query.reshape(batch_size, sequence_length, num_key_heads, self.key_head_dim)
            key = key.reshape(batch_size, sequence_length, num_key_heads, self.key_head_dim)
            value = value.reshape(batch_size, sequence_length, num_value_heads, self.value_head_dim)

            heads_per_key = num_value_heads // num_key_heads
            if heads_per_key > 1:
                query = query.repeat_interleave(heads_per_key, dim=2)
                key = key.repeat_interleave(heads_per_key, dim=2)

            core_output, _ = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=decay,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=context.cu_seqlens if context is not None else cu_seqlens,
                cp_context=context,
            )

        if partition is not None and not head_parallel:
            core_output = partition.shard(core_output, self.cp_context.cp_rank)
            sequence_length = core_output.shape[1]
        if sequence_length == 0:
            core_output = (core_output + output_gate + self.norm.weight.sum().to(core_output.dtype)).reshape(
                batch_size, 0, value_dim
            )
        else:
            core_output = self.norm(
                core_output.reshape(-1, self.value_head_dim),
                output_gate.reshape(-1, self.value_head_dim),
            ).reshape(batch_size, sequence_length, value_dim)
        if head_parallel:
            core_output = head_to_sequence_parallel(core_output, group, degree, cp_total_tokens)
        return self.out_proj(core_output)


__all__ = ["Qwen3_5GatedDeltaNet"]
