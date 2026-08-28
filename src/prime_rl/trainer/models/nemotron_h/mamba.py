"""Nemotron-H Mamba-2 and context-parallel execution.

Packed training needs the convolution and state-space scan to reset at every
document boundary. Under Ulysses context parallelism, each rank also needs the
full sequence for its local Mamba heads because recurrent state cannot be split
along the sequence. Nemotron-H owns that path here instead of adapting a
Transformers mixer or relying on attention-global state.
"""

import math

import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.distributed.collectives import all_to_all_single_equal
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig


def sequence_to_head_parallel(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    world_size: int,
) -> torch.Tensor:
    """Redistribute ``[B, S/world, D]`` into ``[B, S, D/world]``."""
    batch_size, local_sequence_length, feature_size = tensor.shape
    local_feature_size = feature_size // world_size
    tensor = tensor.reshape(batch_size, local_sequence_length, world_size, local_feature_size)
    tensor = tensor.permute(2, 0, 1, 3).contiguous()
    tensor = all_to_all_single_equal(tensor, process_group)
    return tensor.permute(1, 0, 2, 3).reshape(
        batch_size,
        world_size * local_sequence_length,
        local_feature_size,
    )


def head_to_sequence_parallel(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    world_size: int,
) -> torch.Tensor:
    """Redistribute ``[B, S, D/world]`` into ``[B, S/world, D]``."""
    batch_size, sequence_length, local_feature_size = tensor.shape
    local_sequence_length = sequence_length // world_size
    tensor = tensor.reshape(batch_size, world_size, local_sequence_length, local_feature_size)
    tensor = tensor.permute(1, 0, 2, 3).contiguous()
    tensor = all_to_all_single_equal(tensor, process_group)
    return tensor.permute(1, 2, 0, 3).reshape(
        batch_size,
        local_sequence_length,
        world_size * local_feature_size,
    )


class GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, group_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_size = group_size
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * torch.nn.functional.silu(gate.float())
        group_count = hidden_states.shape[-1] // self.group_size
        grouped_states = hidden_states.reshape(*hidden_states.shape[:-1], group_count, self.group_size)
        variance = grouped_states.square().mean(dim=-1, keepdim=True)
        hidden_states = (grouped_states * torch.rsqrt(variance + self.variance_epsilon)).flatten(-2)
        return (self.weight if weight is None else weight) * hidden_states.to(input_dtype)


class NemotronHMamba2(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        from fla.modules.conv import causal_conv1d
        from fla.ops.utils import prepare_sequence_ids
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        self.hidden_size = config.hidden_size
        self.state_size = config.ssm_state_size
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.num_groups = config.n_groups
        self.intermediate_size = self.num_heads * self.head_dim
        self.group_state_size = self.num_groups * self.state_size
        self.conv_size = self.intermediate_size + 2 * self.group_state_size
        self.chunk_size = config.chunk_size
        self.time_step_limit = config.time_step_limit
        self.activation = config.mamba_hidden_act

        projection_size = self.intermediate_size + self.conv_size + self.num_heads
        self.in_proj = nn.Linear(config.hidden_size, projection_size, bias=config.use_bias)
        self.conv1d = nn.Conv1d(
            self.conv_size,
            self.conv_size,
            kernel_size=config.conv_kernel,
            groups=self.conv_size,
            padding=config.conv_kernel - 1,
            bias=config.use_conv_bias,
        )
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))
        self.norm = GatedRMSNorm(
            self.intermediate_size,
            group_size=self.intermediate_size // self.num_groups,
            eps=config.layer_norm_epsilon,
        )
        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=config.use_bias)
        self.causal_conv1d = causal_conv1d
        self.prepare_sequence_ids = prepare_sequence_ids
        self.scan = mamba_chunk_scan_combined

        time_steps = torch.exp(
            torch.rand(self.num_heads, dtype=torch.float32)
            * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        ).clamp(min=config.time_step_floor)
        with torch.no_grad():
            self.dt_bias.copy_((time_steps + torch.log(-torch.expm1(-time_steps))).to(self.dt_bias.dtype))

    def set_context_parallel_attributes(
        self,
        process_group: dist.ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        if self.num_heads % world_size:
            raise ValueError(f"Mamba heads ({self.num_heads}) must be divisible by CP size ({world_size})")
        if self.num_groups % world_size:
            raise ValueError(f"Mamba groups ({self.num_groups}) must be divisible by CP size ({world_size})")
        self.process_group = process_group
        self.context_parallel_rank = rank
        self.context_parallel_world_size = world_size

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError(f"Nemotron-H Mamba expects one packed row, got batch size {batch_size}")

        projected_states = self.in_proj(hidden_states)
        gate, convolution_input, time_step = torch.split(
            projected_states,
            [self.intermediate_size, self.conv_size, self.num_heads],
            dim=-1,
        )

        num_heads = self.num_heads
        num_groups = self.num_groups
        intermediate_size = self.intermediate_size
        conv_weight = self.conv1d.weight
        conv_bias = self.conv1d.bias
        state_decay = -torch.exp(self.A_log.float())
        skip = self.D
        dt_bias = self.dt_bias
        norm_weight = self.norm.weight

        if hasattr(self, "process_group"):
            world_size = self.context_parallel_world_size
            rank = self.context_parallel_rank
            gate = sequence_to_head_parallel(gate, self.process_group, world_size)
            time_step = sequence_to_head_parallel(time_step, self.process_group, world_size)

            recurrent_input, state_input, state_output = torch.split(
                convolution_input,
                [self.intermediate_size, self.group_state_size, self.group_state_size],
                dim=-1,
            )
            convolution_input = torch.cat(
                [
                    sequence_to_head_parallel(recurrent_input, self.process_group, world_size),
                    sequence_to_head_parallel(state_input, self.process_group, world_size),
                    sequence_to_head_parallel(state_output, self.process_group, world_size),
                ],
                dim=-1,
            )

            num_heads //= world_size
            num_groups //= world_size
            intermediate_size //= world_size
            head_start = rank * num_heads
            group_start = rank * num_groups * self.state_size
            intermediate_start = rank * intermediate_size

            state_decay = state_decay[head_start : head_start + num_heads]
            skip = skip[head_start : head_start + num_heads]
            dt_bias = dt_bias[head_start : head_start + num_heads]
            norm_weight = norm_weight[intermediate_start : intermediate_start + intermediate_size]

            state_input_start = self.intermediate_size + group_start
            state_output_start = self.intermediate_size + self.group_state_size + group_start
            conv_weight = torch.cat(
                [
                    conv_weight[intermediate_start : intermediate_start + intermediate_size],
                    conv_weight[state_input_start : state_input_start + num_groups * self.state_size],
                    conv_weight[state_output_start : state_output_start + num_groups * self.state_size],
                ]
            )
            if conv_bias is not None:
                conv_bias = torch.cat(
                    [
                        conv_bias[intermediate_start : intermediate_start + intermediate_size],
                        conv_bias[state_input_start : state_input_start + num_groups * self.state_size],
                        conv_bias[state_output_start : state_output_start + num_groups * self.state_size],
                    ]
                )
            sequence_length = convolution_input.shape[1]

        convolution_output, _ = self.causal_conv1d(
            x=convolution_input,
            weight=conv_weight.squeeze(1),
            bias=conv_bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )
        local_group_state_size = num_groups * self.state_size
        hidden_states, state_input, state_output = torch.split(
            convolution_output,
            [intermediate_size, local_group_state_size, local_group_state_size],
            dim=-1,
        )

        scan_kwargs = {}
        if self.time_step_limit is not None:
            scan_kwargs["dt_limit"] = self.time_step_limit
        hidden_states = self.scan(
            hidden_states.reshape(batch_size, sequence_length, num_heads, self.head_dim),
            time_step,
            state_decay,
            state_input.reshape(batch_size, sequence_length, num_groups, self.state_size),
            state_output.reshape(batch_size, sequence_length, num_groups, self.state_size),
            chunk_size=self.chunk_size,
            D=skip,
            z=None,
            seq_idx=self.prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0),
            return_final_states=False,
            dt_bias=dt_bias,
            dt_softplus=True,
            **scan_kwargs,
        ).reshape(batch_size, sequence_length, intermediate_size)
        hidden_states = self.norm(hidden_states, gate, weight=norm_weight)

        if hasattr(self, "process_group"):
            hidden_states = head_to_sequence_parallel(
                hidden_states,
                self.process_group,
                self.context_parallel_world_size,
            )
        return self.out_proj(hidden_states)


__all__ = [
    "GatedRMSNorm",
    "NemotronHMamba2",
    "head_to_sequence_parallel",
    "sequence_to_head_parallel",
]
