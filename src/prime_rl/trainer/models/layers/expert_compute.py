# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Protocol

import torch
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


class ExpertCompute(Protocol):
    def __call__(
        self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor: ...


def broadcast_expert_bias(
    bias: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    repeats = num_tokens_per_expert.to(torch.int64)
    padding_rows = repeats.new_tensor(target_rows) - repeats.sum()
    return torch.repeat_interleave(
        torch.cat((bias, bias.new_zeros((1, bias.shape[1])))),
        torch.cat((repeats, padding_rows.unsqueeze(0))),
        dim=0,
        output_size=target_rows,
    )


class GroupedGemmExpertCompute(ExpertCompute):
    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2

        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()

        if experts.gate_up_proj is None:
            up_proj = to_local(experts.up_proj).transpose(-2, -1)
            up = experts.grouped_gemm(x_bf16, up_proj.bfloat16(), offs=offsets)

            gate = None
            if experts.gate_proj is not None:
                gate_proj = to_local(experts.gate_proj).transpose(-2, -1)
                gate = experts.grouped_gemm(x_bf16, gate_proj.bfloat16(), offs=offsets)
        else:
            gate_up_proj = to_local(experts.gate_up_proj).transpose(-2, -1)
            gate_up = experts.grouped_gemm(x_bf16, gate_up_proj.bfloat16(), offs=offsets)
            gate, up = gate_up.chunk(2, dim=-1)

        if experts.up_proj_bias is not None:
            up_proj_bias = to_local(experts.up_proj_bias)
            up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        if gate is not None and experts.gate_proj_bias is not None:
            gate_proj_bias = to_local(experts.gate_proj_bias)
            gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()

        hidden = experts.activation.apply(gate, up)
        down_proj = to_local(experts.down_proj).transpose(-2, -1)
        output = experts.grouped_gemm(hidden, down_proj.bfloat16(), offs=offsets)
        if experts.down_proj_bias is not None:
            down_proj_bias = to_local(experts.down_proj_bias)
            output = output + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, output.shape[0]).bfloat16()
        return output.type_as(x)
