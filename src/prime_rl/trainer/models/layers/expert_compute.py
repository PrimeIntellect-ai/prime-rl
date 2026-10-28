# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


def to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class ExpertCompute(Protocol):
    """Expert compute on dispatched tokens grouped by expert. ``host_num_tokens_per_expert`` is
    ``num_tokens_per_expert`` as a CPU tensor when the dispatch has copied it to the host."""

    token_group_alignment: int

    def validate(self, experts: "GroupedExperts") -> None: ...

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        host_num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class FusedDispatchExpertCompute(Protocol):
    """Expert compute that also dispatches tokens to their experts and combines the results.

    It takes this rank's undispatched tokens and their routing, paired with a ``FusedTokenDispatcher``.
    """

    token_group_alignment: int

    def validate(self, experts: "GroupedExperts") -> None: ...

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
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


class GroupedGemmExpertCompute:
    def __init__(
        self,
        gemm: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        token_group_alignment: int = 8,
    ) -> None:
        self.gemm = gemm
        self.token_group_alignment = token_group_alignment

    def validate(self, experts: "GroupedExperts") -> None:
        """The shared forward handles the experts' activation, biases, and weight layout."""

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        host_num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 2

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()

        if experts.gate_up_proj is None:
            up_proj = to_local(experts.up_proj).transpose(-2, -1)
            up = self._grouped_mm(x_bf16, up_proj.bfloat16(), offsets, host_num_tokens_per_expert)

            gate = None
            if experts.gate_proj is not None:
                gate_proj = to_local(experts.gate_proj).transpose(-2, -1)
                gate = self._grouped_mm(x_bf16, gate_proj.bfloat16(), offsets, host_num_tokens_per_expert)
        else:
            gate_up_proj = to_local(experts.gate_up_proj).transpose(-2, -1)
            gate_up = self._grouped_mm(x_bf16, gate_up_proj.bfloat16(), offsets, host_num_tokens_per_expert)
            gate, up = gate_up.chunk(2, dim=-1)

        if experts.up_proj_bias is not None:
            up_proj_bias = to_local(experts.up_proj_bias)
            up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        if gate is not None and experts.gate_proj_bias is not None:
            gate_proj_bias = to_local(experts.gate_proj_bias)
            gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()

        hidden = experts.activation.apply(gate, up)
        down_proj = to_local(experts.down_proj).transpose(-2, -1)
        output = self._grouped_mm(hidden, down_proj.bfloat16(), offsets, host_num_tokens_per_expert)
        if experts.down_proj_bias is not None:
            down_proj_bias = to_local(experts.down_proj_bias)
            output = output + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, output.shape[0]).bfloat16()
        return output.type_as(x)

    def _grouped_mm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        host_num_tokens_per_expert: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.gemm(x, weight, offsets)


class BF16ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        super().__init__(torch._grouped_mm)


class DeepGemmFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        from prime_rl.trainer.models.kernels.fp8_utils import GROUP_ALIGNMENT
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        super().__init__(grouped_fp8_gemm, token_group_alignment=GROUP_ALIGNMENT)

    def _grouped_mm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        host_num_tokens_per_expert: torch.Tensor | None,
    ) -> torch.Tensor:
        # The weight gradient's grouped GEMM takes the rows per expert on the host.
        return self.gemm(x, weight, offsets, host_num_tokens_per_expert)


class MXFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, kernel: ModuleType, high_precision_wgrad: bool) -> None:
        super().__init__(
            partial(kernel.grouped_gemm, high_precision_wgrad=high_precision_wgrad),
            token_group_alignment=kernel.TOKEN_GROUP_ALIGNMENT,
        )
