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


class ExpertCompute(Protocol):
    token_group_alignment: int

    def validate(self, experts: "GroupedExperts") -> None: ...

    def __call__(
        self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor: ...


class GroupedGemmExpertCompute:
    def __init__(
        self,
        gemm: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        token_group_alignment: int = 8,
    ) -> None:
        self.gemm = gemm
        self.token_group_alignment = token_group_alignment

    def validate(self, experts: "GroupedExperts") -> None:
        """The shared forward handles the experts' activation and weight layout."""

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2

        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()

        if experts.gate_up_proj is None:
            up_proj = to_local(experts.up_proj).transpose(-2, -1)
            up = self.gemm(x_bf16, up_proj.bfloat16(), offsets)

            gate = None
            if experts.gate_proj is not None:
                gate_proj = to_local(experts.gate_proj).transpose(-2, -1)
                gate = self.gemm(x_bf16, gate_proj.bfloat16(), offsets)
        else:
            gate_up_proj = to_local(experts.gate_up_proj).transpose(-2, -1)
            gate_up = self.gemm(x_bf16, gate_up_proj.bfloat16(), offsets)
            gate, up = gate_up.chunk(2, dim=-1)

        hidden = experts.activation.apply(gate, up)
        down_proj = to_local(experts.down_proj).transpose(-2, -1)
        output = self.gemm(hidden, down_proj.bfloat16(), offsets)
        return output.type_as(x)


class BF16ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        super().__init__(torch._grouped_mm)


class DeepGemmFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self) -> None:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        super().__init__(grouped_fp8_gemm)


class MXFP8ExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, kernel: ModuleType, high_precision_wgrad: bool) -> None:
        super().__init__(
            partial(kernel.grouped_gemm, high_precision_wgrad=high_precision_wgrad),
            token_group_alignment=kernel.TOKEN_GROUP_ALIGNMENT,
        )
