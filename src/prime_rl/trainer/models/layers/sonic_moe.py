from typing import TYPE_CHECKING

import torch
from sonicmoe import moe_general_routing_inputs
from sonicmoe.enums import ActivationType
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.expert_compute import ExpertCompute

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


class SonicMoEExpertCompute(ExpertCompute):
    token_group_alignment = 8

    def __call__(self, experts: "GroupedExperts", x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_up = experts.gate_up_proj
        down = experts.down_proj
        if isinstance(gate_up, DTensor):
            gate_up = gate_up.to_local()
            down = down.to_local()

        token_indices = torch.arange(x.shape[0], dtype=torch.int32, device=x.device)
        offsets = num_tokens_per_expert.cumsum(0, dtype=torch.int32)
        expert_indices = torch.searchsorted(offsets, token_indices, right=True)
        # Dispatch discards the trailing padding beyond the per-expert counts.
        expert_indices = expert_indices.clamp_max(gate_up.shape[0] - 1).to(torch.int32)
        output, _ = moe_general_routing_inputs(
            x=x.bfloat16(),
            router_scores=torch.ones(x.shape[0], dtype=torch.float32, device=x.device),
            token_indices=token_indices,
            expert_indices=expert_indices,
            w1=gate_up.bfloat16().permute(1, 2, 0),
            b1=None,
            w2=down.bfloat16().permute(1, 2, 0),
            b2=None,
            E=gate_up.shape[0],
            stream_id=0,
            activation_type=ActivationType.SWIGLU,
            concat_layout=True,
        )
        return output.type_as(x)
