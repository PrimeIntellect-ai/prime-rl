import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from prime_rl.trainer.distributed.comet_moe.autograd import CometMoELayerFunction, init_comet_moe_grad_buffers
from prime_rl.trainer.distributed.comet_moe.buffers import CometMoEBuffers, init_comet_moe_buffers
from prime_rl.trainer.models.layers.moe import GroupedExperts


def _to_local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class CometMoETokenDispatcher:
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        group: dist.ProcessGroup,
        block_m: int = 128,
        n_blocks: int = 132,
        capacity_multiplier: int = 4,
        n_chunks: int = 1,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.group = group
        self.block_m = block_m
        self.n_blocks = n_blocks
        self.capacity_multiplier = capacity_multiplier
        self.n_chunks = n_chunks
        self._bufs: CometMoEBuffers | None = None
        self._grad_recv: CometMoEBuffers | None = None
        self._grad_combine: CometMoEBuffers | None = None

    def _ensure_buffers(self, n_local_tokens: int, dim: int, device: torch.device) -> None:
        if self._bufs is not None:
            return

        capacity = self.capacity_multiplier * n_local_tokens * self.top_k + self.num_experts * self.block_m
        capacity = ((capacity + self.block_m - 1) // self.block_m) * self.block_m
        self._bufs = init_comet_moe_buffers(
            self.group,
            hidden_dim=dim,
            dispatch_capacity=capacity,
            combine_capacity=capacity,
            block_m=self.block_m,
            dtype=torch.bfloat16,
            device=device,
        )
        self._grad_recv, self._grad_combine = init_comet_moe_grad_buffers(
            self.group,
            hidden_dim=dim,
            dispatch_capacity=capacity,
            combine_capacity=capacity,
            block_m=self.block_m,
            dtype=torch.bfloat16,
            device=device,
        )

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: GroupedExperts,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        if score_before_experts:
            raise NotImplementedError(
                "CometMoETokenDispatcher only supports score_before_experts=False "
                "(routing scores applied after the expert FFN, in its combine step)."
            )
        if not isinstance(experts, GroupedExperts):
            raise TypeError(f"CometMoETokenDispatcher needs a GroupedExperts, got {type(experts).__name__}.")
        if experts.gate_proj_bias is not None or experts.up_proj_bias is not None or experts.down_proj_bias is not None:
            raise NotImplementedError("CometMoETokenDispatcher does not support per-expert bias yet.")

        self._ensure_buffers(x.shape[0], x.shape[-1], x.device)

        up_proj = _to_local(experts.up_proj).bfloat16()
        down_proj = _to_local(experts.down_proj).bfloat16()
        gate_proj = _to_local(experts.gate_proj).bfloat16() if experts.gate_proj is not None else None

        return CometMoELayerFunction.apply(
            x.bfloat16(),
            top_scores,
            selected_experts_indices,
            up_proj,
            down_proj,
            gate_proj,
            experts.activation,
            self._bufs,
            self._grad_recv,
            self._grad_combine,
            self.group,
            self.num_experts,
            self.top_k,
            self.block_m,
            self.n_blocks,
            self.n_chunks,
        ).type_as(x)

    def synchronize(self) -> None:
        return None
