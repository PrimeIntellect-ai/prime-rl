from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class ExpertFunction(Protocol):
    def __call__(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor: ...


class TokenDispatcher(Protocol):
    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class PermutationState:
    input_shape: torch.Size
    permuted_indices: torch.Tensor


@dataclass(frozen=True)
class LocalDispatchState:
    num_tokens: int
    token_indices_experts_sorted: torch.Tensor
    scores_after_experts: torch.Tensor | None
    permutation: PermutationState


@dataclass(frozen=True)
class TorchDispatchState(LocalDispatchState):
    input_splits: list[int]
    output_splits: list[int]


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: ProcessGroup,
    ) -> torch.Tensor:
        output = x.new_empty((sum(output_splits), *x.shape[1:]))
        dist.all_to_all_single(output, x.contiguous(), output_splits, input_splits, group=group)
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.new_empty((sum(ctx.input_splits), *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: ProcessGroup,
) -> torch.Tensor:
    return _AllToAll.apply(x, output_splits, input_splits, group)


def permute_for_grouped_gemm(
    x: torch.Tensor,
    num_tokens_per_expert_group: torch.Tensor,
    *,
    experts_per_rank: int,
    num_ranks: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, PermutationState]:
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    max_len = x.shape[0] + experts_per_rank * alignment
    max_len = (max_len + alignment - 1) // alignment * alignment
    with torch.no_grad():
        permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
            num_tokens_per_expert_group,
            experts_per_rank,
            num_ranks,
            max_len,
            alignment,
            use_cpu=x.device.type == "cpu",
        )

    x = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))
    state = PermutationState(input_shape=x.shape, permuted_indices=permuted_indices)
    return x[permuted_indices], num_tokens_per_expert, state


def unpermute_from_grouped_gemm(x: torch.Tensor, state: PermutationState) -> torch.Tensor:
    output = x.new_empty(state.input_shape)
    output[state.permuted_indices] = x
    return output[:-1]


def _local_reorder(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened_experts = selected_experts_indices.reshape(-1)
    num_tokens_per_expert = torch.histc(
        flattened_experts.float(),
        bins=num_experts,
        min=0,
        max=num_experts,
    ).to(torch.int64)
    token_indices_experts_sorted = torch.argsort(flattened_experts, stable=True)
    top_scores_experts_sorted = top_scores.reshape(-1)[token_indices_experts_sorted]
    token_indices_experts_sorted = token_indices_experts_sorted // top_k
    routed_input = x[token_indices_experts_sorted]
    return routed_input, token_indices_experts_sorted, top_scores_experts_sorted, num_tokens_per_expert


def _scatter_routed_output(
    routed_output: torch.Tensor,
    *,
    num_tokens: int,
    token_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
) -> torch.Tensor:
    if scores_after_experts is not None:
        routed_output = (routed_output.float() * scores_after_experts.reshape(-1, 1)).to(routed_output.dtype)

    dim = routed_output.shape[-1]
    output = routed_output.new_zeros((num_tokens, dim))
    routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
    return output.scatter_add(0, routed_indices, routed_output)


class LocalTokenDispatcher:
    def __init__(self, num_experts: int, top_k: int, token_group_alignment: int) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.token_group_alignment = token_group_alignment

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, LocalDispatchState]:
        routed_input, token_indices, sorted_scores, num_tokens_per_expert = _local_reorder(
            x,
            top_scores,
            selected_experts_indices,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        scores_after_experts = None
        if score_before_experts:
            routed_input = (routed_input.float() * sorted_scores.reshape(-1, 1)).to(x.dtype)
        else:
            scores_after_experts = sorted_scores

        routed_input, num_tokens_per_expert, permutation = permute_for_grouped_gemm(
            routed_input,
            num_tokens_per_expert,
            experts_per_rank=self.num_experts,
            num_ranks=1,
            alignment=self.token_group_alignment,
        )
        state = LocalDispatchState(
            num_tokens=x.shape[0],
            token_indices_experts_sorted=token_indices,
            scores_after_experts=scores_after_experts,
            permutation=permutation,
        )
        return routed_input, num_tokens_per_expert, state

    def combine(self, routed_output: torch.Tensor, state: LocalDispatchState) -> torch.Tensor:
        routed_output = unpermute_from_grouped_gemm(routed_output, state.permutation)
        return _scatter_routed_output(
            routed_output,
            num_tokens=state.num_tokens,
            token_indices_experts_sorted=state.token_indices_experts_sorted,
            scores_after_experts=state.scores_after_experts,
        )

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        routed_input, num_tokens_per_expert, state = self.dispatch(
            x,
            top_scores,
            selected_experts_indices,
            score_before_experts=score_before_experts,
        )
        routed_output = experts(routed_input, num_tokens_per_expert)
        return self.combine(routed_output, state)

    def synchronize(self) -> None:
        return None


class TorchTokenDispatcher(LocalTokenDispatcher):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        token_group_alignment: int,
        group: ProcessGroup,
    ) -> None:
        super().__init__(num_experts, top_k, token_group_alignment)
        self.group = group

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        return _all_to_all(x, output_splits, input_splits, self.group)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        return _all_to_all(x, output_splits, input_splits, self.group)

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, TorchDispatchState]:
        routed_input, token_indices, sorted_scores, num_tokens_per_expert = _local_reorder(
            x,
            top_scores,
            selected_experts_indices,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        scores_after_experts = None
        if score_before_experts:
            routed_input = (routed_input.float() * sorted_scores.reshape(-1, 1)).to(x.dtype)
        else:
            scores_after_experts = sorted_scores

        ep_degree = self.group.size()
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        with torch.no_grad():
            dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert, group=self.group)
            input_splits = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1).tolist()
            output_splits = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1).tolist()

        routed_input = self._dispatch_tokens(routed_input, output_splits, input_splits)
        experts_per_rank = self.num_experts // ep_degree
        routed_input, num_tokens_per_expert, permutation = permute_for_grouped_gemm(
            routed_input,
            num_tokens_per_expert_group,
            experts_per_rank=experts_per_rank,
            num_ranks=ep_degree,
            alignment=self.token_group_alignment,
        )
        state = TorchDispatchState(
            num_tokens=x.shape[0],
            token_indices_experts_sorted=token_indices,
            scores_after_experts=scores_after_experts,
            permutation=permutation,
            input_splits=input_splits,
            output_splits=output_splits,
        )
        return routed_input, num_tokens_per_expert, state

    def combine(self, routed_output: torch.Tensor, state: TorchDispatchState) -> torch.Tensor:
        routed_output = unpermute_from_grouped_gemm(routed_output, state.permutation)
        routed_output = self._combine_tokens(routed_output, state.input_splits, state.output_splits)
        return _scatter_routed_output(
            routed_output,
            num_tokens=state.num_tokens,
            token_indices_experts_sorted=state.token_indices_experts_sorted,
            scores_after_experts=state.scores_after_experts,
        )


class MXFP8TorchTokenDispatcher(TorchTokenDispatcher):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        token_group_alignment: int,
        group: ProcessGroup,
        kernel: ModuleType,
    ) -> None:
        super().__init__(num_experts, top_k, token_group_alignment, group)
        self.kernel = kernel

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        return self.kernel.all_to_all_dispatch(x, output_splits, input_splits, self.group)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        return self.kernel.all_to_all_combine(x, output_splits, input_splits, self.group)
