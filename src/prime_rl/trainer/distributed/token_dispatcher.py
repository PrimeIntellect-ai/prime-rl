from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.collectives import (
    all_to_all_single,
    all_to_all_single_equal,
    mxfp8_all_to_all_combine,
    mxfp8_all_to_all_dispatch,
)


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


DispatchState = TypeVar("DispatchState")


class TokenDispatcherBase(ABC, Generic[DispatchState]):
    def __init__(self, num_experts: int, token_group_alignment: int) -> None:
        self.num_experts = num_experts
        self.token_group_alignment = token_group_alignment

    @abstractmethod
    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, DispatchState]: ...

    @abstractmethod
    def combine(self, routed_output: torch.Tensor, state: DispatchState) -> torch.Tensor: ...

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
    input_splits: torch.Tensor
    output_splits: torch.Tensor


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


# Adapted from torchtitan/ops/scatter_add.py (BSD-3-Clause).
@torch.library.custom_op("prime_rl::deterministic_scatter_add", mutates_args=())
def _deterministic_scatter_add(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # Keep the cast inside the custom op so compilation preserves rounded inputs.
    dtype = torch.float32 if src.dtype in (torch.bfloat16, torch.float16) else src.dtype
    accumulation_out = out.to(dtype)
    accumulation_src = src.to(dtype)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        result = accumulation_out.scatter_add(0, index, accumulation_src)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return result.to(out.dtype)


@_deterministic_scatter_add.register_fake
def _deterministic_scatter_add_fake(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _deterministic_scatter_add_setup_context(ctx, inputs, output):
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _deterministic_scatter_add_backward(ctx, grad_output):
    (index,) = ctx.saved_tensors
    return grad_output, None, torch.gather(grad_output, 0, index)


_deterministic_scatter_add.register_autograd(
    _deterministic_scatter_add_backward, setup_context=_deterministic_scatter_add_setup_context
)


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
    return _deterministic_scatter_add(output, routed_indices, routed_output)


class LocalTokenDispatcher(TokenDispatcherBase[LocalDispatchState]):
    def __init__(self, num_experts: int, top_k: int, token_group_alignment: int) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.top_k = top_k

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


class TorchTokenDispatcher(TokenDispatcherBase[TorchDispatchState]):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        token_group_alignment: int,
        group: ProcessGroup,
    ) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.top_k = top_k
        self.group = group

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return all_to_all_single(x, output_splits, input_splits, self.group)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return all_to_all_single(x, output_splits, input_splits, self.group)

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
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single_equal(num_tokens_per_expert, self.group)
            input_splits, output_splits = torch.stack(
                (
                    num_tokens_per_expert.view(ep_degree, -1).sum(dim=1),
                    num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1),
                )
            ).cpu()  # Transfer to CPU now once, so a2a can read this without sync

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
    ) -> None:
        super().__init__(num_experts, top_k, token_group_alignment, group)

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return mxfp8_all_to_all_dispatch(x, output_splits, input_splits, self.group)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return mxfp8_all_to_all_combine(x, output_splits, input_splits, self.group)
