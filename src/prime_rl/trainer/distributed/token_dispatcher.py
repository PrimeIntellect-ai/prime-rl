from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.collectives import (
    all_to_all_single,
    all_to_all_single_async,
    all_to_all_single_equal,
    mxfp8_all_to_all_combine,
    mxfp8_all_to_all_dispatch,
    sync_all_to_all,
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
    def __init__(self, num_experts: int, token_group_alignment: int, token_chunk_size: int | None = None) -> None:
        self.num_experts = num_experts
        self.token_group_alignment = token_group_alignment
        self.token_chunk_size = token_chunk_size

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

    def _dispatch_issue(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ):
        return self.dispatch(x, top_scores, selected_experts_indices, score_before_experts=score_before_experts)

    def _dispatch_finalize(self, pending) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
        return pending

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        if self.token_chunk_size is None:
            routed_input, num_tokens_per_expert, state = self.dispatch(
                x,
                top_scores,
                selected_experts_indices,
                score_before_experts=score_before_experts,
            )
            routed_output = experts(routed_input, num_tokens_per_expert)
            return self.combine(routed_output, state)

        return self._run_chunked(x, top_scores, selected_experts_indices, experts, score_before_experts)

    def _chunk_ranges(self, num_tokens: int) -> list[tuple[int, int]]:
        chunk_size = self.token_chunk_size
        assert chunk_size is not None
        return [(start, min(start + chunk_size, num_tokens)) for start in range(0, num_tokens, chunk_size)]

    def _run_chunked(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        score_before_experts: bool,
    ) -> torch.Tensor:
        pending_chunks = (
            self._dispatch_issue(
                x[start:end],
                top_scores[start:end],
                selected_experts_indices[start:end],
                score_before_experts=score_before_experts,
            )
            for start, end in self._chunk_ranges(x.shape[0])
        )
        return self._run_pipeline(pending_chunks, experts)

    def _run_pipeline(self, pending_chunks, experts: ExpertFunction) -> torch.Tensor:
        def run_chunk(pending) -> torch.Tensor:
            routed_input, num_tokens_per_expert, state = self._dispatch_finalize(pending)
            routed_output = experts(routed_input, num_tokens_per_expert)
            return self.combine(routed_output, state)

        pending_chunks = iter(pending_chunks)
        pending = next(pending_chunks)
        routed_outputs: list[torch.Tensor] = []
        for next_pending in pending_chunks:
            routed_outputs.append(run_chunk(pending))
            pending = next_pending
        routed_outputs.append(run_chunk(pending))

        if len(routed_outputs) == 1:
            return routed_outputs[0]
        return torch.cat(routed_outputs, dim=0)

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


class LocalTokenDispatcher(TokenDispatcherBase[LocalDispatchState]):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        token_group_alignment: int,
        token_chunk_size: int | None = None,
    ) -> None:
        super().__init__(num_experts, token_group_alignment, token_chunk_size)
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


@dataclass
class _PendingTorchDispatch:
    routed_input: torch.Tensor
    transport_handle: object
    token_indices: torch.Tensor
    scores_after_experts: torch.Tensor | None
    num_tokens_per_expert_group: torch.Tensor
    input_splits: torch.Tensor
    output_splits: torch.Tensor
    num_tokens: int


class TorchTokenDispatcher(TokenDispatcherBase[TorchDispatchState]):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        token_group_alignment: int,
        group: ProcessGroup,
        token_chunk_size: int | None = None,
    ) -> None:
        super().__init__(num_experts, token_group_alignment, token_chunk_size)
        self.top_k = top_k
        self.group = group

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return all_to_all_single(x, output_splits, input_splits, self.group)

    def _dispatch_tokens_async(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        """Issue the dispatch transport without waiting for it. Paired with `_sync_dispatch_tokens`."""
        return all_to_all_single_async(x, output_splits, input_splits, self.group)

    def _sync_dispatch_tokens(self, transport_handle: object) -> None:
        sync_all_to_all(transport_handle)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return all_to_all_single(x, output_splits, input_splits, self.group)

    def _dispatch_issue(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> _PendingTorchDispatch:
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
            input_splits = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1)
            output_splits = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1)

        routed_input, transport_handle = self._dispatch_tokens_async(routed_input, output_splits, input_splits)
        return _PendingTorchDispatch(
            routed_input=routed_input,
            transport_handle=transport_handle,
            token_indices=token_indices,
            scores_after_experts=scores_after_experts,
            num_tokens_per_expert_group=num_tokens_per_expert_group,
            input_splits=input_splits,
            output_splits=output_splits,
            num_tokens=x.shape[0],
        )

    def _dispatch_finalize(
        self, pending: _PendingTorchDispatch
    ) -> tuple[torch.Tensor, torch.Tensor, TorchDispatchState]:
        self._sync_dispatch_tokens(pending.transport_handle)

        ep_degree = self.group.size()
        experts_per_rank = self.num_experts // ep_degree
        routed_input, num_tokens_per_expert, permutation = permute_for_grouped_gemm(
            pending.routed_input,
            pending.num_tokens_per_expert_group,
            experts_per_rank=experts_per_rank,
            num_ranks=ep_degree,
            alignment=self.token_group_alignment,
        )
        state = TorchDispatchState(
            num_tokens=pending.num_tokens,
            token_indices_experts_sorted=pending.token_indices,
            scores_after_experts=pending.scores_after_experts,
            permutation=permutation,
            input_splits=pending.input_splits,
            output_splits=pending.output_splits,
        )
        return routed_input, num_tokens_per_expert, state

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, TorchDispatchState]:
        pending = self._dispatch_issue(
            x, top_scores, selected_experts_indices, score_before_experts=score_before_experts
        )
        return self._dispatch_finalize(pending)

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
        token_chunk_size: int | None = None,
    ) -> None:
        super().__init__(num_experts, top_k, token_group_alignment, group, token_chunk_size)

    def _dispatch_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return mxfp8_all_to_all_dispatch(x, output_splits, input_splits, self.group)

    def _dispatch_tokens_async(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        return self._dispatch_tokens(x, output_splits, input_splits), None

    def _sync_dispatch_tokens(self, transport_handle: object) -> None:
        del transport_handle

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
    ) -> torch.Tensor:
        return mxfp8_all_to_all_combine(x, output_splits, input_splits, self.group)
