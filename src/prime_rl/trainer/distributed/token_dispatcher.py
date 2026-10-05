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


class OverlappedTorchTokenDispatcher(TorchTokenDispatcher):
    """Torch all-to-all dispatch that hides communication behind the routed experts.

    The tokens are split into `chunks` contiguous pieces, each dispatched, computed and combined
    on its own. Every token all-to-all is issued asynchronously, so while one chunk's experts run
    the next chunk's dispatch and the previous chunk's combine are in flight. Only the first
    dispatch and the last combine stay exposed. All chunks' token counts are exchanged and moved to
    the host up front, in one sync: a per-chunk sync would wait on the all-to-alls issued before it
    and serialize them again. Each collective output is waited on at its first use, so the
    program order alone decides what overlaps.
    """

    def __init__(self, num_experts: int, top_k: int, token_group_alignment: int, group: ProcessGroup, chunks: int):
        super().__init__(num_experts, top_k, token_group_alignment, group)
        self.chunks = chunks

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        import torch.distributed._functional_collectives as funcol

        ep_degree = self.group.size()
        experts_per_rank = self.num_experts // ep_degree
        pieces = []
        for xc, sc, ic in zip(
            x.chunk(self.chunks), top_scores.chunk(self.chunks), selected_experts_indices.chunk(self.chunks)
        ):
            routed_input, token_indices, sorted_scores, num_tokens_per_expert = _local_reorder(
                xc, sc, ic, num_experts=self.num_experts, top_k=self.top_k
            )
            scores_after_experts = None
            if score_before_experts:
                routed_input = (routed_input.float() * sorted_scores.reshape(-1, 1)).to(x.dtype)
            else:
                scores_after_experts = sorted_scores
            pieces.append((xc.shape[0], routed_input, token_indices, scores_after_experts, num_tokens_per_expert))

        with torch.no_grad():
            counts = torch.stack([piece[-1] for piece in pieces])  # (chunks, num_experts)
            counts_group = all_to_all_single_equal(counts.t().contiguous(), self.group).t()
            splits = torch.stack(
                (
                    counts.view(len(pieces), ep_degree, -1).sum(-1),
                    counts_group.reshape(len(pieces), ep_degree, -1).sum(-1),
                )
            ).cpu()
        input_splits, output_splits = splits[0].tolist(), splits[1].tolist()

        def dispatch(c: int) -> torch.Tensor:
            return funcol.all_to_all_single_autograd(pieces[c][1], output_splits[c], input_splits[c], self.group)

        # Chunk c + 1 is dispatched right after chunk c's experts are issued, so its all-to-all runs
        # on the communication stream while those experts run on the compute stream.
        dispatched = dispatch(0)
        combined = []
        for c, (num_tokens, _, token_indices, scores_after_experts, _) in enumerate(pieces):
            routed_input, num_tokens_per_expert, permutation = permute_for_grouped_gemm(
                dispatched,
                counts_group[c].contiguous(),
                experts_per_rank=experts_per_rank,
                num_ranks=ep_degree,
                alignment=self.token_group_alignment,
            )
            expert_output = experts(routed_input, num_tokens_per_expert)
            if c + 1 < len(pieces):
                dispatched = dispatch(c + 1)
            routed_output = unpermute_from_grouped_gemm(expert_output, permutation)
            combined.append(
                (
                    funcol.all_to_all_single_autograd(routed_output, input_splits[c], output_splits[c], self.group),
                    num_tokens,
                    token_indices,
                    scores_after_experts,
                )
            )
        return torch.cat(
            [
                _scatter_routed_output(
                    routed_output,
                    num_tokens=num_tokens,
                    token_indices_experts_sorted=token_indices,
                    scores_after_experts=scores_after_experts,
                )
                for routed_output, num_tokens, token_indices, scores_after_experts in combined
            ]
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
