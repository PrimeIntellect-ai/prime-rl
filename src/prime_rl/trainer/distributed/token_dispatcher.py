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
from prime_rl.trainer.distributed.segment_sum import gather_rows, segment_sum


class ExpertFunction(Protocol):
    def __call__(self, x: torch.Tensor, *routing: torch.Tensor) -> torch.Tensor: ...


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


class FusedTokenDispatcher:
    """Passes undispatched tokens and their routing to an expert compute that dispatches and combines them itself."""

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        if score_before_experts:
            raise ValueError(
                "Fused dispatch applies router weights at combine time and requires score_before_experts=False."
            )
        return experts(x, top_scores, selected_experts_indices)

    def synchronize(self) -> None:
        return None


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
    route_indices_experts_sorted: torch.Tensor
    scores_after_experts: torch.Tensor | None
    permutation: PermutationState


@dataclass(frozen=True)
class TorchDispatchState(LocalDispatchState):
    input_splits: torch.Tensor
    output_splits: torch.Tensor
    num_routed_tokens: int


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
    return _GatherPermutedRows.apply(x, permuted_indices), num_tokens_per_expert, state


class _GatherPermutedRows(torch.autograd.Function):
    """Computes `x[permuted_indices]` for indices that permute the real rows and fill padding slots with the zero row.

    Each real row appears exactly once, so the gather's backward is a gather by the inverse permutation:
    every real row reads its gradient from its one slot, and the constant zero row gets none.
    Autograd's `index` backward cannot assume this and scatter-adds, serializing on the zero row's repeats.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, permuted_indices: torch.Tensor) -> torch.Tensor:
        inverse_permutation = permuted_indices.new_empty(x.shape[0])
        inverse_permutation[permuted_indices] = torch.arange(
            permuted_indices.numel(), device=permuted_indices.device, dtype=permuted_indices.dtype
        )
        ctx.save_for_backward(inverse_permutation[:-1])
        return x[permuted_indices]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (real_row_inverse_permutation,) = ctx.saved_tensors
        grad_x = torch.cat(
            (grad_output[real_row_inverse_permutation], grad_output.new_zeros((1, grad_output.shape[-1])))
        )
        return grad_x, None


def unpermute_from_grouped_gemm(x: torch.Tensor, state: PermutationState) -> torch.Tensor:
    output = x.new_empty(state.input_shape)
    output[state.permuted_indices] = x
    return output[:-1]


@torch.library.custom_op("prime_rl::stable_argsort", mutates_args=())
def stable_argsort(x: torch.Tensor) -> torch.Tensor:
    """Stable argsort that keeps torch.compile from recompiling when the token count changes.

    Left inline, torch.compile recompiles the whole decoder layer whenever a rank's routed-token count crosses 32767,
    the int16 bound inductor's sort lowering guards on:
    https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/lowering.py#L7719-L7726
    Wrapping the sort in a custom op prevents that. Performance should not suffer: above a routed-token count of
    512, torch.compile already runs this same ATen sort:
    https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/ir.py#L3085-L3095
    It needs no backward: it returns integer indices, which carry no gradient.
    """
    return torch.argsort(x, stable=True)


@stable_argsort.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.int64)


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
    route_indices_experts_sorted = stable_argsort(flattened_experts)
    top_scores_experts_sorted = top_scores.reshape(-1)[route_indices_experts_sorted]
    # A route slot is token_index * top_k + expert_slot (position among selected experts).
    # // top_k recovers its token; keep the full slot index to restore outputs in combine.
    routed_input = x[route_indices_experts_sorted // top_k]
    return routed_input, route_indices_experts_sorted, top_scores_experts_sorted, num_tokens_per_expert


def _combine_routed_output(
    routed_output: torch.Tensor,
    *,
    num_tokens: int,
    route_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
) -> torch.Tensor:
    output_dtype = routed_output.dtype
    if scores_after_experts is not None:
        routed_output = routed_output * scores_after_experts.float().reshape(-1, 1)

    if num_tokens == 0:
        return routed_output.to(output_dtype)

    dim = routed_output.shape[-1]
    # There is one output row per selected expert per token.
    top_k = routed_output.shape[0] // num_tokens
    # Undo expert-ID sorting so each token's routes can be summed without atomic additions.
    # Each row is written once, so the buffer needs no initialization.
    output = torch.empty_like(routed_output)
    # Copy source row i to its original slot, route_indices_experts_sorted[i].
    output[route_indices_experts_sorted] = routed_output
    accumulation_dtype = torch.float32 if output.dtype in (torch.bfloat16, torch.float16) else output.dtype
    return output.view(num_tokens, top_k, dim).sum(dim=1, dtype=accumulation_dtype).to(output_dtype)


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
        routed_input, route_indices, sorted_scores, num_tokens_per_expert = _local_reorder(
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
            route_indices_experts_sorted=route_indices,
            scores_after_experts=scores_after_experts,
            permutation=permutation,
        )
        return routed_input, num_tokens_per_expert, state

    def combine(self, routed_output: torch.Tensor, state: LocalDispatchState) -> torch.Tensor:
        routed_output = unpermute_from_grouped_gemm(routed_output, state.permutation)
        return _combine_routed_output(
            routed_output,
            num_tokens=state.num_tokens,
            route_indices_experts_sorted=state.route_indices_experts_sorted,
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
        output_size: int,
    ) -> torch.Tensor:
        return all_to_all_single(x, output_splits, input_splits, self.group, output_size)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        return all_to_all_single(x, output_splits, input_splits, self.group, output_size)

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, TorchDispatchState]:
        routed_input, route_indices, sorted_scores, num_tokens_per_expert = _local_reorder(
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

        num_routed_tokens = routed_input.shape[0]
        num_received_tokens = int(output_splits.sum())
        routed_input = self._dispatch_tokens(routed_input, output_splits, input_splits, num_received_tokens)
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
            route_indices_experts_sorted=route_indices,
            scores_after_experts=scores_after_experts,
            permutation=permutation,
            input_splits=input_splits,
            output_splits=output_splits,
            num_routed_tokens=num_routed_tokens,
        )
        return routed_input, num_tokens_per_expert, state

    def combine(self, routed_output: torch.Tensor, state: TorchDispatchState) -> torch.Tensor:
        routed_output = unpermute_from_grouped_gemm(routed_output, state.permutation)
        routed_output = self._combine_tokens(
            routed_output, state.input_splits, state.output_splits, state.num_routed_tokens
        )
        return _combine_routed_output(
            routed_output,
            num_tokens=state.num_tokens,
            route_indices_experts_sorted=state.route_indices_experts_sorted,
            scores_after_experts=state.scores_after_experts,
        )


@dataclass(frozen=True)
class SegmentIndex:
    """CSR over the rows of a gathered tensor: segment s owns entries seg_ptr[s]:seg_ptr[s + 1] of `rows`."""

    rows: torch.Tensor
    segment_of_row: torch.Tensor
    seg_ptr: torch.Tensor


@dataclass(frozen=True)
class PerRankDispatchState:
    input_splits: torch.Tensor
    output_splits: torch.Tensor
    num_pairs: int
    home: SegmentIndex
    expert: SegmentIndex
    expert_weights: torch.Tensor | None


def _segment_index(rows: torch.Tensor, segment_of_row: torch.Tensor, num_segments: int) -> SegmentIndex:
    counts = segment_of_row.new_zeros(num_segments).scatter_add_(0, segment_of_row, torch.ones_like(segment_of_row))
    seg_ptr = torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))
    return SegmentIndex(rows=rows, segment_of_row=segment_of_row, seg_ptr=seg_ptr)


class PerRankTorchTokenDispatcher(TokenDispatcherBase[PerRankDispatchState]):
    """Expert-parallel dispatch that sends each token once to every rank holding one of its experts, with its
    routing, instead of once per selected expert (``token_copies = "per_rank"``). The receiving rank expands the
    token to its experts, applies the routing weights and sums the token's expert outputs before the combine, which
    therefore returns one row per token and rank as well. At top-8 of 256 experts on 8 ranks a token reaches 5.30
    ranks on average, so both all-to-alls move ~34% fewer bytes. The backward pass does not need the combine output
    (the weights are applied before it), so activation checkpointing does not replay the combine."""

    def __init__(self, num_experts: int, top_k: int, token_group_alignment: int, group: ProcessGroup) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.top_k = top_k
        self.group = group

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, PerRankDispatchState]:
        ranks = self.group.size()
        experts_per_rank = self.num_experts // ranks
        top_k, alignment = self.top_k, self.token_group_alignment
        num_tokens = x.shape[0]
        with torch.no_grad():
            destination = selected_experts_indices // experts_per_rank
            # [tokens, ranks]: the token has an expert on the rank, so it is sent there once.
            sends = torch.zeros((num_tokens, ranks), dtype=torch.bool, device=x.device)
            sends.scatter_(1, destination, True)
            pairs_per_rank = sends.sum(dim=0)
            routes_per_expert = torch.histc(
                selected_experts_indices.reshape(-1).float(), bins=self.num_experts, min=0, max=self.num_experts
            ).to(torch.int64)
            # Per destination rank: the number of tokens sent there, then its routes per local expert.
            counts = torch.cat((pairs_per_rank.unsqueeze(1), routes_per_expert.view(ranks, experts_per_rank)), dim=1)
            received = all_to_all_single_equal(counts.reshape(-1), self.group).view(ranks, 1 + experts_per_rank)
            routes_per_local_expert = received[:, 1:].sum(dim=0)
            aligned_counts = ((routes_per_local_expert + alignment - 1) // alignment).clamp(min=1) * alignment
            # One transfer: the split sizes, the number of routes and the size of the aligned expert input.
            host = torch.cat(
                (
                    pairs_per_rank,
                    received[:, 0],
                    routes_per_local_expert.sum().view(1),
                    aligned_counts.sum().view(1),
                )
            ).cpu()
        input_splits, output_splits = host[:ranks], host[ranks : 2 * ranks]
        num_pairs, num_received = int(input_splits.sum()), int(output_splits.sum())
        num_routes, aligned_length = int(host[-2]), int(host[-1])

        with torch.no_grad():
            # Send buffer: rank-major, tokens in order within a rank.
            pair_offset = torch.cumsum(pairs_per_rank, dim=0) - pairs_per_rank
            position = torch.cumsum(sends.to(torch.int64), dim=0) - 1 + pair_offset
            token_rank = torch.arange(num_tokens * ranks, device=x.device).view(num_tokens, ranks)
            token_rank_of_pair = torch.full((num_pairs + 1,), -1, dtype=torch.int64, device=x.device)
            token_rank_of_pair[torch.where(sends, position, num_pairs)] = token_rank
            token_rank_of_pair = token_rank_of_pair[:num_pairs]
            token_of_pair = token_rank_of_pair // ranks
            owned = destination[token_of_pair] == (token_rank_of_pair % ranks).unsqueeze(1)
            # The selected experts of each sent token; -1 marks the ones on other ranks.
            local_experts = torch.where(owned, selected_experts_indices[token_of_pair] % experts_per_rank, -1)
            # Per token, the pairs that carry it (token-major).
            home_rows = torch.argsort(token_rank_of_pair)
            home = _segment_index(home_rows, token_of_pair[home_rows], num_tokens)

        tokens_send = gather_rows(x, token_of_pair, home.rows, home.segment_of_row, home.seg_ptr)
        scores_send = gather_rows(top_scores, token_of_pair, home.rows, home.segment_of_row, home.seg_ptr)
        scores_send = scores_send * owned.to(scores_send.dtype)
        received_tokens = all_to_all_single(tokens_send, output_splits, input_splits, self.group, num_received)
        received_scores = all_to_all_single(scores_send, output_splits, input_splits, self.group, num_received)
        with torch.no_grad():
            received_experts = all_to_all_single(
                local_experts.to(torch.int32), output_splits, input_splits, self.group, num_received
            ).reshape(-1)

            # Expert input: each received token once per selected local expert, grouped by expert in receive order;
            # each group starts at a multiple of the alignment.
            valid = received_experts >= 0
            slot_order = torch.argsort(torch.where(valid, received_experts, experts_per_rank), stable=True)
            routes = slot_order[:num_routes]
            route_experts = received_experts[routes].to(torch.int64)
            route_start = torch.cumsum(routes_per_local_expert, dim=0) - routes_per_local_expert
            aligned_start = torch.cumsum(aligned_counts, dim=0) - aligned_counts
            rank_in_expert = torch.arange(num_routes, device=x.device) - route_start[route_experts]
            aligned_position = aligned_start[route_experts] + rank_in_expert
            token_of_aligned_position = torch.full((aligned_length,), -1, dtype=torch.int64, device=x.device)
            token_of_aligned_position[aligned_position] = routes // top_k
            # Per received token, the aligned rows of its experts (token-major).
            by_token = torch.argsort(routes)
            expert = _segment_index(aligned_position[by_token], routes[by_token] // top_k, num_received)
            expert_slots = routes[by_token]

        routed_input = gather_rows(
            received_tokens, token_of_aligned_position, expert.rows, expert.segment_of_row, expert.seg_ptr
        )
        expert_weights = received_scores.reshape(-1)[expert_slots]
        if score_before_experts:
            aligned_weights = expert_weights.new_zeros(aligned_length).index_put((expert.rows,), expert_weights)
            routed_input = (routed_input.float() * aligned_weights.unsqueeze(1)).to(x.dtype)
            expert_weights = None
        state = PerRankDispatchState(
            input_splits=input_splits,
            output_splits=output_splits,
            num_pairs=num_pairs,
            home=home,
            expert=expert,
            expert_weights=expert_weights,
        )
        return routed_input, aligned_counts.to(torch.int32), state

    def combine(self, routed_output: torch.Tensor, state: PerRankDispatchState) -> torch.Tensor:
        expert, home = state.expert, state.home
        # Per received token: the weighted sum of its outputs from this rank's experts.
        partial = segment_sum(routed_output, expert.rows, expert.segment_of_row, expert.seg_ptr, state.expert_weights)
        partial = all_to_all_single(partial, state.input_splits, state.output_splits, self.group, state.num_pairs)
        return segment_sum(partial, home.rows, home.segment_of_row, home.seg_ptr)


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
        output_size: int,
    ) -> torch.Tensor:
        return mxfp8_all_to_all_dispatch(x, output_splits, input_splits, self.group, output_size)

    def _combine_tokens(
        self,
        x: torch.Tensor,
        output_splits: torch.Tensor,
        input_splits: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        return mxfp8_all_to_all_combine(x, output_splits, input_splits, self.group, output_size)
