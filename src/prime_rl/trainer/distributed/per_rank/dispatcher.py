"""The fused per-rank dispatch (``model.moe.dispatch.fused``): `PerRankTorchTokenDispatcher` with its indices built
without sorting the routes on the receiving rank and its rows moved by the slot-map ops, one pass per direction."""

from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.trainer.distributed.collectives import all_to_all_single, all_to_all_single_equal
from prime_rl.trainer.distributed.per_rank.ops import SlotMap, scatter_slots, sum_slots
from prime_rl.trainer.distributed.token_dispatcher import PerRankTorchTokenDispatcher


@dataclass(frozen=True)
class FusedDispatchState:
    input_splits: Tensor
    output_splits: Tensor
    home: SlotMap
    expert: SlotMap
    expert_weights: Tensor | None


def _send_buffer_rows(sends: Tensor) -> tuple[Tensor, Tensor]:
    """The pairs per destination rank (``[ranks]``) and every token's row in the send buffer per destination rank, -1
    where it is not sent (int32, ``[tokens, ranks]``). The send buffer is rank-major, with the tokens in order within a
    rank."""
    pairs = sends.sum(dim=0)
    first_row = torch.cumsum(pairs, dim=0) - pairs
    return pairs, torch.where(sends, torch.cumsum(sends, dim=0) - 1 + first_row, -1).to(torch.int32)


def _rank_by_key(keys: Tensor, num_keys: int) -> tuple[Tensor, Tensor]:
    """The count of every key (``[num_keys]``) and each entry's rank among the entries with its key, in order (a stable
    sort)."""
    # histc has a static output size; bincount's depends on the largest key.
    counts = torch.histc(keys.float(), bins=num_keys, min=0, max=num_keys).to(torch.int64)
    order = torch.argsort(keys.to(torch.int32), stable=True)
    ranks = torch.empty_like(keys)
    ranks[order] = torch.arange(keys.numel(), device=keys.device) - (torch.cumsum(counts, dim=0) - counts)[keys[order]]
    return counts, ranks


class FusedPerRankTokenDispatcher(PerRankTorchTokenDispatcher):
    """``token_copies = "per_rank"`` with the slot-map kernels. Home slots: token t's slot r owns its row in the send
    buffer to rank r, so the send buffer is a scatter of the tokens and the final combine a sum over their slots.
    Expert slots: received token q's slot k owns the expert-input row of its k-th selected expert if that expert is
    local, so the routing weights are read by slot. Every route travels as ``rank * num_experts + expert`` (int32), with
    its rank among the sending rank's routes to the same expert, in token order; the receiving rank places it in its
    expert's group after the routes from lower ranks, the same order as the reference `PerRankTorchTokenDispatcher`."""

    def dispatch(
        self,
        x: Tensor,
        top_scores: Tensor,
        selected_experts_indices: Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[Tensor, Tensor, FusedDispatchState]:
        ranks, num_experts = self.group.size(), self.num_experts
        experts_per_rank, alignment = num_experts // ranks, self.token_group_alignment
        num_tokens = x.shape[0]
        with torch.no_grad():
            sends = torch.zeros((num_tokens, ranks), dtype=torch.bool, device=x.device)
            sends.scatter_(1, selected_experts_indices // experts_per_rank, True)
            pairs, home_rows = _send_buffer_rows(sends)
            routes_per_expert, route_ranks = _rank_by_key(selected_experts_indices.view(-1), num_experts)
            travelling_routes = route_ranks.view_as(selected_experts_indices) * num_experts + selected_experts_indices
            # Per destination rank: the pairs sent there, then the routes per local expert.
            counts = torch.cat((pairs.unsqueeze(1), routes_per_expert.view(ranks, experts_per_rank)), dim=1)
            received = all_to_all_single_equal(counts.reshape(-1), self.group).view(counts.shape)
            routes_per_local_expert = received[:, 1:].sum(dim=0)
            aligned_counts = ((routes_per_local_expert + alignment - 1) // alignment).clamp(min=1) * alignment
            # One transfer: the split sizes and the size of the aligned expert input.
            host = torch.cat((pairs, received[:, 0], aligned_counts.sum().view(1))).cpu()
        input_splits, output_splits = host[:ranks], host[ranks : 2 * ranks]
        num_received = int(output_splits.sum())

        home = SlotMap(home_rows, int(input_splits.sum()))
        tokens_send = scatter_slots(x, home)
        scores_send = scatter_slots(top_scores, home)
        received_tokens = all_to_all_single(tokens_send, output_splits, input_splits, self.group, num_received)
        received_scores = all_to_all_single(scores_send, output_splits, input_splits, self.group, num_received)
        with torch.no_grad():
            routes_send = scatter_slots(travelling_routes.to(torch.int32), home)
            received_routes = all_to_all_single(routes_send, output_splits, input_splits, self.group, num_received)
            expert = self._expert_slots(received_routes, received, aligned_counts, int(host[-1]))

        routed_input = scatter_slots(received_tokens, expert, received_scores if score_before_experts else None)
        state = FusedDispatchState(
            input_splits=input_splits,
            output_splits=output_splits,
            home=home,
            expert=expert,
            expert_weights=None if score_before_experts else received_scores,
        )
        return routed_input, aligned_counts.to(torch.int32), state

    def _expert_slots(
        self, received_routes: Tensor, received_counts: Tensor, aligned_counts: Tensor, aligned_length: int
    ) -> SlotMap:
        """Every received route's row in the expert input: its expert's first row, plus the routes to that expert
        from lower source ranks, plus its rank among its source's routes to the expert. The groups are aligned; the
        rows after each group's routes are padding."""
        experts_per_rank, alignment = self.num_experts // self.group.size(), self.token_group_alignment
        routes_from, device = received_counts[:, 1:], received_routes.device
        group_start = torch.cumsum(aligned_counts, dim=0) - aligned_counts
        first_row = group_start + torch.cumsum(routes_from, dim=0) - routes_from
        source = torch.searchsorted(
            torch.cumsum(received_counts[:, 0], dim=0),
            torch.arange(received_routes.shape[0], device=device),
            right=True,
        )
        expert = received_routes % self.num_experts - self.group.rank() * experts_per_rank
        local = (expert >= 0) & (expert < experts_per_rank)
        rank_in_source = received_routes // self.num_experts
        rows = first_row[source.unsqueeze(1), expert.clamp(0, experts_per_rank - 1)] + rank_in_source
        offset = routes_from.sum(dim=0).unsqueeze(1) + torch.arange(alignment, device=device)
        padding = torch.where(offset < aligned_counts.unsqueeze(1), group_start.unsqueeze(1) + offset, -1)
        index = torch.where(local, rows, -1).to(torch.int32)
        return SlotMap(index, aligned_length, padding.view(-1).to(torch.int32))

    def combine(self, routed_output: Tensor, state: FusedDispatchState) -> Tensor:
        # Per received token: the weighted sum of its outputs from this rank's experts.
        partial = sum_slots(routed_output, state.expert, state.expert_weights)
        partial = all_to_all_single(partial, state.input_splits, state.output_splits, self.group, state.home.num_rows)
        return sum_slots(partial, state.home)
