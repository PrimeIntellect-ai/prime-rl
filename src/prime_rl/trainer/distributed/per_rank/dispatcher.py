"""Expert-parallel dispatch with one token copy per rank (``model.moe.dispatch.token_copies = "per_rank"``)."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.collectives import all_to_all_counts, all_to_all_single
from prime_rl.trainer.distributed.per_rank.ops import SlotMap, scatter_slots, sum_slots
from prime_rl.trainer.distributed.token_dispatcher import TokenDispatcherBase, stable_argsort


@dataclass(frozen=True)
class PerRankDispatchState:
    input_splits: Tensor
    output_splits: Tensor
    home: SlotMap
    expert: SlotMap
    expert_weights: Tensor | None
    host_num_tokens_per_expert: Tensor


def _send_buffer_rows(sends: Tensor) -> tuple[Tensor, Tensor]:
    """The tokens per destination rank (``[ranks]``) and every token's row in the send buffer per destination rank, -1
    where it is not sent (int32, ``[tokens, ranks]``). The send buffer is rank-major, with the tokens in order within a
    rank."""
    pairs = sends.sum(dim=0)
    first_row = torch.cumsum(pairs, dim=0) - pairs
    return pairs, torch.where(sends, torch.cumsum(sends, dim=0) - 1 + first_row, -1).to(torch.int32)


def _rank_by_key(keys: Tensor, num_keys: int) -> tuple[Tensor, Tensor]:
    """The count of every key (``[num_keys]``) and each entry's rank among the entries with its key, in order."""
    # histc has a static output size; bincount's depends on the largest key.
    counts = torch.histc(keys.float(), bins=num_keys, min=0, max=num_keys).to(torch.int64)
    order = stable_argsort(keys.to(torch.int32))
    ranks = torch.empty_like(keys)
    ranks[order] = torch.arange(keys.numel(), device=keys.device) - (torch.cumsum(counts, dim=0) - counts)[keys[order]]
    return counts, ranks


class PerRankTokenDispatcher(TokenDispatcherBase[PerRankDispatchState]):
    """Sends each token once to every EP rank that holds one of its selected experts, with its routing, instead of
    once per selected expert. The receiving rank lays out the expert input as `TorchTokenDispatcher` does (aligned
    groups in expert order, each expert's routes by source rank, then token order), applies the routing weights and
    sums each token's expert outputs before the combine, which therefore also returns one row per token and rank, in
    the experts' output dtype. The backward does not need the combine's output, so under torch.compile full activation
    checkpointing does not replay the combine.

    Two slot maps move every row (`scatter_slots`, `sum_slots`). Home map: token t's slot r owns its row in the send
    buffer to rank r, so the send buffer is a scatter of the tokens and the final combine a sum over their slots.
    Expert map: received token q's slot k owns the expert-input row of its k-th selected expert if that expert is
    local, so the routing weights are read by slot. Every route travels as ``rank * num_experts + expert`` (int32),
    with ``rank`` its position among the sending rank's routes to that expert, so the receiving rank places it without
    sorting: its expert's first row, plus the routes from lower source ranks, plus ``rank``."""

    def __init__(self, num_experts: int, token_group_alignment: int, group: ProcessGroup) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.group = group

    def dispatch(
        self,
        x: Tensor,
        top_scores: Tensor,
        selected_experts_indices: Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[Tensor, Tensor, PerRankDispatchState]:
        ranks, num_experts = self.group.size(), self.num_experts
        experts_per_rank, alignment = num_experts // ranks, self.token_group_alignment
        with torch.no_grad():
            sends = torch.zeros((x.shape[0], ranks), dtype=torch.bool, device=x.device)
            sends.scatter_(1, selected_experts_indices // experts_per_rank, True)
            pairs, home_rows = _send_buffer_rows(sends)
            routes_per_expert, route_ranks = _rank_by_key(selected_experts_indices.reshape(-1), num_experts)
            routes = route_ranks.view_as(selected_experts_indices) * num_experts + selected_experts_indices
            # Per destination rank: the tokens sent there, then the routes per local expert.
            counts = torch.cat((pairs.unsqueeze(1), routes_per_expert.view(ranks, experts_per_rank)), dim=1)
            received = all_to_all_counts(counts.view(-1), self.group).view(counts.shape)
            routes_per_local_expert = received[:, 1:].sum(dim=0)
            rows_per_expert = ((routes_per_local_expert + alignment - 1) // alignment).clamp(min=1) * alignment
            # One transfer to the host: the split sizes and the expert input's rows per local expert.
            host = torch.cat((pairs, received[:, 0], rows_per_expert)).cpu()
        input_splits, output_splits = host[:ranks], host[ranks : 2 * ranks]
        host_num_tokens_per_expert = host[2 * ranks :]
        num_received = int(output_splits.sum())

        home = SlotMap(home_rows, int(input_splits.sum()))
        tokens = all_to_all_single(scatter_slots(x, home), output_splits, input_splits, self.group, num_received)
        scores = all_to_all_single(
            scatter_slots(top_scores, home), output_splits, input_splits, self.group, num_received
        )
        with torch.no_grad():
            routes = scatter_slots(routes.to(torch.int32), home)
            routes = all_to_all_single(routes, output_splits, input_splits, self.group, num_received)
            expert = self._expert_slots(routes, received, rows_per_expert, int(host_num_tokens_per_expert.sum()))
        routed_input = scatter_slots(tokens, expert, scores if score_before_experts else None)
        state = PerRankDispatchState(
            input_splits=input_splits,
            output_splits=output_splits,
            home=home,
            expert=expert,
            expert_weights=None if score_before_experts else scores,
            host_num_tokens_per_expert=host_num_tokens_per_expert,
        )
        return routed_input, rows_per_expert.to(torch.int32), state

    def _expert_slots(self, routes: Tensor, received: Tensor, rows_per_expert: Tensor, num_rows: int) -> SlotMap:
        """Every received route's row in the expert input: its expert's first row, plus the routes to that expert
        from lower source ranks, plus its rank among its source's routes to the expert. The rows of a group after its
        routes (alignment) are padding."""
        experts_per_rank, alignment = self.num_experts // self.group.size(), self.token_group_alignment
        routes_from, device = received[:, 1:], routes.device
        group_start = torch.cumsum(rows_per_expert, dim=0) - rows_per_expert
        first_row = group_start + torch.cumsum(routes_from, dim=0) - routes_from
        source = torch.searchsorted(
            torch.cumsum(received[:, 0], dim=0), torch.arange(routes.shape[0], device=device), right=True
        )
        expert = routes % self.num_experts - self.group.rank() * experts_per_rank
        local = (expert >= 0) & (expert < experts_per_rank)
        rows = first_row[source.unsqueeze(1), expert.clamp(0, experts_per_rank - 1)] + routes // self.num_experts
        offset = routes_from.sum(dim=0).unsqueeze(1) + torch.arange(alignment, device=device)
        padding = torch.where(offset < rows_per_expert.unsqueeze(1), group_start.unsqueeze(1) + offset, -1)
        return SlotMap(torch.where(local, rows, -1).to(torch.int32), num_rows, padding.view(-1).to(torch.int32))

    def host_num_tokens_per_expert(self, state: PerRankDispatchState) -> Tensor:
        return state.host_num_tokens_per_expert

    def combine(self, routed_output: Tensor, state: PerRankDispatchState) -> Tensor:
        # Per received token: the weighted sum of its outputs from this rank's experts.
        partial = sum_slots(routed_output, state.expert, state.expert_weights)
        partial = all_to_all_single(partial, state.input_splits, state.output_splits, self.group, state.home.num_rows)
        return sum_slots(partial, state.home)
