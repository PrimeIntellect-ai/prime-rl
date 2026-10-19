import torch
import torch.nn.functional as F

from prime_rl.trainer.distributed.token_dispatcher import TokenDispatcher
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
    record_moe_routing_statistics,
)


class PrimeRouter(TokenChoiceTopKRouter):
    """Sigmoid-scored top-k router with a selection bias and ``num_null_experts`` zero-output experts next to the
    routed ones (AdaMoE). Null experts are scored, selected and balanced like the others, so a token that selects
    them uses fewer routed experts. Returns the scores of all experts too, for the sequence-wise balance loss."""

    def __init__(
        self, *, dim: int, num_experts: int, num_null_experts: int, top_k: int, route_norm: bool, route_scale: float
    ):
        super().__init__(
            dim=dim,
            num_experts=num_experts + num_null_experts,
            top_k=top_k,
            score_func="sigmoid",
            route_norm=route_norm,
            route_scale=route_scale,
            selection_bias=True,
        )
        self.num_null_experts = num_null_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like ``TokenChoiceTopKRouter.forward`` with sigmoid scores; also returns the ``(bs*slen, num_experts)``
        scores of all experts."""
        if self.fp32_gate:
            gate_bias = self.gate.bias.float() if self.gate.bias is not None else None
            logits = F.linear(x.float(), self.gate.weight.float(), gate_bias)
        else:
            logits = self.gate(x)
        scores = torch.sigmoid(logits.float())
        # The selection bias only affects which experts are selected, not their scores.
        _, selected_experts_indices = torch.topk(
            scores + self.selection_bias, k=self.top_k, dim=1, sorted=self.topk_sorted
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        with torch.no_grad():
            routing_confidence_sum = (top_scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).sum()

        if self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1).float(), bins=self.num_experts, min=0, max=self.num_experts
        ).to(torch.int64)
        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum, scores


def sequence_balance(scores: torch.Tensor, selected_experts: torch.Tensor, num_rows: int) -> torch.Tensor:
    """DeepSeek-V3's sequence-wise balance loss without its coefficient, per row: ``sum_i f_i P_i`` with
    ``f_i`` the row's share of expert ``i``'s selections times ``num_experts / top_k`` (1 when balanced) and
    ``P_i`` its mean score normalized over all experts. 1 for a balanced row, up to ``num_experts / top_k``.

    Args:
        scores: ``(num_rows * slen, num_experts)`` scores of all experts.
        selected_experts: ``(num_rows * slen, top_k)`` selected experts.
    """
    num_experts = scores.shape[1]
    selections = selected_experts.view(num_rows, -1)
    counts = torch.zeros(num_rows, num_experts, device=scores.device).scatter_add_(
        1, selections, torch.ones_like(selections, dtype=torch.float32)
    )
    f = counts * (num_experts / selections.shape[1])
    p = (scores / scores.sum(dim=-1, keepdim=True)).view(num_rows, -1, num_experts).mean(dim=1)
    return (f * p).sum(dim=-1)


class PrimeMoE(MoE):
    """The shared token-choice MoE runtime with a ``PrimeRouter``. With null experts, tokens use different numbers of
    routed experts: only the (token, routed expert) pairs are dispatched, one per row. Also computes the sequence-wise
    balance for its loss (``balance_rows``) and counts the routed experts per token (``routed_k_counts``)."""

    def __init__(self, *, router: PrimeRouter, experts: GroupedExperts, shared_expert: FeedForward | None) -> None:
        super().__init__(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            score_before_experts=False,
            load_balance_coeff=1e-3,
        )
        self.variable_k = router.num_null_experts > 0
        self.set_token_dispatcher(self.token_dispatcher)
        # Usage of every expert, null experts included.
        self.register_buffer(
            "tokens_per_expert", torch.zeros(router.num_experts, dtype=torch.float32), persistent=False
        )
        # Training tokens per number of routed experts used, for variable-k routing.
        self.register_buffer(
            "routed_k_counts",
            torch.zeros(router.top_k + 1, dtype=torch.float32) if self.variable_k else None,
            persistent=False,
        )
        # Sequence-wise balance (DeepSeek-V3 eq. 17-20): with balance_rows set by the training loop, which knows
        # how many packed rows a sample holds, every forward with grad computes the per-row balance for its loss.
        self.balance_rows: int | None = None
        self.seq_balance: torch.Tensor | None = None

    def set_token_dispatcher(self, token_dispatcher: TokenDispatcher) -> None:
        # Variable-k routing dispatches one (token, expert) pair per row.
        token_dispatcher.top_k = 1 if self.variable_k else self.router.top_k
        self.token_dispatcher = token_dispatcher

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        assert routed_experts is None, "PrimeMoE does not support router replay"
        bs, slen, dim = x.shape
        x = x.view(-1, dim)
        top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum, scores = self.router(x)

        if self.balance_rows is not None and torch.is_grad_enabled():
            self.seq_balance = sequence_balance(scores, selected_experts_indices, self.balance_rows)

        with torch.no_grad():
            record_moe_routing_statistics(
                self.tokens_per_expert, self.routing_confidence_sum, num_tokens_per_expert, routing_confidence_sum
            )

        if self.variable_k:
            # Dispatch only the (token, routed expert) pairs, without the null experts.
            routed = selected_experts_indices < self.experts.num_experts
            if torch.is_grad_enabled():
                with torch.no_grad():
                    self.routed_k_counts += torch.bincount(routed.sum(dim=1), minlength=self.routed_k_counts.numel())
            token_indices, slots = routed.nonzero(as_tuple=True)
            pair_output = self.token_dispatcher.run(
                x[token_indices],
                top_scores[token_indices, slots].unsqueeze(1),
                selected_experts_indices[token_indices, slots].unsqueeze(1),
                self.experts,
                score_before_experts=self.score_before_experts,
            )
            routed_output = (
                torch.zeros(x.shape, dtype=torch.float32, device=x.device)
                .index_add(0, token_indices, pair_output.float())
                .to(x.dtype)
            )
        else:
            routed_output = self.token_dispatcher.run(
                x, top_scores, selected_experts_indices, self.experts, score_before_experts=self.score_before_experts
            )

        shared_output = self.shared_expert(x) if self.shared_expert is not None else None
        self.token_dispatcher.synchronize()
        if shared_output is not None:
            routed_output = routed_output + shared_output
        return routed_output.reshape(bs, slen, dim)

    def init_weights(self, init_std: float, buffer_device: torch.device):
        super().init_weights(init_std, buffer_device)
        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.router.num_experts, dtype=torch.float32)
            self.router.selection_bias = torch.zeros(self.router.num_experts, dtype=torch.float32)
            if self.routed_k_counts is not None:
                self.routed_k_counts = torch.zeros_like(self.routed_k_counts)
