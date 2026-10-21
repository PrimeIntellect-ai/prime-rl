from typing import Literal

import torch
from torch import nn

from prime_rl.trainer.distributed.token_dispatcher import TokenDispatcher
from prime_rl.trainer.models.layers.activations import Activation
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
    record_moe_routing_statistics,
)
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig


class SituGlu(Activation):
    """Sigmoid Tanh Unit GLU (Kimi K3 report, eq. 12): SwiGLU with both factors soft-capped,
    ``(4 tanh(gate / 4) * sigmoid(gate)) * (25 tanh(up / 25))``; close to SwiGLU near 0 and bounded by 100."""

    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))


class PrimeRouter(TokenChoiceTopKRouter):
    """Sigmoid-scored top-k router with a selection bias and ``num_null_experts`` zero-output experts next to the
    routed ones (AdaMoE). Null experts are scored, selected and balanced like the others, so a token that selects
    them uses fewer routed experts. Returns the scores of all experts too, for the sequence-wise balance loss.

    With ``bias_on_logits`` the selection bias is added to the logits instead of the sigmoid scores. With
    ``quantile_balance`` set by the training loop, every forward with grad records each expert's quantile balancing
    threshold (``quantile_sum``, ``quantile_count``), from which the loop sets the next step's bias. If the loop also
    sets ``quantile_hist`` (``(num_experts, bins)`` counts) and ``quantile_grid`` (its ``[lo, hi]``), the forward
    instead adds each expert's required biases to that histogram (``record_quantile_histogram``)."""

    def __init__(
        self,
        *,
        dim: int,
        num_experts: int,
        num_null_experts: int,
        top_k: int,
        route_norm: bool,
        route_scale: float,
        bias_on_logits: bool,
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
        self.bias_on_logits = bias_on_logits
        self.quantile_balance = False
        self.register_buffer("quantile_sum", torch.zeros(self.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("quantile_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.quantile_hist: torch.Tensor | None = None
        self.quantile_grid: torch.Tensor | None = None
        self.quantile_range: torch.Tensor | None = None

    @torch.no_grad()
    def record_quantiles(self, selection_scores: torch.Tensor, biased_scores: torch.Tensor) -> None:
        """Quantile balancing (Su 2026; Kimi K3, Marin): each token's threshold is its largest unselected biased score,
        and each expert's new bias is minus the margin over that threshold that ``top_k / num_experts`` of the tokens
        exceed, so that with it every expert would have been selected by exactly that share of these tokens."""
        thresholds = biased_scores.topk(self.top_k + 1, dim=1).values[:, -1:]
        margins = selection_scores - thresholds
        rank = max(1, round(margins.shape[0] * self.top_k / self.num_experts))
        self.quantile_sum += margins.topk(rank, dim=0).values[-1]
        self.quantile_count += 1

    @torch.no_grad()
    def record_quantile_histogram(self, selection_scores: torch.Tensor, biased_scores: torch.Tensor) -> None:
        """Histogram estimator of quantile balancing (Kimi K3 report, §2.3.3 and appendix D): add each expert's
        required biases ``threshold - score`` (the bias that puts the expert exactly at the token's threshold) to its
        row of ``quantile_hist`` on the grid ``quantile_grid``. Values outside the grid count in the edge bins, so the
        counts below any in-grid value stay exact. Also tracks the values' ``[min, max]`` in ``quantile_range``."""
        thresholds = biased_scores.topk(self.top_k + 1, dim=1).values[:, -1:]
        required = thresholds - selection_scores
        num_experts, bins = self.quantile_hist.shape
        lo, hi = self.quantile_grid
        index = ((required - lo) * (bins / (hi - lo))).floor().clamp_(0, bins - 1).long()
        index += torch.arange(num_experts, device=index.device) * bins
        # scatter_add_ instead of bincount: CUDA bincount reads its input's maximum to the host, one sync per call.
        self.quantile_hist.view(-1).scatter_add_(
            0, index.view(-1), torch.ones_like(index.view(-1), dtype=self.quantile_hist.dtype)
        )
        self.quantile_range[0] = torch.minimum(self.quantile_range[0], required.min())
        self.quantile_range[1] = torch.maximum(self.quantile_range[1], required.max())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like ``TokenChoiceTopKRouter.forward`` with sigmoid scores; also returns the ``(bs*slen, num_experts)``
        scores of all experts."""
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        # The selection bias only affects which experts are selected, not their scores.
        selection_scores = logits.float() if self.bias_on_logits else scores
        biased_scores = selection_scores + self.selection_bias
        _, selected_experts_indices = torch.topk(biased_scores, k=self.top_k, dim=1, sorted=self.topk_sorted)
        if self.quantile_balance and torch.is_grad_enabled():
            if self.quantile_hist is None:
                self.record_quantiles(selection_scores, biased_scores)
            else:
                self.record_quantile_histogram(selection_scores, biased_scores)
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


def histogram_quantile_biases(hist: torch.Tensor, grid: torch.Tensor, top_k: int) -> torch.Tensor:
    """Each expert's ``top_k / num_experts`` quantile of the required biases counted in ``hist`` (``(num_experts,
    bins)`` on ``grid = [lo, hi]``), mean-centred: the first bin whose cumulative count reaches the target ``q`` plus
    linear interpolation inside it. With it, every expert would have been selected by ``q`` of the counted tokens.
    ``q`` comes from each expert's own count, so a forward recomputed under activation checkpointing (which counts
    every token twice) leaves the result unchanged."""
    num_experts, bins = hist.shape
    cumulative = hist.cumsum(dim=1)
    target = cumulative[:, -1] * top_k / num_experts
    selected = (cumulative < target.ceil()[:, None]).sum(dim=1).clamp(max=bins - 1)[:, None]
    inside = hist.gather(1, selected)
    below = cumulative.gather(1, selected) - inside
    fraction = ((target[:, None] - below) / inside.clamp(min=1)).clamp(0, 1)
    lo, hi = grid
    biases = (lo + (selected + fraction) * (hi - lo) / bins).squeeze(1)
    return biases - biases.mean()


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
    balance for its loss (``balance_rows``) and counts the routed experts per token (``routed_k_counts``).

    With ``latent_dim`` (LatentMoE), the routed experts work in that width: the token is projected down before the
    dispatch and the combined expert output is projected back up, with an RMS norm on the expert input
    (``latent_norm = "input"``, Marin) or on the combined output before the up-projection (``"output"``, Kimi K3's
    Stable LatentMoE). The router and the shared experts read the full-width token."""

    def __init__(
        self,
        *,
        router: PrimeRouter,
        experts: GroupedExperts,
        shared_expert: FeedForward | None,
        dim: int,
        latent_dim: int | None,
        latent_norm: Literal["input", "output"],
        norm_eps: float,
    ) -> None:
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
        self.latent_down = self.latent_norm = self.latent_up = None
        self.norm_latent_output = latent_norm == "output"
        if latent_dim is not None:
            self.latent_down = nn.Linear(dim, latent_dim, bias=False)
            self.latent_norm = RMSNorm(RMSNormConfig(hidden_size=latent_dim, eps=norm_eps))
            self.latent_up = nn.Linear(latent_dim, dim, bias=False)

    def prepare_expert_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_down is None:
            return x
        x = self.latent_down(x)
        return x if self.norm_latent_output else self.latent_norm(x)

    def prepare_expert_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_up is None:
            return x
        if self.norm_latent_output:
            x = self.latent_norm(x)
        return self.latent_up(x)

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

        expert_input = self.prepare_expert_input(x)
        if self.variable_k:
            # Dispatch only the (token, routed expert) pairs, without the null experts.
            routed = selected_experts_indices < self.experts.num_experts
            if torch.is_grad_enabled():
                with torch.no_grad():
                    self.routed_k_counts += torch.bincount(routed.sum(dim=1), minlength=self.routed_k_counts.numel())
            token_indices, slots = routed.nonzero(as_tuple=True)
            pair_output = self.token_dispatcher.run(
                expert_input[token_indices],
                top_scores[token_indices, slots].unsqueeze(1),
                selected_experts_indices[token_indices, slots].unsqueeze(1),
                self.experts,
                score_before_experts=self.score_before_experts,
            )
            routed_output = (
                torch.zeros(expert_input.shape, dtype=torch.float32, device=x.device)
                .index_add(0, token_indices, pair_output.float())
                .to(x.dtype)
            )
        else:
            routed_output = self.token_dispatcher.run(
                expert_input,
                top_scores,
                selected_experts_indices,
                self.experts,
                score_before_experts=self.score_before_experts,
            )

        shared_output = self.shared_expert(x) if self.shared_expert is not None else None
        self.token_dispatcher.synchronize()
        routed_output = self.prepare_expert_output(routed_output)
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
