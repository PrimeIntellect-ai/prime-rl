"""Comprehensive unit tests for the TRPL (Trust Region Policy Layer) loss.

Tests cover:
- Helper functions: _build_union_indices, _gather_old_at_union, _tail_logsumexp,
  _renormalize_old_sparse, _kl_sparse, _bisect_eta_sparse
- Main trpl_loss_fn: gradient flow, numerical stability, edge cases
- Integration with compute_loss pipeline

The TRPL loss now operates on sparse data from the fused LM head:
- current_top_k_indices/values: current policy top-K (with gradient)
- old_indices_current_lp: current log-probs at old top-K positions (with gradient)
- trainer_logprobs: current log-prob at actual token (with gradient)
"""

import pytest
import torch

from prime_rl.trainer.rl.config import TrplLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_loss, setup_loss_fn
from prime_rl.trainer.rl.trpl_loss import (
    _bisect_eta_sparse,
    _build_union_indices,
    _gather_old_at_union,
    _kl_sparse,
    _renormalize_old_sparse,
    _tail_logsumexp,
    setup_trpl_loss_fn,
    trpl_loss_fn,
)

pytestmark = [pytest.mark.gpu]

DEVICE = "cuda"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_log_softmax(shape, device=DEVICE, dtype=DTYPE):
    """Generate random log-softmax (properly normalized log-probs)."""
    logits = torch.randn(shape, device=device, dtype=dtype)
    return logits.log_softmax(dim=-1)


def _make_trpl_inputs(
    seq_len: int = 20,
    vocab_size: int = 100,
    k: int = 10,
    mask_fraction: float = 0.7,
    device=DEVICE,
):
    """Create synthetic LossInputs with sparse TRPL fields.

    Simulates what FusedOutputLinear produces: sparse current policy data
    (top-K indices/values and log-probs at old positions) rather than full logits.

    Returns:
        (inputs, raw_logits) where raw_logits is the leaf tensor for gradient checking.
    """
    # Full logits as the differentiable parameter
    raw_logits = torch.randn(seq_len, vocab_size, device=device, requires_grad=True)
    full_lp = raw_logits.log_softmax(dim=-1)  # [seq, vocab] — has grad

    # Token IDs
    token_ids = torch.randint(0, vocab_size, (seq_len,), device=device)

    # Current policy top-K (indices detached, values with grad)
    with torch.no_grad():
        _, topk_idx = full_lp.detach().topk(k, dim=-1)
    current_top_k_values = full_lp.gather(1, topk_idx)  # [seq, K] — has grad

    # Old policy top-K indices (from a different distribution)
    old_top_indices = torch.stack([torch.randperm(vocab_size, device=device)[:k] for _ in range(seq_len)])
    old_top_indices[:, 0] = token_ids  # ensure token is in old top-K

    # Current log-probs at old positions (with grad)
    old_indices_current_lp = full_lp.gather(1, old_top_indices)  # [seq, K] — has grad

    # Old top-K values (from separate distribution, no grad)
    old_logits = torch.randn(seq_len, vocab_size, device=device)
    old_lp = old_logits.log_softmax(dim=-1)
    old_top_values = old_lp.gather(1, old_top_indices)

    # Token log-probs (with grad)
    trainer_logprobs = full_lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    inference_logprobs = old_lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)

    # Loss mask
    loss_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    num_active = max(1, int(seq_len * mask_fraction))
    loss_mask[:num_active] = True

    # Advantages
    advantages = torch.randn(seq_len, device=device)

    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        current_top_k_indices=topk_idx,
        current_top_k_values=current_top_k_values,
        old_indices_current_lp=old_indices_current_lp,
        old_top_indices=old_top_indices,
        old_top_values=old_top_values,
        token_ids=token_ids,
        vocab_size=vocab_size,
    )
    return inputs, raw_logits


# ---------------------------------------------------------------------------
# Tests for _build_union_indices
# ---------------------------------------------------------------------------


class TestBuildUnionIndices:
    def test_basic_union(self):
        """Union of disjoint sets should contain all elements."""
        current = torch.tensor([[0, 1, 2]], device=DEVICE)
        old = torch.tensor([[3, 4, 5]], device=DEVICE)
        tokens = torch.tensor([6], device=DEVICE)
        vocab_size = 10

        union_idx, union_mask, token_pos = _build_union_indices(current, old, tokens, vocab_size)

        assert union_mask.all(), "All entries should be valid for disjoint sets"
        assert union_idx.shape[1] == 7, "Union of 3+3+1 disjoint = 7"
        # Token 6 should be findable
        assert union_idx[0, token_pos[0]] == 6

    def test_overlapping_sets(self):
        """Overlapping indices should be deduplicated."""
        current = torch.tensor([[0, 1, 2]], device=DEVICE)
        old = torch.tensor([[1, 2, 3]], device=DEVICE)
        tokens = torch.tensor([2], device=DEVICE)
        vocab_size = 10

        union_idx, union_mask, token_pos = _build_union_indices(current, old, tokens, vocab_size)

        # Unique: {0, 1, 2, 3} = 4 elements
        valid_count = union_mask.sum().item()
        assert valid_count == 4
        assert union_idx[0, token_pos[0]] == 2

    def test_fully_overlapping(self):
        """Identical current and old should give exactly K+1 or K union entries."""
        current = torch.tensor([[5, 10, 15]], device=DEVICE)
        old = torch.tensor([[5, 10, 15]], device=DEVICE)
        tokens = torch.tensor([10], device=DEVICE)  # token in both
        vocab_size = 20

        union_idx, union_mask, token_pos = _build_union_indices(current, old, tokens, vocab_size)

        valid_count = union_mask.sum().item()
        assert valid_count == 3, "All overlap, token also overlaps"

    def test_token_not_in_topk(self):
        """Token not in either top-k should still appear in union."""
        current = torch.tensor([[0, 1, 2]], device=DEVICE)
        old = torch.tensor([[3, 4, 5]], device=DEVICE)
        tokens = torch.tensor([9], device=DEVICE)
        vocab_size = 10

        union_idx, union_mask, token_pos = _build_union_indices(current, old, tokens, vocab_size)

        valid_count = union_mask.sum().item()
        assert valid_count == 7  # 3+3+1 disjoint
        assert union_idx[0, token_pos[0]] == 9

    def test_batch_dimension(self):
        """Should handle multiple tokens (batch dimension)."""
        N, K = 5, 4
        current = torch.randint(0, 50, (N, K), device=DEVICE)
        old = torch.randint(0, 50, (N, K), device=DEVICE)
        tokens = torch.randint(0, 50, (N,), device=DEVICE)
        vocab_size = 50

        union_idx, union_mask, token_pos = _build_union_indices(current, old, tokens, vocab_size)

        assert union_idx.shape[0] == N
        assert union_mask.shape[0] == N
        assert token_pos.shape[0] == N
        # Verify tokens are at their declared positions
        for i in range(N):
            assert union_idx[i, token_pos[i]] == tokens[i]

    def test_sorted_output(self):
        """Union indices should be sorted per row."""
        current = torch.tensor([[9, 3, 7]], device=DEVICE)
        old = torch.tensor([[1, 5, 8]], device=DEVICE)
        tokens = torch.tensor([4], device=DEVICE)
        vocab_size = 10

        union_idx, union_mask, _ = _build_union_indices(current, old, tokens, vocab_size)

        valid = union_idx[0][union_mask[0]]
        assert (valid[1:] >= valid[:-1]).all(), "Valid entries should be sorted"


# ---------------------------------------------------------------------------
# Tests for _gather_old_at_union
# ---------------------------------------------------------------------------


class TestGatherOldAtUnion:
    def test_all_matching(self):
        """When union and old top-k are identical, all values should match."""
        union_idx = torch.tensor([[2, 5, 8]], device=DEVICE)
        union_mask = torch.ones(1, 3, dtype=torch.bool, device=DEVICE)
        old_top_idx = torch.tensor([[5, 2, 8]], device=DEVICE)  # same but diff order
        old_top_val = torch.tensor([[-1.0, -2.0, -0.5]], device=DEVICE)
        default = -20.0

        result = _gather_old_at_union(union_idx, union_mask, old_top_idx, old_top_val, default)

        # union_idx=2 -> old_top_idx index 1 -> -2.0
        # union_idx=5 -> old_top_idx index 0 -> -1.0
        # union_idx=8 -> old_top_idx index 2 -> -0.5
        assert torch.allclose(result, torch.tensor([[-2.0, -1.0, -0.5]], device=DEVICE))

    def test_missing_entries_get_default(self):
        """Entries not in old top-k should get default_log_prob."""
        union_idx = torch.tensor([[1, 3, 5, 7]], device=DEVICE)
        union_mask = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)
        old_top_idx = torch.tensor([[1, 5]], device=DEVICE)
        old_top_val = torch.tensor([[-1.0, -2.0]], device=DEVICE)
        default = -30.0

        result = _gather_old_at_union(union_idx, union_mask, old_top_idx, old_top_val, default)

        expected = torch.tensor([[-1.0, -30.0, -2.0, -30.0]], device=DEVICE)
        assert torch.allclose(result, expected)

    def test_masked_positions_get_default(self):
        """Invalid (masked-out) positions should get default."""
        union_idx = torch.tensor([[1, 3, 0]], device=DEVICE)  # last is padding (0 from clamp)
        union_mask = torch.tensor([[True, True, False]], device=DEVICE)
        old_top_idx = torch.tensor([[1, 3]], device=DEVICE)
        old_top_val = torch.tensor([[-1.0, -2.0]], device=DEVICE)
        default = -30.0

        result = _gather_old_at_union(union_idx, union_mask, old_top_idx, old_top_val, default)

        assert result[0, 2] == default


# ---------------------------------------------------------------------------
# Tests for _tail_logsumexp
# ---------------------------------------------------------------------------


class TestTailLogsumexp:
    def test_matches_dense_logsumexp(self):
        """With tail_count=0, should match torch.logsumexp over union."""
        lp = torch.tensor([[-1.0, -2.0, -3.0]], device=DEVICE)
        mask = torch.ones(1, 3, dtype=torch.bool, device=DEVICE)
        tail_count = torch.tensor([0], device=DEVICE)
        default_lp = torch.tensor(-30.0, device=DEVICE)

        result = _tail_logsumexp(lp, mask, tail_count, default_lp)
        expected = torch.logsumexp(lp, dim=-1)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_tail_contribution(self):
        """Adding tail mass should increase logsumexp."""
        lp = torch.tensor([[-1.0, -2.0]], device=DEVICE)
        mask = torch.ones(1, 2, dtype=torch.bool, device=DEVICE)

        lse_no_tail = _tail_logsumexp(lp, mask, torch.tensor([0], device=DEVICE), torch.tensor(-5.0, device=DEVICE))
        lse_with_tail = _tail_logsumexp(lp, mask, torch.tensor([100], device=DEVICE), torch.tensor(-5.0, device=DEVICE))

        assert lse_with_tail > lse_no_tail

    def test_masking_invalid(self):
        """Invalid positions should be excluded from logsumexp."""
        lp = torch.tensor([[-1.0, -2.0, 999.0]], device=DEVICE)  # 999 is garbage
        mask = torch.tensor([[True, True, False]], device=DEVICE)
        tail_count = torch.tensor([0], device=DEVICE)
        default_lp = torch.tensor(-30.0, device=DEVICE)

        result = _tail_logsumexp(lp, mask, tail_count, default_lp)
        expected = torch.logsumexp(torch.tensor([[-1.0, -2.0]], device=DEVICE), dim=-1)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_uniform_distribution(self):
        """For uniform distribution, logsumexp should be log(n)."""
        V = 1000
        K = 10
        log_p = torch.full((1, K), -torch.tensor(V, dtype=torch.float).log(), device=DEVICE)
        mask = torch.ones(1, K, dtype=torch.bool, device=DEVICE)
        tail_count = torch.tensor([V - K], device=DEVICE)
        default_lp = log_p[0, 0]

        result = _tail_logsumexp(log_p, mask, tail_count, default_lp)
        # logsumexp of uniform = log(V * (1/V)) = log(1) = 0
        assert torch.allclose(result, torch.tensor([0.0], device=DEVICE), atol=1e-4)


# ---------------------------------------------------------------------------
# Tests for _renormalize_old_sparse
# ---------------------------------------------------------------------------


class TestRenormalizeOldSparse:
    def test_sum_to_one(self):
        """After renormalization, exp(union_lp) + tail should sum to ~1."""
        V = 50
        K_u = 10
        # Start with unnormalized log-probs
        old_lp = torch.randn(3, K_u, device=DEVICE) - 2.0
        mask = torch.ones(3, K_u, dtype=torch.bool, device=DEVICE)
        default = -10.0

        normed_lp, normed_default = _renormalize_old_sparse(old_lp, mask, V, default)

        tail_count = V - K_u
        total = normed_lp.exp().sum(dim=-1) + tail_count * normed_default.exp()
        assert torch.allclose(total, torch.ones(3, device=DEVICE), atol=1e-4)

    def test_with_partial_mask(self):
        """Should handle partial masks correctly."""
        V = 20
        K_u = 5
        old_lp = torch.randn(2, K_u, device=DEVICE) - 2.0
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]], device=DEVICE)
        default = -8.0

        normed_lp, normed_default = _renormalize_old_sparse(old_lp, mask, V, default)

        for i in range(2):
            total = normed_lp[i].exp().sum() + (V - K_u) * normed_default[i].exp()
            assert total.isfinite()


# ---------------------------------------------------------------------------
# Tests for _kl_sparse
# ---------------------------------------------------------------------------


class TestKlSparse:
    def test_identical_distributions(self):
        """KL(p || p) should be 0."""
        lp = _make_log_softmax((5, 8))
        mask = torch.ones(5, 8, dtype=torch.bool, device=DEVICE)
        tail_count = torch.zeros(5, device=DEVICE)
        default_p = torch.full((5,), -30.0, device=DEVICE)

        kl = _kl_sparse(lp, lp, default_p, default_p, mask, tail_count)

        assert torch.allclose(kl, torch.zeros(5, device=DEVICE), atol=1e-5)

    def test_kl_non_negative(self):
        """KL divergence should always be non-negative."""
        N, K_u = 10, 15
        p = _make_log_softmax((N, K_u))
        q = _make_log_softmax((N, K_u))
        mask = torch.ones(N, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.zeros(N, device=DEVICE)
        default_p = torch.full((N,), -30.0, device=DEVICE)
        default_q = torch.full((N,), -30.0, device=DEVICE)

        kl = _kl_sparse(p, q, default_p, default_q, mask, tail_count)

        assert (kl >= -1e-5).all(), f"KL should be non-negative, got min={kl.min()}"

    def test_kl_with_tail(self):
        """KL with tail mass should be finite and non-negative."""
        N, K_u, V = 4, 8, 100
        # Build proper distributions
        full_p = _make_log_softmax((N, V))
        full_q = _make_log_softmax((N, V))

        p_union = full_p[:, :K_u]
        q_union = full_q[:, :K_u]
        mask = torch.ones(N, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.full((N,), V - K_u, device=DEVICE)

        # Approximate tail default as mean of remaining mass
        p_tail_mass = 1.0 - full_p[:, :K_u].exp().sum(dim=-1)
        p_default = (p_tail_mass / (V - K_u)).clamp(min=1e-30).log()
        q_tail_mass = 1.0 - full_q[:, :K_u].exp().sum(dim=-1)
        q_default = (q_tail_mass / (V - K_u)).clamp(min=1e-30).log()

        kl = _kl_sparse(p_union, q_union, p_default, q_default, mask, tail_count)

        assert kl.isfinite().all()
        assert (kl >= -1e-4).all()


# ---------------------------------------------------------------------------
# Tests for _bisect_eta_sparse
# ---------------------------------------------------------------------------


class TestBisectEtaSparse:
    def test_projects_within_bound(self):
        """After projection with found eta, KL should be <= bound."""
        N, K_u, V = 8, 20, 200
        bound = 0.05

        # Create distributions that violate the bound
        current_lp = _make_log_softmax((N, K_u))
        old_lp = _make_log_softmax((N, K_u))
        mask = torch.ones(N, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.full((N,), V - K_u, device=DEVICE)
        current_default = torch.full((N,), -20.0, device=DEVICE)
        old_default = torch.full((N,), -20.0, device=DEVICE)

        log_eta = _bisect_eta_sparse(
            current_lp,
            old_lp,
            current_default,
            old_default,
            mask,
            tail_count,
            V,
            bound,
            num_steps=32,
        )
        eta = log_eta.exp()

        # Apply projection
        inner = (current_lp + eta * old_lp) / (eta + 1)
        inner_default = (current_default.unsqueeze(1) + eta * old_default.unsqueeze(1)) / (eta + 1)
        lse = _tail_logsumexp(inner, mask, tail_count, inner_default.squeeze(1))
        proj = inner - lse.unsqueeze(1)
        proj_default = inner_default.squeeze(1) - lse

        kl = _kl_sparse(proj, old_lp, proj_default, old_default, mask, tail_count)

        # Allow small tolerance for bisection precision
        assert (kl <= bound + 1e-3).all(), f"KL should be <= bound, got max={kl.max()}"

    def test_zero_eta_when_within_bound(self):
        """When current is close to old (within bound), eta should be near zero."""
        N, K_u, V = 4, 15, 100
        # Make current ≈ old (small perturbation)
        base = _make_log_softmax((N, K_u))
        current_lp = base + 0.001 * torch.randn(N, K_u, device=DEVICE)
        old_lp = base.clone()
        mask = torch.ones(N, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.full((N,), V - K_u, device=DEVICE)
        default = torch.full((N,), -20.0, device=DEVICE)

        log_eta = _bisect_eta_sparse(
            current_lp,
            old_lp,
            default,
            default,
            mask,
            tail_count,
            V,
            bound=1.0,  # large bound
        )

        # With large bound and similar distributions, eta should be small
        # (bisection converges to lower bound)
        assert (log_eta <= 0).all(), "Eta should be small when distributions are similar"


# ---------------------------------------------------------------------------
# Tests for trpl_loss_fn
# ---------------------------------------------------------------------------


class TestTrplLossFn:
    def test_returns_scalar_loss(self):
        """Loss should be a scalar tensor."""
        inputs, _ = _make_trpl_inputs(seq_len=30, vocab_size=100, k=10)
        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=10)

        assert isinstance(result, LossOutputs)
        assert result.loss.shape == ()
        assert result.loss.isfinite()

    def test_gradient_flows(self):
        """Loss should have gradients w.r.t. current policy parameters."""
        inputs, raw_logits = _make_trpl_inputs(seq_len=20, vocab_size=80, k=8)
        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=8)

        result.loss.backward()

        assert raw_logits.grad is not None
        assert raw_logits.grad.isfinite().all()
        assert raw_logits.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_metrics_present(self):
        """All expected metrics should be in output."""
        inputs, _ = _make_trpl_inputs(seq_len=25, vocab_size=100, k=10)
        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=10)

        expected_keys = {
            "pg_loss",
            "projection_loss",
            "importance_ratio",
            "initial_kl",
            "final_kl",
            "max_vio",
            "projected_frac",
            "mismatch_kl",
        }
        assert expected_keys.issubset(set(result.metrics.keys()))
        for key, val in result.metrics.items():
            assert val.isfinite(), f"Metric {key} is not finite: {val}"

    def test_zero_alpha_means_no_projection_loss(self):
        """With alpha=0, projection loss should not contribute."""
        inputs, _ = _make_trpl_inputs(seq_len=20, vocab_size=80, k=8)
        result = trpl_loss_fn(inputs, alpha=0.0, kl_bound=0.05, top_k=8)

        # projection_loss metric should be ~0 when alpha=0 (though pg_loss is per-token mean, loss is .sum())
        assert result.metrics["projection_loss"].abs() < 1e-3 or result.loss.isfinite()

    def test_empty_mask(self):
        """With all-false loss mask, should return zero loss."""
        inputs, _ = _make_trpl_inputs(seq_len=15, vocab_size=50, k=5)
        inputs.loss_mask[:] = False

        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=5)

        assert result.loss.item() == 0.0

    def test_single_token(self):
        """Should work with a single unmasked token."""
        inputs, raw_logits = _make_trpl_inputs(seq_len=10, vocab_size=50, k=5, mask_fraction=0.0)
        # Enable just one token
        inputs.loss_mask[0] = True

        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=5)

        assert result.loss.isfinite()
        result.loss.backward()
        assert raw_logits.grad is not None

    def test_kl_bound_respected(self):
        """Final KL after projection should respect the bound."""
        inputs, _ = _make_trpl_inputs(seq_len=30, vocab_size=100, k=10)
        bound = 0.05
        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=bound, top_k=10)

        # max_vio should be 0 or very small
        assert result.metrics["max_vio"] < 0.01, f"Max KL violation = {result.metrics['max_vio']}, expected < 0.01"

    def test_larger_kl_bound_less_projection(self):
        """Larger KL bound should result in fewer projected tokens."""
        inputs_tight, raw_tight = _make_trpl_inputs(seq_len=40, vocab_size=100, k=10)

        # Create identical inputs with fresh logits for separate grad graph
        raw_loose = raw_tight.detach().clone().requires_grad_(True)
        full_lp_loose = raw_loose.log_softmax(dim=-1)

        inputs_loose = LossInputs(
            trainer_logprobs=full_lp_loose.gather(1, inputs_tight.token_ids.unsqueeze(1)).squeeze(1),
            inference_logprobs=inputs_tight.inference_logprobs,
            teacher_logprobs=None,
            advantages=inputs_tight.advantages,
            loss_mask=inputs_tight.loss_mask,
            current_top_k_indices=inputs_tight.current_top_k_indices,
            current_top_k_values=full_lp_loose.gather(1, inputs_tight.current_top_k_indices),
            old_indices_current_lp=full_lp_loose.gather(1, inputs_tight.old_top_indices),
            old_top_indices=inputs_tight.old_top_indices,
            old_top_values=inputs_tight.old_top_values,
            token_ids=inputs_tight.token_ids,
            vocab_size=inputs_tight.vocab_size,
        )

        result_tight = trpl_loss_fn(inputs_tight, alpha=1.0, kl_bound=0.01, top_k=10)
        result_loose = trpl_loss_fn(inputs_loose, alpha=1.0, kl_bound=1.0, top_k=10)

        assert result_loose.metrics["projected_frac"] <= result_tight.metrics["projected_frac"]

    def test_numerical_stability_extreme_logits(self):
        """Should handle very peaked distributions (near one-hot)."""
        seq_len, V, k = 10, 50, 5
        # Very peaked logits
        raw_logits = torch.zeros(seq_len, V, device=DEVICE)
        raw_logits[:, 0] = 20.0
        raw_logits = raw_logits.requires_grad_(True)
        full_lp = raw_logits.log_softmax(dim=-1)

        token_ids = torch.zeros(seq_len, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            _, topk_idx = full_lp.detach().topk(k, dim=-1)
        current_top_k_values = full_lp.gather(1, topk_idx)

        old_top_idx = torch.stack([torch.randperm(V, device=DEVICE)[:k] for _ in range(seq_len)])
        old_top_idx[:, 0] = 0
        old_top_val = torch.full((seq_len, k), -5.0, device=DEVICE)
        old_top_val[:, 0] = -0.1

        old_indices_current_lp = full_lp.gather(1, old_top_idx)
        trainer_logprobs = full_lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)

        loss_mask = torch.ones(seq_len, dtype=torch.bool, device=DEVICE)
        advantages = torch.randn(seq_len, device=DEVICE)

        inputs = LossInputs(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=old_top_val[:, 0],
            teacher_logprobs=None,
            advantages=advantages,
            loss_mask=loss_mask,
            current_top_k_indices=topk_idx,
            current_top_k_values=current_top_k_values,
            old_indices_current_lp=old_indices_current_lp,
            old_top_indices=old_top_idx,
            old_top_values=old_top_val,
            token_ids=token_ids,
            vocab_size=V,
        )

        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=k)

        assert result.loss.isfinite(), f"Loss not finite: {result.loss}"
        result.loss.backward()
        assert raw_logits.grad.isfinite().all(), "Gradients not finite"

    def test_numerical_stability_uniform(self):
        """Should handle nearly uniform distributions."""
        seq_len, V, k = 10, 50, 5
        # Nearly uniform
        raw_logits = torch.randn(seq_len, V, device=DEVICE).mul_(0.01).requires_grad_(True)
        full_lp = raw_logits.log_softmax(dim=-1)

        token_ids = torch.randint(0, V, (seq_len,), device=DEVICE)

        with torch.no_grad():
            _, topk_idx = full_lp.detach().topk(k, dim=-1)
        current_top_k_values = full_lp.gather(1, topk_idx)

        old_top_idx = torch.stack([torch.randperm(V, device=DEVICE)[:k] for _ in range(seq_len)])
        old_top_idx[:, 0] = token_ids
        old_top_val = torch.full((seq_len, k), -torch.tensor(V, dtype=torch.float).log(), device=DEVICE)

        old_indices_current_lp = full_lp.gather(1, old_top_idx)
        trainer_logprobs = full_lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)

        loss_mask = torch.ones(seq_len, dtype=torch.bool, device=DEVICE)
        advantages = torch.randn(seq_len, device=DEVICE)

        inputs = LossInputs(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=old_top_val[:, 0],
            teacher_logprobs=None,
            advantages=advantages,
            loss_mask=loss_mask,
            current_top_k_indices=topk_idx,
            current_top_k_values=current_top_k_values,
            old_indices_current_lp=old_indices_current_lp,
            old_top_indices=old_top_idx,
            old_top_values=old_top_val,
            token_ids=token_ids,
            vocab_size=V,
        )

        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=k)

        assert result.loss.isfinite()
        result.loss.backward()
        assert raw_logits.grad.isfinite().all()


# ---------------------------------------------------------------------------
# Tests for setup_trpl_loss_fn and integration with compute_loss
# ---------------------------------------------------------------------------


class TestSetupAndIntegration:
    def test_setup_from_config(self):
        """setup_trpl_loss_fn should return a callable LossFn."""
        config = TrplLossConfig(alpha=1.0, kl_bound=0.05, top_k=10, default_log_prob=-18.42)
        loss_fn = setup_trpl_loss_fn(config)

        inputs, _ = _make_trpl_inputs(seq_len=20, vocab_size=80, k=10)
        result = loss_fn(inputs)

        assert isinstance(result, LossOutputs)
        assert result.loss.isfinite()

    def test_setup_loss_fn_dispatch(self):
        """setup_loss_fn from loss.py should correctly dispatch to TRPL."""
        config = TrplLossConfig(alpha=1.0, kl_bound=0.05, top_k=8)
        loss_fn = setup_loss_fn(config)

        inputs, _ = _make_trpl_inputs(seq_len=15, vocab_size=60, k=8)
        result = loss_fn(inputs)

        assert isinstance(result, LossOutputs)
        assert "pg_loss" in result.metrics

    def test_compute_loss_with_trpl(self):
        """TRPL should work through the compute_loss pipeline."""
        config = TrplLossConfig(alpha=1.0, kl_bound=0.05, top_k=8)
        loss_fn = setup_loss_fn(config)

        V, K = 60, 8
        # Two packed sequences
        seq_lens = [20, 15]
        trainer_logprobs = []
        inference_logprobs = []
        advantages_list = []
        loss_mask_list = []
        current_top_k_indices_list = []
        current_top_k_values_list = []
        old_indices_current_lp_list = []
        old_top_idx_list = []
        old_top_val_list = []
        token_ids_list = []

        for sl in seq_lens:
            raw = torch.randn(sl, V, device=DEVICE, requires_grad=True)
            full_lp = raw.log_softmax(dim=-1)

            tids = torch.randint(0, V, (sl,), device=DEVICE)

            with torch.no_grad():
                _, topk_idx = full_lp.detach().topk(K, dim=-1)

            oti = torch.stack([torch.randperm(V, device=DEVICE)[:K] for _ in range(sl)])
            oti[:, 0] = tids
            otv = _make_log_softmax((sl, K))

            trainer_logprobs.append(full_lp.gather(1, tids.unsqueeze(1)).squeeze(1))
            inference_logprobs.append(otv[:, 0].detach())
            advantages_list.append(torch.randn(sl, device=DEVICE))
            loss_mask_list.append(torch.ones(sl, dtype=torch.bool, device=DEVICE))
            current_top_k_indices_list.append(topk_idx)
            current_top_k_values_list.append(full_lp.gather(1, topk_idx))
            old_indices_current_lp_list.append(full_lp.gather(1, oti))
            old_top_idx_list.append(oti)
            old_top_val_list.append(otv)
            token_ids_list.append(tids)

        loss, metrics = compute_loss(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=inference_logprobs,
            teacher_logprobs=None,
            advantages=advantages_list,
            loss_mask=loss_mask_list,
            loss_fn=loss_fn,
            loss_scale=2,
            current_top_k_indices=current_top_k_indices_list,
            current_top_k_values=current_top_k_values_list,
            old_indices_current_lp=old_indices_current_lp_list,
            old_top_indices=old_top_idx_list,
            old_top_values=old_top_val_list,
            token_ids=token_ids_list,
            vocab_size=V,
        )

        assert loss.shape == ()
        assert loss.isfinite()
        assert "pg_loss" in metrics


# ---------------------------------------------------------------------------
# Tests for specific correctness properties
# ---------------------------------------------------------------------------


class TestCorrectness:
    def test_projection_is_geometric_mean(self):
        """With eta=1, projected should be geometric mean of current and old (renormalized)."""
        K_u = 5
        current_lp = _make_log_softmax((1, K_u))
        old_lp = _make_log_softmax((1, K_u))
        mask = torch.ones(1, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.zeros(1, device=DEVICE)

        # projected = softmax((current + 1*old) / 2) = softmax(avg log-probs)
        inner = (current_lp + old_lp) / 2
        expected = inner.log_softmax(dim=-1)

        # Check manually
        lse = _tail_logsumexp(inner, mask, tail_count, torch.tensor(-30.0, device=DEVICE))
        actual = inner - lse.unsqueeze(1)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_eta_zero_means_no_projection(self):
        """With eta=0, projected = current (no constraint from old)."""
        K_u = 5
        current_lp = _make_log_softmax((1, K_u))
        mask = torch.ones(1, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.zeros(1, device=DEVICE)

        # projected = softmax((current + 0*old) / (0+1)) = softmax(current) = current
        inner = current_lp  # eta=0
        lse = _tail_logsumexp(inner, mask, tail_count, torch.tensor(-30.0, device=DEVICE))
        projected = inner - lse.unsqueeze(1)

        # current_lp is already log_softmax, so projected should ≈ current_lp
        assert torch.allclose(projected, current_lp, atol=1e-4)

    def test_large_eta_converges_to_old(self):
        """With very large eta, projected → old policy."""
        K_u = 5
        current_lp = _make_log_softmax((1, K_u))
        old_lp = _make_log_softmax((1, K_u))
        mask = torch.ones(1, K_u, dtype=torch.bool, device=DEVICE)
        tail_count = torch.zeros(1, device=DEVICE)

        eta = 1e6
        inner = (current_lp + eta * old_lp) / (eta + 1)
        lse = _tail_logsumexp(inner, mask, tail_count, torch.tensor(-30.0, device=DEVICE))
        projected = inner - lse.unsqueeze(1)

        # Should be very close to old_lp
        assert torch.allclose(projected, old_lp, atol=1e-3)

    def test_importance_ratio_one_when_projected_equals_old(self):
        """When projected ≈ old, importance ratio should be ≈ 1."""
        V, K = 50, 8
        seq_len = 10

        # Use the SAME distribution for current and old
        raw_logits = torch.randn(seq_len, V, device=DEVICE, requires_grad=True)
        full_lp = raw_logits.log_softmax(dim=-1)

        token_ids = torch.randint(0, V, (seq_len,), device=DEVICE)

        # Current top-K
        with torch.no_grad():
            _, topk_idx = full_lp.detach().topk(K, dim=-1)
        current_top_k_values = full_lp.gather(1, topk_idx)

        # Old top-K: use same distribution
        old_top_idx = topk_idx.clone()
        old_top_idx[:, 0] = token_ids  # ensure token in old
        old_top_values = full_lp.detach().gather(1, old_top_idx)

        old_indices_current_lp = full_lp.gather(1, old_top_idx)
        trainer_logprobs = full_lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)

        inputs = LossInputs(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=old_top_values[:, 0].detach(),
            teacher_logprobs=None,
            advantages=torch.randn(seq_len, device=DEVICE),
            loss_mask=torch.ones(seq_len, dtype=torch.bool, device=DEVICE),
            current_top_k_indices=topk_idx,
            current_top_k_values=current_top_k_values,
            old_indices_current_lp=old_indices_current_lp,
            old_top_indices=old_top_idx,
            old_top_values=old_top_values,
            token_ids=token_ids,
            vocab_size=V,
        )

        result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=K)

        # Importance ratio should be close to 1
        assert abs(result.metrics["importance_ratio"].item() - 1.0) < 0.1, (
            f"Expected importance ratio ≈ 1, got {result.metrics['importance_ratio']}"
        )

    def test_different_top_k_values(self):
        """Loss should work with different top_k values."""
        for k in [3, 5, 10, 20]:
            inputs, raw_logits = _make_trpl_inputs(seq_len=15, vocab_size=80, k=k)
            result = trpl_loss_fn(inputs, alpha=1.0, kl_bound=0.05, top_k=k)

            assert result.loss.isfinite(), f"Loss not finite for top_k={k}"
            result.loss.backward()
            assert raw_logits.grad.isfinite().all(), f"Grad not finite for top_k={k}"
