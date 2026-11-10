import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, ReinforceLossConfig
from prime_rl.trainer.models.layers.lm_head import TokenSupport
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    _mismatch_kl_from_log_ratio,
    compute_entropy,
    compute_loss,
    ref_kl_loss_fn,
    selective_log_softmax_with_support,
    setup_rl_loss_fn,
)

pytestmark = [pytest.mark.gpu]


# Component-plumbing tests use a plain policy gradient, which needs no sampler support.
PLAIN_PG = CustomLossConfig(import_path="tests.unit.train.rl.test_loss._plain_pg_loss")


def _ragged(trainer: torch.Tensor, sampler: torch.Tensor, valid: torch.Tensor, replayed: torch.Tensor) -> dict:
    """Padded [n, k] support fields as the flat entries of the valid slots."""
    rows = torch.arange(valid.shape[0]).unsqueeze(-1).expand_as(valid)
    return dict(
        trainer_support_logprobs=trainer[valid],
        sampler_support_logprobs=sampler[valid],
        support_rows=rows[valid],
        support_replayed=replayed,
    )


def _support_case():
    """Every action of one prefix as its own token, replayed against support {0..3} with p renormalized over it."""
    logits = torch.tensor([0.3, -0.4, 0.1, -0.9, -1.5, -2.0], requires_grad=True)
    logp = logits[:4].log_softmax(-1)
    q = torch.tensor([0.4, 0.2, 0.3, 0.1])
    n = len(q)
    support = _ragged(
        logp.expand(n, -1), q.log().expand(n, -1), torch.ones((n, n), dtype=torch.bool), torch.ones(n, dtype=torch.bool)
    )
    return logits, logp, q, support


def _support_inputs(logp, q, support, advantages, **kwargs):
    n = len(q)
    return LossInputs(
        trainer_logprobs=logp,
        inference_logprobs=q.log(),
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=torch.ones(n, dtype=torch.bool),
        loss_weights=q,
        **support,
        **kwargs,
    )


@pytest.mark.parametrize("constant_advantage", [False, True])
def test_score_centering_matches_dense_gradient(constant_advantage):
    logits, logp, q, support = _support_case()
    n = len(q)
    advantages = torch.ones(n) if constant_advantage else torch.linspace(-1.0, 1.3, n)
    loss = setup_rl_loss_fn(ReinforceLossConfig(kl_coeff=0.0)).loss(_support_inputs(logp, q, support, advantages)).loss

    reference = -(q * advantages * (logp - (q * logp).sum())).sum()
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(reference, logits)[0], atol=2e-7, rtol=2e-6)
    if constant_advantage:
        torch.testing.assert_close(gradient, torch.zeros_like(logits), atol=2e-7, rtol=0)


def test_kl_matches_dense_kl():
    logits, logp, q, support = _support_case()
    n = len(q)
    inputs = _support_inputs(logp, q, support, torch.zeros(n))
    inputs.loss_weights = None
    result = setup_rl_loss_fn(ReinforceLossConfig(kl_coeff=0.5)).loss(inputs)

    kl = (q * (q.log() - logp)).sum()
    reference = 0.5 * n * kl
    torch.testing.assert_close(result.loss, reference)
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, logits, retain_graph=True)[0], torch.autograd.grad(reference, logits)[0]
    )
    torch.testing.assert_close(result.metrics["kl"], kl.detach().expand(n))
    assert (result.metrics["score_centering/replayed"] == 1).all()


def test_sampled_estimator_is_unbiased():
    """Over many prefixes, the K-draw center and KL match the exact sums over q in expectation, and the
    gradient vanishes at constant advantage."""
    gen = torch.Generator().manual_seed(0)
    n, vocab, k = 20000, 6, 4
    logits = torch.randn(vocab, generator=gen).requires_grad_()
    q = (logits.detach() + 0.5 * torch.randn(vocab, generator=gen)).softmax(-1)
    sampled = torch.multinomial(q, n, replacement=True, generator=gen)
    draws = torch.multinomial(q, n * k, replacement=True, generator=gen).view(n, k)
    full = torch.arange(vocab).expand(n, -1)

    def run(ids, replayed, kl_coeff):
        logp = logits.log_softmax(-1).expand(n, -1)
        inputs = LossInputs(
            trainer_logprobs=logp.gather(-1, sampled[:, None]).squeeze(-1),
            inference_logprobs=q.log()[sampled],
            ref_logprobs=None,
            advantages=torch.full((n,), 0.0 if kl_coeff else 1.0),
            loss_mask=torch.ones(n, dtype=torch.bool),
            **_ragged(
                logp.gather(-1, ids), q.log()[ids], torch.ones(ids.shape, dtype=torch.bool), torch.full((n,), replayed)
            ),
        )
        loss = setup_rl_loss_fn(ReinforceLossConfig(kl_coeff=kl_coeff)).loss(inputs).loss / n
        return loss, torch.autograd.grad(loss, logits)[0]

    for kl_coeff in (0.0, 0.5):
        estimate, estimate_grad = run(draws, False, kl_coeff)
        exact, exact_grad = run(full, True, kl_coeff)
        torch.testing.assert_close(estimate, exact, atol=0.01, rtol=0)
        torch.testing.assert_close(estimate_grad, exact_grad, atol=0.01, rtol=0)
    torch.testing.assert_close(run(draws, False, 0.0)[1], torch.zeros(vocab), atol=0.02, rtol=0)


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        advantages,
        loss_mask=loss_mask,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_rl_loss_fn_with_custom_config():
    """Test setup_rl_loss_fn with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    rl_loss_fn = setup_rl_loss_fn(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        ref_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = rl_loss_fn.loss(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_ref_kl_loss_stays_finite_with_extreme_ratios_and_masked_nan():
    trainer_logprobs = torch.tensor([-10.0, 0.0, float("nan")], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([-100.0, -100.0, float("nan")], device="cuda"),
        ref_logprobs=torch.tensor([-1.0, -1.0, float("nan")], device="cuda"),
        advantages=torch.zeros(3, device="cuda"),
        loss_mask=torch.tensor([True, True, False], device="cuda"),
    )

    result = ref_kl_loss_fn(inputs)

    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    assert torch.isfinite(trainer_logprobs.grad).all()


def test_mismatch_kl_retains_small_positive_values():
    log_ratio = torch.tensor([1e-4], device="cuda")
    torch.testing.assert_close(
        _mismatch_kl_from_log_ratio(log_ratio), torch.tensor([5e-9], device="cuda"), rtol=1e-3, atol=0
    )


def test_ce_component_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(3, dtype=torch.float32).cuda()]
    ce_weights = [torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=2,
        ref_kl_scale=1,
    )

    # loss = -sum(member logprobs) / ce_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_ce_component_applies_weights():
    """ECHO-style observation training: the ce weight stream scales the NLL per token."""
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(3, dtype=torch.float32).cuda()]
    ce_weights = [torch.tensor([0.1, 0.0, 0.1], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    # loss = 0.1 * (0.1 + 0.2) = 0.03
    assert torch.isclose(loss, torch.tensor(0.03, device=loss.device), atol=1e-6)


def test_explicit_rl_weights_match_absent_stream():
    """An explicit all-ones rl stream must equal the rl_weights=None hot path."""
    torch.manual_seed(0)
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda()]
    loss_mask = [torch.rand(50).cuda() > 0.3]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )
    loss_absent, _ = compute_loss(rl_weights=None, **kwargs)
    loss_explicit, _ = compute_loss(rl_weights=[torch.ones(50, dtype=torch.float32).cuda()], **kwargs)

    assert torch.equal(loss_absent, loss_explicit)


def test_disjoint_components_in_one_sequence():
    """ECHO/OPD-shaped sequence: rl, ce, and ref_kl on disjoint token sets."""
    n = 12
    torch.manual_seed(1)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    rl_weights = torch.zeros(n, dtype=torch.float32)
    rl_weights[:4] = 1.0
    ce_weights = torch.zeros(n, dtype=torch.float32)
    ce_weights[4:8] = 1.0
    ref_kl_weights = torch.zeros(n, dtype=torch.float32)
    ref_kl_weights[8:] = 1.0

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=[rl_weights.cuda()],
        ce_weights=[ce_weights.cuda()],
        ref_kl_weights=[ref_kl_weights.cuda()],
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    assert loss.shape == ()
    assert "nll" in metrics
    assert "ref_kl" in metrics
    assert "pg" in metrics


def test_empty_components_keep_backward_valid():
    """A fully truncated distillation sample (stamped streams survive truncation
    as all-zero prefixes) must train as a zero-gradient no-op, not crash backward."""
    trainer_logprobs = [torch.randn(6, dtype=torch.float32, device="cuda", requires_grad=True)]
    inference_logprobs = [torch.zeros(6, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(6, dtype=torch.float32).cuda()]
    loss_mask = [torch.zeros(6, dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(6, dtype=torch.float32).cuda()]
    ce_weights = [torch.zeros(6, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rl_weights=rl_weights,
        ce_weights=ce_weights,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert trainer_logprobs[0].grad is not None
    assert torch.equal(trainer_logprobs[0].grad, torch.zeros_like(trainer_logprobs[0].grad))


def test_overlapping_components_sum():
    """Components may overlap on the same token (e.g. RL + a CE behavior-cloning
    regularizer): the total is the sum of each component computed alone, each
    over its own normalization."""
    n = 8
    torch.manual_seed(2)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    ce_weights = [torch.full((n,), 0.5, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(PLAIN_PG)
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        ref_kl_weights=None,
        rl_loss_fn=rl_loss_fn,
        rl_scale=4,
        ce_scale=8,
        ref_kl_scale=1,
    )
    rl_only, _ = compute_loss(rl_weights=None, ce_weights=None, **kwargs)
    ce_only, _ = compute_loss(rl_weights=[torch.zeros(n, dtype=torch.float32).cuda()], ce_weights=ce_weights, **kwargs)
    both, _ = compute_loss(rl_weights=None, ce_weights=ce_weights, **kwargs)

    assert torch.isclose(both, rl_only + ce_only, atol=1e-6)


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )


def _plain_pg_loss(inputs: LossInputs) -> LossOutputs:
    mask = inputs.loss_mask
    loss = -inputs.advantages[mask] * inputs.trainer_logprobs[mask]
    if inputs.loss_weights is not None:
        loss = loss * inputs.loss_weights[mask]
    return LossOutputs(loss=loss.sum(), metrics={"pg": loss.detach()})


@pytest.mark.parametrize("constant_advantage", [False, True])
def test_score_centering_replay_matches_exact_supported_gradient(constant_advantage):
    logits = torch.tensor([0.2, -0.4, 0.8, 3.0], requires_grad=True)
    labels = torch.arange(3)
    q = torch.tensor([0.7, 0.2, 0.1])
    support = TokenSupport(
        ids=labels.repeat(3).int(),
        offsets=torch.arange(4) * 3,
        replayed=torch.ones(3, dtype=torch.bool),
        logprobs=q.log().repeat(3),
    )
    sampled, head = selective_log_softmax_with_support(logits.expand(1, 3, -1), labels.unsqueeze(0), support)
    logp = logits[:3].log_softmax(-1)
    advantages = torch.ones(3) if constant_advantage else torch.tensor([1.0, -0.5, 2.0])
    actual, metrics = compute_loss(
        trainer_logprobs=[sampled[0]],
        inference_logprobs=[q.log()],
        ref_logprobs=None,
        advantages=[advantages],
        loss_mask=[torch.ones(3, dtype=torch.bool)],
        rl_weights=[q],
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(ReinforceLossConfig(kl_coeff=0.0)),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        support=support,
        trainer_support_logprobs=head,
    )
    reference = -(q * advantages * (logp - (q * logp).sum())).sum()
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[3] == 0
    assert (metrics["score_centering/replayed"] == 1).all()
    if constant_advantage:
        torch.testing.assert_close(actual_grad, torch.zeros_like(logits), atol=1e-7, rtol=0)
