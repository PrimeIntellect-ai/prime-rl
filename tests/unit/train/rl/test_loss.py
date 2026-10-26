import pytest
import torch

from prime_rl.configs.trainer import (
    CustomLossConfig,
    IcePopLossConfig,
    IPOLossConfig,
    ReinforceLossConfig,
)
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    _mismatch_kl_from_log_ratio,
    compute_entropy,
    compute_loss,
    ref_kl_loss_fn,
    setup_rl_loss_fn,
)

pytestmark = [pytest.mark.gpu]


WEIGHTINGS = [
    IPOLossConfig(eps=0.3, score_centering=True, score_centering_estimator="topk"),
    IcePopLossConfig(ratio_low=0.5, ratio_high=2.0, score_centering=True, score_centering_estimator="topk"),
    ReinforceLossConfig(score_centering=True, score_centering_estimator="topk"),
]


def _dense_weight(config, p, q):
    ratio = p / q
    if config.type == "ipo":
        return ((p - q).abs() <= config.eps) * ratio.clamp(max=config.max_importance_ratio)
    if config.type == "icepop":
        return ((ratio >= config.ratio_low) & (ratio <= config.ratio_high)) * ratio
    return torch.ones_like(p)


def _ragged(trainer: torch.Tensor, sampler: torch.Tensor, valid: torch.Tensor, replayed: torch.Tensor) -> dict:
    """Padded [n, k] support fields as the flat entries of the valid slots."""
    rows = torch.arange(valid.shape[0]).unsqueeze(-1).expand_as(valid)
    return dict(
        trainer_support_logprobs=trainer[valid],
        sampler_support_logprobs=sampler[valid],
        support_rows=rows[valid],
        support_replayed=replayed,
    )


def _support_case(replay: bool):
    """Every action of one prefix as its own token. Replay: support {0..3}, p renormalized over it and the
    tail empty. No replay: head {0, 1} of the full vocabulary and a sampler tail proportional to p."""
    logits = torch.tensor([0.3, -0.4, 0.1, -0.9, -1.5, -2.0], requires_grad=True)
    if replay:
        logp = logits[:4].log_softmax(-1)
        q = torch.tensor([0.4, 0.2, 0.3, 0.1])
        head, q_tail = 4, 0.0
    else:
        logp = logits.log_softmax(-1)
        p = logp.detach().exp()
        head = 2
        q_head = torch.tensor([0.45, 0.25])
        q_tail = 1 - q_head.sum()
        q = torch.cat((q_head, q_tail * p[head:] / p[head:].sum()))
    n = len(q)
    support = _ragged(
        logp[:head].expand(n, -1),
        q[:head].log().expand(n, -1),
        torch.ones((n, head), dtype=torch.bool),
        torch.full((n,), replay),
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


@pytest.mark.parametrize("replay", [True, False])
@pytest.mark.parametrize("config", WEIGHTINGS, ids=lambda c: c.type)
@pytest.mark.parametrize("constant_advantage", [False, True])
def test_score_centering_matches_dense_gradient(config, replay, constant_advantage):
    logits, logp, q, support = _support_case(replay)
    n = len(q)
    advantages = torch.ones(n) if constant_advantage else torch.linspace(-1.0, 1.3, n)
    loss = setup_rl_loss_fn(config).loss(_support_inputs(logp, q, support, advantages)).loss

    w = _dense_weight(config, logp.detach().exp(), q)
    baseline = (q * w * logp).sum()
    reference = -(q * advantages * (w * logp - baseline)).sum()
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(reference, logits)[0], atol=2e-7, rtol=2e-6)
    if constant_advantage:
        torch.testing.assert_close(gradient, torch.zeros_like(logits), atol=2e-7, rtol=0)


@pytest.mark.parametrize("replay", [True, False])
@pytest.mark.parametrize("kl_staleness", ["none", "inv_sqrt"])
def test_kl_matches_dense_kl(replay, kl_staleness):
    logits, logp, q, support = _support_case(replay)
    n = len(q)
    ages = torch.arange(n)
    config = ReinforceLossConfig(kl_coeff=0.5, kl_staleness=kl_staleness, score_centering_estimator="topk")
    inputs = _support_inputs(logp, q, support, torch.zeros(n), policy_ages=ages)
    inputs.loss_weights = None
    result = setup_rl_loss_fn(config).loss(inputs)

    beta = (
        torch.full((n,), 0.5) if kl_staleness == "none" else torch.where(ages > 0, 0.5 / ages.clamp_min(1).sqrt(), 0.0)
    )
    reference = (beta * (q * (q.log() - logp)).sum()).sum()
    torch.testing.assert_close(result.loss, reference)
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, logits, retain_graph=True)[0], torch.autograd.grad(reference, logits)[0]
    )
    assert torch.equal(result.metrics["kl/beta"], beta)


def _main_icepop_score_centering(config, inputs, head, sampler, valid):
    """IcePop with score centering as on main after #4012 (90fbb8db2)."""
    mask = inputs.loss_mask
    log_ratio = inputs.trainer_logprobs[mask] - inputs.inference_logprobs[mask]
    advantages = inputs.advantages[mask]
    low, high = log_ratio.new_tensor(config.ratio_low).log(), log_ratio.new_tensor(config.ratio_high).log()
    keep = (log_ratio.detach() >= low) & (log_ratio.detach() <= high)
    loss = (-config.adv_tau * advantages[keep] * torch.exp(log_ratio[keep])).sum()
    head, valid, sampler = head[mask], valid[mask], sampler[mask].detach()
    with torch.no_grad():
        p = head.exp().masked_fill(~valid, 0.0)
        q = sampler.exp().masked_fill(~valid, 0.0)
        accepted = valid & (head - sampler >= low) & (head - sampler <= high)
        mass = p.masked_fill(~accepted, 0.0)
        rho = (1.0 - q.sum(-1)).clamp_min(1e-6) / (1.0 - p.sum(-1)).clamp_min(1e-6)
        alpha = ((-rho.log() >= low) & (-rho.log() <= high)).to(p.dtype)
        mass = mass - alpha.unsqueeze(-1) * p
    return loss + (config.adv_tau * advantages * (mass * head).sum(-1)).sum()


def test_icepop_topk_tail_matches_main():
    torch.manual_seed(0)
    n, vocab, k = 64, 32, 8
    logits = torch.randn(n, vocab, requires_grad=True)
    logp = logits.log_softmax(-1)
    sampler = (logits.detach() + 0.5 * torch.randn(n, vocab)).log_softmax(-1)
    ids = sampler.topk(k, -1).indices
    labels = torch.multinomial(sampler.exp(), 1).squeeze(-1)
    q_head = sampler.gather(-1, ids)
    head, valid = logp.gather(-1, ids), torch.rand(n, k) > 0.1
    inputs = LossInputs(
        trainer_logprobs=logp.gather(-1, labels[:, None]).squeeze(-1),
        inference_logprobs=sampler.gather(-1, labels[:, None]).squeeze(-1),
        ref_logprobs=None,
        advantages=torch.randn(n),
        loss_mask=torch.rand(n) > 0.2,
        **_ragged(head, q_head, valid, torch.zeros(n, dtype=torch.bool)),
    )
    config = IcePopLossConfig(score_centering=True, score_centering_estimator="topk")
    actual = setup_rl_loss_fn(config).loss(inputs).loss
    expected = _main_icepop_score_centering(config, inputs, head, q_head, valid)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0]
    )


@pytest.mark.parametrize("config", WEIGHTINGS, ids=lambda c: c.type)
def test_sampled_estimator_is_unbiased(config):
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
        cfg = config.model_copy(update={"score_centering_estimator": "sampled", "kl_coeff": kl_coeff})
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
        loss = setup_rl_loss_fn(cfg).loss(inputs).loss / n
        return loss, torch.autograd.grad(loss, logits)[0]

    for kl_coeff in (0.0, 0.5):
        estimate, estimate_grad = run(draws, False, kl_coeff)
        exact, exact_grad = run(full, True, kl_coeff)
        torch.testing.assert_close(estimate, exact, atol=0.01, rtol=0)
        torch.testing.assert_close(estimate_grad, exact_grad, atol=0.01, rtol=0)
    torch.testing.assert_close(run(draws, False, 0.0)[1], torch.zeros(vocab), atol=0.02, rtol=0)


def test_score_centering_auto_centers_only_replayed_tokens():
    logits, logp, q, support = _support_case(replay=False)
    advantages = torch.ones(len(q))
    inputs = _support_inputs(logp, q, support, advantages)
    result = setup_rl_loss_fn(ReinforceLossConfig()).loss(inputs)
    uncentered = -(q * advantages * logp).sum()
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, logits, retain_graph=True)[0], torch.autograd.grad(uncentered, logits)[0]
    )
    assert bool((result.metrics["score_centering/uncentered"] == 1).all())


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    ref_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig(eps=10.0))
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

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig(eps=10.0))
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


def test_icepop_loss_masks_ratios_outside_inclusive_band():
    ratios = torch.tensor([0.1, 0.2, 1.0, 5.0, 10.0], device="cuda")
    trainer_logprobs = ratios.log().requires_grad_()
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.zeros_like(trainer_logprobs),
        ref_logprobs=None,
        advantages=torch.ones_like(trainer_logprobs),
        loss_mask=torch.ones_like(trainer_logprobs, dtype=torch.bool),
    )

    result = setup_rl_loss_fn(IcePopLossConfig()).loss(inputs)

    assert torch.isclose(result.loss, torch.tensor(-6.2, device="cuda"))
    assert torch.isclose(result.metrics["is_masked"], torch.tensor(0.4, device="cuda"))
    result.loss.backward()
    assert torch.allclose(trainer_logprobs.grad, torch.tensor([0.0, -0.2, -1.0, -5.0, 0.0], device="cuda"))


def test_icepop_loss_masks_extreme_ratio_without_nan():
    trainer_logprobs = torch.tensor([100.0], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.zeros_like(trainer_logprobs),
        ref_logprobs=None,
        advantages=torch.ones_like(trainer_logprobs),
        loss_mask=torch.ones_like(trainer_logprobs, dtype=torch.bool),
    )

    result = setup_rl_loss_fn(IcePopLossConfig()).loss(inputs)

    assert torch.equal(result.loss, torch.zeros_like(result.loss))
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    assert torch.equal(trainer_logprobs.grad, torch.zeros_like(trainer_logprobs.grad))


@pytest.mark.parametrize("config", [IPOLossConfig(), IcePopLossConfig()])
def test_ipo_icepop_match_original_on_finite_ratios(config):
    torch.manual_seed(23)
    trainer_logprobs = (-8 * torch.rand(128, device="cuda")).requires_grad_()
    inference_logprobs = -8 * torch.rand(128, device="cuda")
    advantages = torch.randn(128, device="cuda")
    loss_mask = torch.rand(128, device="cuda") > 0.3
    weights = torch.rand(128, device="cuda")
    inputs = LossInputs(trainer_logprobs, inference_logprobs, None, advantages, loss_mask, weights)

    result = setup_rl_loss_fn(config).loss(inputs)
    log_ratio = trainer_logprobs - inference_logprobs
    ratio = log_ratio.exp()
    if isinstance(config, IPOLossConfig):
        keep = loss_mask & ((trainer_logprobs.exp() - inference_logprobs.exp()).abs() <= config.eps)
        expected = (-(keep * config.adv_tau * advantages * ratio) * weights).sum()
    else:
        keep = (
            loss_mask
            & (log_ratio.detach() >= torch.tensor(config.ratio_low, device="cuda").log())
            & (log_ratio.detach() <= torch.tensor(config.ratio_high, device="cuda").log())
        )
        expected = (-(keep * config.adv_tau * advantages * ratio) * weights).sum()

    torch.testing.assert_close(result.loss, expected, rtol=1e-5, atol=1e-5)
    actual_grad = torch.autograd.grad(result.loss, trainer_logprobs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, trainer_logprobs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_ipo_excludes_masked_tokens_and_caps_accepted_extreme_ratio():
    trainer_logprobs = torch.tensor([-10.0, 0.0, float("nan")], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([-100.0, -100.0, float("nan")], device="cuda"),
        ref_logprobs=None,
        advantages=torch.ones(3, device="cuda"),
        loss_mask=torch.tensor([True, True, False], device="cuda"),
    )

    result = setup_rl_loss_fn(IPOLossConfig()).loss(inputs)

    torch.testing.assert_close(result.loss, torch.tensor(-1e4, device="cuda"))
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([-1e4, 0.0, 0.0], device="cuda"))


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

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig())
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

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig())
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

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig())
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

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig(eps=10.0))
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
    assert "is_masked" in metrics


def test_empty_components_keep_backward_valid():
    """A fully truncated distillation sample (stamped streams survive truncation
    as all-zero prefixes) must train as a zero-gradient no-op, not crash backward."""
    trainer_logprobs = [torch.randn(6, dtype=torch.float32, device="cuda", requires_grad=True)]
    inference_logprobs = [torch.zeros(6, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(6, dtype=torch.float32).cuda()]
    loss_mask = [torch.zeros(6, dtype=torch.bool).cuda()]
    rl_weights = [torch.zeros(6, dtype=torch.float32).cuda()]
    ce_weights = [torch.zeros(6, dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig())
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

    rl_loss_fn = setup_rl_loss_fn(IPOLossConfig(eps=10.0))
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
