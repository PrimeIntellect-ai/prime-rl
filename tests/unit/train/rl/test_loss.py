import pytest
import torch

from prime_rl.configs.trainer import (
    CISPOLossConfig,
    CustomLossConfig,
    IcePopLossConfig,
    IPOLossConfig,
    IPOTISLossConfig,
    PPOLossConfig,
    ScoreCenteringLossConfig,
)
from prime_rl.trainer.rl.loss import (
    IcePopLoss,
    IPOTISLoss,
    LossInputs,
    LossOutputs,
    _mismatch_kl_from_log_ratio,
    compute_entropy,
    compute_loss,
    ref_kl_loss_fn,
    setup_rl_loss_fn,
)

pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("head_size", [2, 5])
def test_score_centering_matches_full_modeled_sampler_gradient(head_size):
    logits = torch.tensor([0.7, -0.4, 0.2, 0.1, -1.0], device="cuda", requires_grad=True)
    logp = logits.log_softmax(-1)
    q = torch.tensor([0.35, 0.3, 0.2, 0.1, 0.05], device="cuda")
    if head_size < len(q):
        q[head_size:] = (1 - q[:head_size].sum()) * logp[head_size:].detach().softmax(-1)
    # Include an action outside the head, and a masked token with no evidence.
    sampled = torch.tensor([0, 4, 2], device="cuda")
    mask = torch.tensor([True, True, False], device="cuda")
    advantage = torch.tensor([1.3, -0.7, float("nan")], device="cuda")
    weights = torch.tensor([0.4, 2.0, 0.0], device="cuda")
    head = logp[:head_size].expand(3, -1)
    valid = mask[:, None].expand_as(head)
    inputs = LossInputs(
        logp[sampled],
        q[sampled].log(),
        None,
        advantage,
        mask,
        weights,
        head,
        q[:head_size].log().expand_as(head),
        valid,
    )
    loss = setup_rl_loss_fn(ScoreCenteringLossConfig(topk=head_size)).loss(inputs).loss
    reference = (-advantage[mask] * weights[mask] * (logp[sampled[mask]] - (q * logp).sum())).sum()
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("eps,cap", [(0.1, 2.0), (1.0, 1.1)])
@pytest.mark.parametrize("topk", [None, 4])
@pytest.mark.parametrize(
    "config_cls,cap_field", [(IPOLossConfig, "max_importance_ratio"), (IPOTISLossConfig, "ratio_cap")]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ipo_score_centering_cancels_constant_advantage_drift(eps, cap, topk, config_cls, cap_field, device):
    logits = torch.tensor([0.4, -0.7, 0.1, -1.0], device=device, requires_grad=True)
    logp = logits.log_softmax(-1)
    q = torch.tensor([0.7, 0.1, 0.15, 0.05], device=device)
    # Enumerate every possible sampled action, weighted by its sampler probability.
    inputs = LossInputs(
        logp,
        q.log(),
        None,
        torch.ones_like(q),
        torch.ones_like(q, dtype=torch.bool),
        q,
        logp.expand(4, -1),
        q.log().expand(4, -1),
        torch.ones((4, 4), device=device, dtype=torch.bool),
    )
    config = config_cls(eps=eps, **{cap_field: cap})
    plain_loss = setup_rl_loss_fn(config)
    if config_cls is IPOTISLossConfig:
        assert isinstance(plain_loss, IPOTISLoss)
    plain = plain_loss.loss(inputs).loss
    centered = (
        setup_rl_loss_fn(config_cls(eps=eps, **{cap_field: cap}, score_centering=True, score_centering_topk=topk))
        .loss(inputs)
        .loss
    )
    drift = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    centered_drift = torch.autograd.grad(centered, logits)[0]
    assert drift.norm() > 1e-3
    torch.testing.assert_close(centered_drift, torch.zeros_like(logits), atol=2e-7, rtol=0)


@pytest.mark.parametrize("q_head,cap", [([0.45, 0.4], 2.0), ([0.6, 0.38], 1.1)])
@pytest.mark.parametrize(
    "config_cls,cap_field", [(IPOLossConfig, "max_importance_ratio"), (IPOTISLossConfig, "ratio_cap")]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ipo_score_centering_topk_matches_full_modeled_sampler(q_head, cap, config_cls, cap_field, device):
    logits = torch.tensor([1.5, 0.3, -1.0, -2.0, -3.0], device=device, requires_grad=True)
    logp = logits.log_softmax(-1)
    p = logp.exp().detach()
    q_head = torch.tensor(q_head, device=device)
    q = torch.cat([q_head, (1 - q_head.sum()) * p[2:] / p[2:].sum()])
    sampled = torch.tensor([0, 3, 4], device=device)
    mask = torch.tensor([True, True, False], device=device)
    advantage = torch.tensor([1.3, -0.7, float("nan")], device=device)
    weights = torch.tensor([0.4, 2.0, 0.0], device=device)
    head = torch.cat([logp[:2], logp.new_zeros(1)]).expand(3, -1)
    sampler = torch.cat([q_head.log(), q_head.new_zeros(1)]).expand(3, -1)
    valid = torch.tensor([[True, True, False]], device=device).expand(3, -1)
    inputs = LossInputs(logp[sampled], q[sampled].log(), None, advantage, mask, weights, head, sampler, valid)
    config = config_cls(eps=0.1, **{cap_field: cap}, score_centering=True, score_centering_topk=2)
    result = setup_rl_loss_fn(config).loss(inputs)
    actual = result.loss
    w = (p / q).clamp_max(cap) * ((p - q).abs() <= config.eps)
    center = ((q * w).detach() * logp).sum()
    correction_l1 = torch.autograd.grad(center, logits, retain_graph=True)[0].abs().sum()
    torch.testing.assert_close(
        result.metrics["score_centering/logit_correction_l1"], correction_l1.expand(2), atol=2e-7, rtol=2e-6
    )
    expected = (-advantage[mask] * weights[mask] * (w[sampled[mask]] * logp[sampled[mask]] - center)).sum()
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=2e-7,
        rtol=2e-6,
    )


def test_ipo_score_centering_rejects_unproven_tail_mask():
    logp = torch.tensor([0.4, 0.2, 0.25, 0.15], device="cuda").log().requires_grad_()
    q = torch.tensor([0.6, 0.35, 0.03125, 0.01875], device="cuda")
    inputs = LossInputs(
        logp[:1],
        q[:1].log(),
        None,
        torch.ones(1, device="cuda"),
        torch.ones(1, device="cuda", dtype=torch.bool),
        trainer_topk_logprobs=logp[:2].unsqueeze(0),
        sampler_topk_logprobs=q[:2].log().unsqueeze(0),
        topk_valid=torch.ones((1, 2), device="cuda", dtype=torch.bool),
    )
    config = IPOLossConfig(eps=0.1, score_centering=True, score_centering_topk=2)
    with pytest.raises(ValueError, match="tail may cross the trust region"):
        setup_rl_loss_fn(config).loss(inputs)


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

    result = IcePopLoss(IcePopLossConfig()).loss(inputs)

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
    log_ratio = torch.tensor([1e-4])
    torch.testing.assert_close(_mismatch_kl_from_log_ratio(log_ratio), torch.tensor([5e-9]), rtol=1e-3, atol=0)
    large_ratios = torch.tensor([0.0, 1.0, 60.0, 70.0, 1000.0, 1e30, 1e31])
    mismatch = _mismatch_kl_from_log_ratio(large_ratios)
    assert torch.isfinite(mismatch).all()
    assert (mismatch[1:] >= mismatch[:-1]).all()
    torch.testing.assert_close(mismatch[-4:], torch.full((4,), 1e30), rtol=1e-6, atol=0)


def test_ppo_clips_by_advantage_sign_and_keeps_unclipped_gradients():
    ratios = torch.tensor([0.5, 0.9, 1.0, 1.1, 2.0, 2.0], device="cuda")
    trainer_logprobs = (-10 + ratios.log()).requires_grad_()
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.full_like(trainer_logprobs, -10),
        ref_logprobs=None,
        advantages=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], device="cuda"),
        loss_mask=torch.ones(6, dtype=torch.bool, device="cuda"),
    )

    result = setup_rl_loss_fn(PPOLossConfig()).loss(inputs)

    torch.testing.assert_close(result.loss, torch.tensor(-2.7, device="cuda"))
    torch.testing.assert_close(result.metrics["is_clipped"], torch.tensor(1 / 6, device="cuda"))
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([-0.5, -0.9, -1.0, -1.1, 0.0, 2.0], device="cuda"))


def test_ppo_caps_unbounded_negative_advantage_ratio():
    trainer_logprobs = torch.tensor([-10.0, float("nan")], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([-100.0, float("nan")], device="cuda"),
        ref_logprobs=None,
        advantages=torch.tensor([-1.0, 1.0], device="cuda"),
        loss_mask=torch.tensor([True, False], device="cuda"),
    )

    result = setup_rl_loss_fn(PPOLossConfig()).loss(inputs)

    torch.testing.assert_close(result.loss, torch.tensor(1e4, device="cuda"))
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([1e4, 0.0], device="cuda"))


def test_cispo_clips_detached_weight_without_dropping_gradients():
    ratios = torch.tensor([0.1, 1.0, 5.0, 10.0], device="cuda")
    trainer_logprobs = (-10 + ratios.log()).requires_grad_()
    advantages = torch.tensor([1.0, -1.0, 1.0, 1.0], device="cuda")
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.full_like(trainer_logprobs, -10),
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=torch.ones(4, dtype=torch.bool, device="cuda"),
    )

    result = setup_rl_loss_fn(CISPOLossConfig()).loss(inputs)

    expected_weight = torch.tensor([0.1, 1.0, 5.0, 5.0], device="cuda")
    torch.testing.assert_close(result.loss, -(expected_weight * advantages * trainer_logprobs.detach()).sum())
    torch.testing.assert_close(result.metrics["is_clipped"], torch.tensor(0.25, device="cuda"))
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([-0.1, 1.0, -5.0, -5.0], device="cuda"))


def test_cispo_handles_extreme_ratio_and_optional_lower_clip():
    trainer_logprobs = torch.tensor([-10.0, 0.0, float("nan")], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor(
            [-10.0 - torch.log(torch.tensor(0.1)).item(), -100.0, float("nan")], device="cuda"
        ),
        ref_logprobs=None,
        advantages=torch.ones(3, device="cuda"),
        loss_mask=torch.tensor([True, True, False], device="cuda"),
    )

    result = setup_rl_loss_fn(CISPOLossConfig(ratio_low=0.2)).loss(inputs)

    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([-0.2, -5.0, 0.0], device="cuda"))


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


@pytest.mark.parametrize("masked_value", [-1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("config", [IPOLossConfig(), ScoreCenteringLossConfig(topk=2)])
def test_empty_components_keep_backward_valid(masked_value, config):
    """A fully truncated distillation sample (stamped streams survive truncation
    as all-zero prefixes) must train as a zero-gradient no-op, not crash backward."""
    trainer_logprobs = [torch.full((6,), masked_value, dtype=torch.float32, requires_grad=True)]
    inference_logprobs = [torch.zeros(6, dtype=torch.float32)]
    advantages = [torch.zeros(6, dtype=torch.float32)]
    loss_mask = [torch.zeros(6, dtype=torch.bool)]
    rl_weights = [torch.zeros(6, dtype=torch.float32)]
    ce_weights = [torch.zeros(6, dtype=torch.float32)]

    rl_loss_fn = setup_rl_loss_fn(config)
    empty_loss = rl_loss_fn.loss(
        LossInputs(trainer_logprobs[0], inference_logprobs[0], None, advantages[0], loss_mask[0])
    ).loss
    assert torch.equal(empty_loss, torch.zeros_like(empty_loss))
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


@pytest.mark.parametrize("config", [IPOLossConfig(), ScoreCenteringLossConfig(topk=2)])
def test_compute_loss_ignores_nonfinite_masked_logprobs(config):
    trainer = torch.tensor([-1.0, float("nan")], requires_grad=True)
    sampler = torch.tensor([-1.0, float("nan")])
    advantage = torch.tensor([1.0, float("nan")])
    mask = torch.tensor([True, False])
    head = torch.tensor([[-1.0, -2.0], [float("nan"), float("nan")]], requires_grad=True)
    sampler_head = head.detach().clone()
    valid = torch.tensor([[True, True], [False, False]])
    loss_fn = setup_rl_loss_fn(config)
    expected = loss_fn.loss(LossInputs(trainer, sampler, None, advantage, mask, None, head, sampler_head, valid)).loss
    loss, _ = compute_loss(
        trainer_logprobs=[trainer],
        inference_logprobs=[sampler],
        ref_logprobs=None,
        advantages=[advantage],
        loss_mask=[mask],
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=loss_fn,
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        trainer_topk_logprobs=[head],
        sampler_topk_logprobs=[sampler_head],
        topk_valid=[valid],
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(trainer.grad, torch.tensor([-1.0, 0.0]))
    if config.type == "score_centering":
        assert torch.isfinite(head.grad).all()
        assert (head.grad[1] == 0).all()


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
