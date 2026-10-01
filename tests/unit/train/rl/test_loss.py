import math

import pytest
import torch

from prime_rl.configs.trainer import (
    CISPOLossConfig,
    CustomLossConfig,
    IcePopLossConfig,
    IPOLossConfig,
    IPOV2LossConfig,
    PPOLossConfig,
)
from prime_rl.trainer.rl.loss import (
    IcePopLoss,
    LossInputs,
    LossOutputs,
    _mismatch_kl_from_log_ratio,
    compute_entropy,
    compute_loss,
    ref_kl_loss_fn,
    setup_rl_loss_fn,
)

pytestmark = [pytest.mark.gpu]


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


@pytest.mark.parametrize("config", [IPOLossConfig(kl_tau=0.01), IcePopLossConfig()])
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
        expected = (
            (-(keep * config.adv_tau * advantages * ratio) + loss_mask * config.kl_tau * log_ratio.square()) * weights
        ).sum()
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


@pytest.mark.parametrize(
    ("inference_logprob", "max_importance_ratio", "expected_ratio"),
    [(-30.0, None, math.exp(20)), (-100.0, 1e4, 1e4)],
)
def test_ipo_excludes_masked_tokens_with_optional_ratio_cap(inference_logprob, max_importance_ratio, expected_ratio):
    trainer_logprobs = torch.tensor([-10.0, 0.0, float("nan")], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([inference_logprob, -100.0, float("nan")], device="cuda"),
        ref_logprobs=None,
        advantages=torch.ones(3, device="cuda"),
        loss_mask=torch.tensor([True, True, False], device="cuda"),
    )

    result = setup_rl_loss_fn(IPOLossConfig(max_importance_ratio=max_importance_ratio)).loss(inputs)

    torch.testing.assert_close(result.loss, torch.tensor(-expected_ratio, device="cuda"))
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([-expected_ratio, 0.0, 0.0], device="cuda"))


@pytest.mark.parametrize("loss_type", ["ipo", "ipo_v2"])
@pytest.mark.parametrize(
    ("max_kl_log_ratio", "expected_loss", "expected_gradient"), [(None, 9.0, 6.0), (2.0, 4.0, 0.0)]
)
def test_optional_kl_log_ratio_bound(loss_type, max_kl_log_ratio, expected_loss, expected_gradient):
    trainer_logprobs = torch.tensor([-10.0], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([-13.0], device="cuda"),
        ref_logprobs=None,
        advantages=torch.zeros(1, device="cuda"),
        loss_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
        entropy=torch.zeros(1, device="cuda"),
    )
    config = (
        IPOLossConfig(kl_tau=1.0, max_kl_log_ratio=max_kl_log_ratio)
        if loss_type == "ipo"
        else IPOV2LossConfig(kl_tau=1.0, max_kl_log_ratio=max_kl_log_ratio)
    )

    result = setup_rl_loss_fn(config).loss(inputs)
    torch.testing.assert_close(result.loss, torch.tensor(expected_loss, device="cuda"))
    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, torch.tensor([expected_gradient], device="cuda"))


@pytest.mark.parametrize("ratio_cap", [5.0, None])
def test_ipo_v2_optional_cap_gate_and_entropy_gradient(ratio_cap):
    ratios = torch.tensor([1.0, 4.0, 5.0, 20.0, 100.0], device="cuda")
    trainer_logprobs = torch.cat(
        [-12 + ratios.log(), torch.tensor([math.log(0.6), float("nan")], device="cuda")]
    ).requires_grad_()
    inference_logprobs = torch.tensor([-12.0] * 5 + [math.log(0.1), float("nan")], device="cuda")
    entropy = torch.arange(7, dtype=torch.float32, device="cuda").mul(0.1).requires_grad_()
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0], device="cuda")
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=torch.tensor([True] * 6 + [False], device="cuda"),
        entropy=entropy,
    )

    loss_fn = setup_rl_loss_fn(IPOV2LossConfig(ratio_cap=ratio_cap, entropy_lambda_init=0.2))
    result = loss_fn.loss(inputs)
    if ratio_cap is None:
        surrogate = ratios
        expected_gradient = torch.tensor([-1.0, 4.0, -5.0, 20.0, -100.0, 0.0, 0.0], device="cuda")
        expected_saturated = 0.0
    else:
        surrogate = torch.tensor([1.0, 4.0, 5.0, 5 * (1 + math.log(4)), 5 * (1 + math.log(20))], device="cuda")
        expected_gradient = torch.tensor([-1.0, 4.0, -5.0, 5.0, -5.0, 0.0, 0.0], device="cuda")
        expected_saturated = 2 / 5
    expected_loss = -(advantages[:5] * surrogate).sum() - 0.2 * entropy[:6].sum()
    torch.testing.assert_close(result.loss, expected_loss)
    torch.testing.assert_close(result.metrics["is_masked"], torch.tensor(1 / 6, device="cuda"))
    torch.testing.assert_close(result.metrics["ratio_saturated"], torch.tensor(expected_saturated, device="cuda"))

    result.loss.backward()
    torch.testing.assert_close(trainer_logprobs.grad, expected_gradient)
    torch.testing.assert_close(entropy.grad, torch.tensor([-0.2] * 6 + [0.0], device="cuda"))


def test_ipo_v2_entropy_multiplier_update():
    disabled_loss_fn = setup_rl_loss_fn(IPOV2LossConfig())
    disabled_loss_fn.update_entropy_lambda(0.05)
    assert disabled_loss_fn.entropy_lambda == 0.0

    loss_fn = setup_rl_loss_fn(IPOV2LossConfig(entropy_floor=0.15, entropy_lambda_init=0.01, entropy_lambda_lr=0.1))
    loss_fn.update_entropy_lambda(0.05)
    assert loss_fn.entropy_lambda == pytest.approx(0.02)
    loss_fn.update_entropy_lambda(0.35)
    assert loss_fn.entropy_lambda == 0.0


def test_ref_kl_loss_stays_finite_with_extreme_ratios_and_masked_nan():
    trainer_logprobs = torch.tensor([-10.0, -10.0, float("nan")], device="cuda", requires_grad=True)
    inputs = LossInputs(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=torch.tensor([-30.0, 0.0, float("nan")], device="cuda"),
        ref_logprobs=torch.tensor([-1.0, -1.0, float("nan")], device="cuda"),
        advantages=torch.zeros(3, device="cuda"),
        loss_mask=torch.tensor([True, True, False], device="cuda"),
    )

    result = ref_kl_loss_fn(inputs)

    assert torch.isfinite(result.loss)
    torch.testing.assert_close(result.loss, torch.tensor(-9 * math.exp(20) + 0.5, device="cuda"))
    assert all(torch.isfinite(value) for value in result.metrics.values())
    result.loss.backward()
    torch.testing.assert_close(
        trainer_logprobs.grad, torch.tensor([-9 * math.exp(20) + 0.04, -0.02, 0.0], device="cuda")
    )


def test_mismatch_kl_retains_small_positive_values():
    log_ratio = torch.tensor([1e-4], device="cuda")
    torch.testing.assert_close(
        _mismatch_kl_from_log_ratio(log_ratio), torch.tensor([5e-9], device="cuda"), rtol=1e-3, atol=0
    )


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
