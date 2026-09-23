from dataclasses import replace

import pytest
import torch

from prime_rl.configs.trainer import DistributionalIPOLossConfig, IPOLossConfig
from prime_rl.trainer.rl.loss import DistributionalIPOLoss, IPOLoss, LossInputs, setup_rl_loss_fn


@pytest.fixture
def make_inputs():
    def build(trainer_probs, sampler_probs, head_size=None):
        logits = torch.tensor(trainer_probs, dtype=torch.float64).log().requires_grad_()
        trainer_logp = logits.log_softmax(-1)
        sampler_logp = torch.tensor(sampler_probs, dtype=torch.float64).log()
        head_size = head_size or trainer_logp.shape[-1]
        inputs = LossInputs(
            trainer_logprobs=trainer_logp[:, 0],
            inference_logprobs=sampler_logp[:, 0],
            ref_logprobs=None,
            advantages=torch.ones(logits.shape[0], dtype=logits.dtype),
            loss_mask=torch.ones(logits.shape[0], dtype=torch.bool),
            trainer_topk_logprobs=trainer_logp[:, :head_size],
            sampler_topk_logprobs=sampler_logp[:, :head_size],
            topk_valid=torch.ones_like(trainer_logp[:, :head_size], dtype=torch.bool),
        )
        return inputs, logits

    return build


def test_distributional_ipo_catches_unsampled_movement(make_inputs):
    inputs, logits = make_inputs([[0.5, 0.49, 0.01]], [[0.5, 0.25, 0.25]])
    config = DistributionalIPOLossConfig(eps=0.2)
    loss_fn = setup_rl_loss_fn(config)
    assert isinstance(loss_fn, DistributionalIPOLoss)

    result = loss_fn.loss(inputs)
    sampled_result = IPOLoss(IPOLossConfig(eps=0.2)).loss(inputs)
    assert result.metrics["is_masked"].item() == 1
    assert sampled_result.metrics["is_masked"].item() == 0
    torch.testing.assert_close(result.metrics["total_variation"], torch.tensor(0.24, dtype=logits.dtype))
    result.loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_distributional_ipo_gate_includes_boundary_and_safely_rejects_extreme_ratios(make_inputs):
    inputs, _ = make_inputs([[0.5, 0.49, 0.01]], [[0.5, 0.25, 0.25]])
    distance = DistributionalIPOLoss(DistributionalIPOLossConfig()).loss(inputs).metrics["total_variation"].item()
    result = DistributionalIPOLoss(DistributionalIPOLossConfig(eps=distance)).loss(inputs)
    assert result.metrics["is_masked"].item() == 0

    sampled_logp = torch.tensor([1000.0], requires_grad=True)
    rejected = replace(inputs, trainer_logprobs=sampled_logp)
    result = DistributionalIPOLoss(DistributionalIPOLossConfig(eps=0.0)).loss(rejected)
    result.loss.backward()
    assert result.loss.item() == 0
    torch.testing.assert_close(sampled_logp.grad, torch.zeros_like(sampled_logp))


@pytest.mark.parametrize("head_size", [2, 4])
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_distributional_ipo_matches_categorical_objective_and_gradient(make_inputs, head_size, eps):
    inputs, logits = make_inputs([[0.4, 0.3, 0.2, 0.1]], [[0.5, 0.2, 0.15, 0.15]], head_size)
    inputs = replace(inputs, loss_weights=torch.tensor([0.7], dtype=logits.dtype))
    result = DistributionalIPOLoss(DistributionalIPOLossConfig(eps=eps, kl_tau=0.3)).loss(inputs)
    actual_grad = torch.autograd.grad(result.loss, logits, retain_graph=True)[0]

    trainer = logits.softmax(-1)[0]
    sampler = torch.tensor([0.5, 0.2, 0.15, 0.15], dtype=logits.dtype)
    if head_size < 4:
        trainer = torch.cat([trainer[:head_size], trainer[head_size:].sum().unsqueeze(0)])
        sampler = torch.cat([sampler[:head_size], sampler[head_size:].sum().unsqueeze(0)])
    penalty = (sampler * (trainer.log() - sampler.log()).square()).sum()
    pg = -trainer[0] / sampler[0] if eps == 1.0 else 0.0
    expected = 0.7 * (pg + 0.3 * penalty)
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.abs().sum() > 0  # The penalty still acts when the PG term is gated out.


def test_distributional_ipo_padding_and_nonmember_rows_do_not_change_gradient(make_inputs):
    inputs, logits = make_inputs([[0.4, 0.6], [0.5, 0.5]], [[0.5, 0.5], [0.6, 0.4]])
    inputs = replace(inputs, loss_mask=torch.tensor([True, False]))
    loss_fn = DistributionalIPOLoss(DistributionalIPOLossConfig(eps=1.0, kl_tau=0.2))
    reference = loss_fn.loss(inputs).loss
    reference_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]

    padding = torch.full((2, 1), float("nan"), dtype=logits.dtype)
    padded = replace(
        inputs,
        trainer_topk_logprobs=torch.cat([inputs.trainer_topk_logprobs, padding], -1),
        sampler_topk_logprobs=torch.cat([inputs.sampler_topk_logprobs, padding], -1),
        topk_valid=torch.tensor([[True, True, False], [False, False, False]]),
    )
    result = loss_fn.loss(padded)
    grad = torch.autograd.grad(result.loss, logits)[0]
    torch.testing.assert_close(result.loss, reference)
    torch.testing.assert_close(grad, reference_grad)
    torch.testing.assert_close(grad[1], torch.zeros_like(grad[1]))


def test_distributional_ipo_accounts_for_tail_mass_but_not_internal_tail_movement(make_inputs):
    inputs, _ = make_inputs([[0.5, 0.3, 0.1, 0.1]], [[0.5, 0.2, 0.15, 0.15]], head_size=2)
    loss_fn = DistributionalIPOLoss(DistributionalIPOLossConfig(eps=1.0, kl_tau=0.2))
    result = loss_fn.loss(inputs)
    assert result.metrics["total_variation"].item() == pytest.approx(0.1)

    tail_only, _ = make_inputs([[0.5, 0.2, 0.29, 0.01]], [[0.5, 0.2, 0.15, 0.15]], head_size=2)
    result = loss_fn.loss(tail_only)
    assert result.metrics["total_variation"].item() == pytest.approx(0.0, abs=1e-15)
    assert result.metrics["squared_log_ratio"].item() == pytest.approx(0.0, abs=1e-15)


def test_distributional_ipo_equal_policies_and_empty_loss_are_finite(make_inputs):
    inputs, logits = make_inputs([[0.5, 0.5]], [[0.5, 0.5]])
    loss_fn = DistributionalIPOLoss(DistributionalIPOLossConfig(kl_tau=0.3))
    result = loss_fn.loss(inputs)
    assert result.metrics["squared_log_ratio"].item() == 0
    torch.testing.assert_close(result.loss, IPOLoss(IPOLossConfig(kl_tau=0.3)).loss(inputs).loss)

    empty = replace(
        inputs, loss_mask=torch.zeros(1, dtype=torch.bool), topk_valid=torch.zeros((1, 2), dtype=torch.bool)
    )
    result = loss_fn.loss(empty)
    result.loss.backward()
    assert result.loss.item() == 0
    assert all(torch.isfinite(value) for value in result.metrics.values())
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("missing", ["trainer_topk_logprobs", "sampler_topk_logprobs", "topk_valid", "row"])
def test_distributional_ipo_rejects_missing_candidates(make_inputs, missing):
    inputs, _ = make_inputs([[0.5, 0.5]], [[0.5, 0.5]])
    update = {missing: None} if missing != "row" else {"topk_valid": torch.zeros((1, 2), dtype=torch.bool)}
    with pytest.raises(ValueError, match="requires.*candidate logprobs"):
        DistributionalIPOLoss(DistributionalIPOLossConfig()).loss(replace(inputs, **update))
