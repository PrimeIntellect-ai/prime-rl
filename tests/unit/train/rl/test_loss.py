import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig
from prime_rl.trainer.rl.loss import (
    LossChannel,
    LossInputs,
    LossOutputs,
    ce_loss_fn,
    compute_entropy,
    compute_loss,
    setup_rl_loss_fn,
)

pytestmark = [pytest.mark.gpu]


def _loss_fns():
    return {
        "rl": setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0)),
        "ce": ce_loss_fn,
    }


def test_rl_loss_channel_returns_scalar():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32, device="cuda")]
    inference_logprobs = [torch.randn(50, dtype=torch.float32, device="cuda")]
    advantages = [torch.randn(50, device="cuda")]
    mask = [torch.ones(50, dtype=torch.bool, device="cuda")]

    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        channels=[LossChannel(loss="rl", advantages=advantages, mask=mask)],
        loss_fns=_loss_fns(),
        loss_scales={"rl": 1, "ce": 1},
    )

    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32, device="cuda")
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_rl_loss_fn_with_custom_config():
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    rl_loss_fn = setup_rl_loss_fn(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32, device="cuda"),
        inference_logprobs=torch.randn(50, dtype=torch.float32, device="cuda"),
        advantages=torch.randn(50, device="cuda"),
        loss_mask=torch.ones(50, dtype=torch.bool, device="cuda"),
    )

    result = rl_loss_fn(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_ce_channel_uses_advantages_as_weights():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32, device="cuda")]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32, device="cuda")]
    values = [torch.tensor([0.1, 0.0, 0.1], dtype=torch.float32, device="cuda")]
    mask = [torch.tensor([True, False, True], dtype=torch.bool, device="cuda")]

    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        channels=[LossChannel(loss="ce", advantages=values, mask=mask)],
        loss_fns=_loss_fns(),
        loss_scales={"rl": 1, "ce": 1},
    )

    assert torch.isclose(loss, torch.tensor(0.03, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "ce_weight" in metrics


def test_overlapping_channels_sum():
    torch.manual_seed(2)
    n = 8
    trainer_logprobs = [torch.randn(n, dtype=torch.float32, device="cuda")]
    inference_logprobs = [torch.randn(n, dtype=torch.float32, device="cuda")]
    rl_values = [torch.randn(n, device="cuda")]
    ce_values = [torch.full((n,), 0.5, dtype=torch.float32, device="cuda")]
    mask = [torch.ones(n, dtype=torch.bool, device="cuda")]
    loss_fns = _loss_fns()
    scales = {"rl": n, "ce": n}

    rl_only, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        channels=[LossChannel(loss="rl", advantages=rl_values, mask=mask)],
        loss_fns=loss_fns,
        loss_scales=scales,
    )
    ce_only, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        channels=[LossChannel(loss="ce", advantages=ce_values, mask=mask)],
        loss_fns=loss_fns,
        loss_scales=scales,
    )
    both, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        channels=[
            LossChannel(loss="rl", advantages=rl_values, mask=mask),
            LossChannel(loss="ce", advantages=ce_values, mask=mask),
        ],
        loss_fns=loss_fns,
        loss_scales=scales,
    )

    assert torch.isclose(both, rl_only + ce_only, atol=1e-6)


def test_empty_channels_keep_backward_valid():
    trainer_logprobs = [torch.randn(6, dtype=torch.float32, device="cuda", requires_grad=True)]
    inference_logprobs = [torch.zeros(6, dtype=torch.float32, device="cuda")]
    values = [torch.zeros(6, dtype=torch.float32, device="cuda")]
    mask = [torch.zeros(6, dtype=torch.bool, device="cuda")]

    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        channels=[LossChannel(loss="ce", advantages=values, mask=mask)],
        loss_fns=_loss_fns(),
        loss_scales={"rl": 1, "ce": 1},
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert trainer_logprobs[0].grad is not None
    assert torch.equal(trainer_logprobs[0].grad, torch.zeros_like(trainer_logprobs[0].grad))


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(loss=loss, metrics={"custom_metric": loss.detach()})
