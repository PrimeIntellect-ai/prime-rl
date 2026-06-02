import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_loss_fns

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        rl_loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        rl_loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_loss_fns_with_custom_config():
    """Test setup_loss_fns with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    loss_fns = setup_loss_fns(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        teacher_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = loss_fns["rl"](inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_sft_loss_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        rl_loss_scale=2,
        training_mode="sft",
    )

    # loss = -sum(masked logprobs) / rl_loss_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics


def test_sft_loss_override_uses_masked_nll_with_default_loss_config():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.ones(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        rl_loss_scale=2,
        training_mode="sft",
    )

    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_default_loss_fn_uses_separate_echo_loss_scale():
    trainer_logprobs = [torch.tensor([-0.2, -0.4], dtype=torch.float32, device="cuda", requires_grad=True)]
    inference_logprobs = [trainer_logprobs[0].detach().clone()]
    advantages = [torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")]
    loss_mask = [torch.tensor([True, True], dtype=torch.bool, device="cuda")]
    echo_mask = [torch.tensor([False, True], dtype=torch.bool, device="cuda")]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0, dppo_mask_low=10.0, kl_tau=0.0))
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        rl_loss_scale=1,
        echo_loss_scale=4,
        echo_mask=echo_mask,
    )

    # RL term: -adv * ratio = -1.0. Echo term: -alpha * logprob / 4 = 0.2.
    assert torch.isclose(loss, torch.tensor(-0.8, device=loss.device), atol=1e-6)
    assert metrics["echo_token_count"].item() == 1
    loss.backward()
    assert torch.isclose(trainer_logprobs[0].grad[1], torch.tensor(-0.5, device="cuda"), atol=1e-6)


@pytest.mark.parametrize("training_mode", ["sft", "opd"])
def test_echo_rejected_for_non_rl_modes(training_mode):
    trainer_logprobs = [torch.tensor([-0.1, -0.2], dtype=torch.float32, device="cuda")]
    inference_logprobs = [torch.zeros(2, dtype=torch.float32, device="cuda")]
    teacher_logprobs = [torch.zeros(2, dtype=torch.float32, device="cuda")] if training_mode == "opd" else None
    advantages = [torch.ones(2, dtype=torch.float32, device="cuda")]
    loss_mask = [torch.ones(2, dtype=torch.bool, device="cuda")]
    echo_mask = [torch.tensor([False, True], dtype=torch.bool, device="cuda")]

    with pytest.raises(ValueError, match="Echo is only supported"):
        compute_loss(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=inference_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            loss_fns=setup_loss_fns(DefaultLossConfig()),
            rl_loss_scale=1,
            training_mode=training_mode,
            echo_mask=echo_mask,
        )


def test_echo_rejected_for_custom_rl_loss():
    loss_fns = setup_loss_fns(
        CustomLossConfig(
            import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
            kwargs={"multiplier": 2.0},
        )
    )
    inputs = LossInputs(
        trainer_logprobs=torch.randn(2, dtype=torch.float32, device="cuda"),
        inference_logprobs=torch.randn(2, dtype=torch.float32, device="cuda"),
        teacher_logprobs=None,
        advantages=torch.ones(2, dtype=torch.float32, device="cuda"),
        loss_mask=torch.ones(2, dtype=torch.bool, device="cuda"),
        echo_mask=torch.tensor([False, True], dtype=torch.bool, device="cuda"),
    )

    with pytest.raises(ValueError, match="Echo is only supported with the default RL loss"):
        loss_fns["rl"](inputs)


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
