import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_rl_loss_fn
from prime_rl.transport.types import LOSS_CORE_CE, LOSS_CORE_RL, LOSS_CORE_TEACHER_KL

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_core_ids=None,
        loss_weights=None,
        rl_loss_fn=rl_loss_fn,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_core_ids=None,
        loss_weights=None,
        rl_loss_fn=rl_loss_fn,
        loss_scale=1.0,
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
        teacher_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = rl_loss_fn(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_ce_core_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    loss_core_ids = [torch.full((3,), LOSS_CORE_CE, dtype=torch.long).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_core_ids=loss_core_ids,
        loss_weights=None,
        rl_loss_fn=rl_loss_fn,
        loss_scale=2,
    )

    # loss = -sum(masked logprobs) / loss_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_ce_core_applies_loss_weights():
    """ECHO-style routing: weighted CE on observation tokens."""
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]
    loss_core_ids = [torch.full((3,), LOSS_CORE_CE, dtype=torch.long).cuda()]
    loss_weights = [torch.tensor([0.1, 1.0, 0.1], dtype=torch.float32).cuda()]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_core_ids=loss_core_ids,
        loss_weights=loss_weights,
        rl_loss_fn=rl_loss_fn,
        loss_scale=1,
    )

    # loss = 0.1 * (0.1 + 0.2) = 0.03
    assert torch.isclose(loss, torch.tensor(0.03, device=loss.device), atol=1e-6)


def test_routed_all_rl_matches_unrouted():
    """An explicit all-RL core routing must equal the cores=None hot path."""
    torch.manual_seed(0)
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda()]
    loss_mask = [torch.rand(50).cuda() > 0.3]

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig())
    kwargs = dict(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_weights=None,
        rl_loss_fn=rl_loss_fn,
        loss_scale=1,
    )
    loss_unrouted, _ = compute_loss(loss_core_ids=None, **kwargs)
    loss_routed, _ = compute_loss(loss_core_ids=[torch.full((50,), LOSS_CORE_RL, dtype=torch.long).cuda()], **kwargs)

    assert torch.equal(loss_unrouted, loss_routed)


def test_mixed_cores_in_one_sequence():
    """ECHO-shaped sequence: RL on action tokens, weighted CE on observation tokens."""
    n = 12
    torch.manual_seed(1)
    trainer_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(n, dtype=torch.float32).cuda()]
    advantages = [torch.randn(n).cuda()]
    loss_mask = [torch.ones(n, dtype=torch.bool).cuda()]
    cores = torch.full((n,), LOSS_CORE_RL, dtype=torch.long)
    cores[4:8] = LOSS_CORE_CE
    cores[8:] = LOSS_CORE_TEACHER_KL

    rl_loss_fn = setup_rl_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_core_ids=[cores.cuda()],
        loss_weights=None,
        rl_loss_fn=rl_loss_fn,
        loss_scale=1,
    )

    assert loss.shape == ()
    assert "nll" in metrics
    assert "teacher_kl" in metrics
    assert "is_masked" in metrics


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
