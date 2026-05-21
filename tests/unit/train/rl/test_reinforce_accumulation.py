import torch

from prime_rl.configs.trainer import CustomLossConfig
from prime_rl.trainer.rl.loss import compute_loss, setup_loss_fn


def test_reinforce_accumulated_microbatch_gradients_match_single_batch_gradient():
    loss_fn = setup_loss_fn(
        CustomLossConfig(import_path="prime_rl.trainer.rl.loss.reinforce_loss_fn", kwargs={"kl_tau": 0.0})
    )
    seq_values = [[0.2, -0.4], [1.0], [-0.3, 0.7, 0.5]]
    advantage_values = [[1.0, 2.0], [-0.5], [1.5, -1.0, 0.25]]
    microbatch_indices = [[0], [1, 2]]
    loss_scale = sum(len(seq) for seq in seq_values)

    def make_tensors():
        trainer_logprobs = [torch.tensor(values, dtype=torch.float32, requires_grad=True) for values in seq_values]
        inference_logprobs = [torch.zeros_like(values) for values in trainer_logprobs]
        advantages = [torch.tensor(values, dtype=torch.float32) for values in advantage_values]
        loss_mask = [torch.ones_like(values, dtype=torch.bool) for values in trainer_logprobs]
        return trainer_logprobs, inference_logprobs, advantages, loss_mask

    actual_logprobs, actual_inference, actual_advantages, actual_masks = make_tensors()
    actual_loss = torch.zeros(())
    for indices in microbatch_indices:
        loss, _ = compute_loss(
            trainer_logprobs=[actual_logprobs[i] for i in indices],
            inference_logprobs=[actual_inference[i] for i in indices],
            teacher_logprobs=None,
            advantages=[actual_advantages[i] for i in indices],
            loss_mask=[actual_masks[i] for i in indices],
            loss_fn=loss_fn,
            loss_scale=loss_scale,
        )
        actual_loss = actual_loss + loss.detach()
        loss.backward()
    actual_grads = [logprobs.grad.clone() for logprobs in actual_logprobs]

    expected_logprobs, expected_inference, expected_advantages, expected_masks = make_tensors()
    expected_loss, _ = compute_loss(
        trainer_logprobs=expected_logprobs,
        inference_logprobs=expected_inference,
        teacher_logprobs=None,
        advantages=expected_advantages,
        loss_mask=expected_masks,
        loss_fn=loss_fn,
        loss_scale=loss_scale,
    )
    expected_loss.backward()
    expected_grads = [logprobs.grad.clone() for logprobs in expected_logprobs]

    torch.testing.assert_close(actual_loss, expected_loss.detach(), rtol=0, atol=1e-6)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
