import torch

from prime_rl.trainer.rl.loss import compute_ttt_prompt_loss


def test_compute_ttt_prompt_loss_uses_prompt_non_pad_tokens():
    trainer_logprobs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    input_ids = torch.tensor([[10, 11, 1, 12]])
    loss_mask = torch.tensor([[False, True, False, False]])

    loss, metrics = compute_ttt_prompt_loss(
        trainer_logprobs=trainer_logprobs,
        input_ids=input_ids,
        loss_mask=loss_mask,
        pad_token_id=1,
        loss_scale=2,
        weight=0.5,
    )

    assert torch.isclose(loss, torch.tensor(1.25))
    assert torch.equal(metrics["ttt_prompt_tokens"], torch.tensor([2.0]))
    assert torch.isclose(metrics["ttt_prompt_nll"], torch.tensor([2.5]))
