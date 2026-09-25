import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, FusedOutputLinear


def _logprob_sft_loss(lm_head: FusedOutputLinear, hidden, target_ids, loss_mask):
    """SFT loss through the per-token logprob path: negative target logprob summed over the loss mask."""
    out = lm_head(hidden, target_ids, temperature=torch.ones_like(target_ids, dtype=torch.float32))
    return -out["logprobs"][loss_mask].sum()


def _cross_entropy_sft_loss(lm_head: FusedOutputLinear, hidden, target_ids, loss_mask):
    """SFT loss through the fused cross-entropy path, which computes its gradients in forward."""
    lm_head.return_loss = True
    return lm_head(hidden, target_ids.masked_fill(~loss_mask, IGNORE_INDEX))["loss"]


@pytest.mark.parametrize("sft_loss", [_logprob_sft_loss, _cross_entropy_sft_loss], ids=["logprobs", "cross_entropy"])
@pytest.mark.parametrize("chunk_size", [1, 5, 64])
def test_fused_lm_head_sft_loss_matches_cross_entropy(sft_loss, chunk_size):
    torch.manual_seed(0)
    b, s, h, v = 2, 7, 8, 37
    hidden = torch.randn(b, s, h, requires_grad=True)
    weight = torch.randn(v, h, requires_grad=True)
    target_ids = torch.randint(0, v, (b, s))
    loss_mask = torch.rand(b, s) > 0.3
    upstream = 0.25  # stands in for the 1 / grad_accum_steps scaling in the SFT loop

    reference = F.cross_entropy((hidden @ weight.t()).view(-1, v), target_ids.view(-1), reduction="none")
    reference = reference[loss_mask.view(-1)].sum()
    (reference * upstream).backward()

    lm_head = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    lm_head.weight = torch.nn.Parameter(weight.detach().clone())
    hidden_fused = hidden.detach().clone().requires_grad_(True)
    loss = sft_loss(lm_head, hidden_fused, target_ids, loss_mask)
    (loss * upstream).backward()

    torch.testing.assert_close(loss, reference.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden_fused.grad, hidden.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lm_head.weight.grad, weight.grad, rtol=1e-5, atol=1e-5)
