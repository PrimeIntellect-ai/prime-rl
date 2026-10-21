import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX, FusedOutputLinear, VanillaOutputLinear

B, S, H, V = 2, 7, 8, 37
UPSTREAM = 0.25  # stands in for the 1 / grad_accum_steps scaling in the SFT loop

HEADS = {
    "fused": lambda chunk_size: FusedOutputLinear(H, V, chunk_size=chunk_size),
    "vanilla": lambda chunk_size: VanillaOutputLinear(H, V),
}


def _reference_and_grads(hidden, weight, target_ids, loss_mask):
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    logits = hidden @ weight.t()
    loss = F.cross_entropy(logits.view(-1, V), target_ids.view(-1), reduction="none")[loss_mask.view(-1)].sum()
    (loss * UPSTREAM).backward()
    return loss.detach(), hidden.grad, weight.grad


def _inputs():
    torch.manual_seed(0)
    hidden = torch.randn(B, S, H)
    weight = torch.randn(V, H)
    target_ids = torch.randint(0, V, (B, S))
    loss_mask = torch.rand(B, S) > 0.3
    return hidden, weight, target_ids, loss_mask


@pytest.mark.parametrize("head", list(HEADS))
@pytest.mark.parametrize("chunk_size", [1, 5, 64])
def test_lm_head_labels_without_temperature_returns_summed_cross_entropy(head, chunk_size):
    hidden, weight, target_ids, loss_mask = _inputs()
    reference, ref_hidden_grad, ref_weight_grad = _reference_and_grads(hidden, weight, target_ids, loss_mask)

    lm_head = HEADS[head](chunk_size)
    lm_head.weight = torch.nn.Parameter(weight.clone())
    hidden = hidden.clone().requires_grad_(True)
    out = lm_head(hidden, target_ids.masked_fill(~loss_mask, IGNORE_INDEX))
    assert set(out) == {"loss"}
    (out["loss"] * UPSTREAM).backward()

    torch.testing.assert_close(out["loss"], reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, ref_hidden_grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lm_head.weight.grad, ref_weight_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 5, 64])
def test_fused_lm_head_with_temperature_keeps_logprob_path(chunk_size):
    hidden, weight, target_ids, loss_mask = _inputs()
    reference, ref_hidden_grad, ref_weight_grad = _reference_and_grads(hidden, weight, target_ids, loss_mask)

    lm_head = FusedOutputLinear(H, V, chunk_size=chunk_size)
    lm_head.weight = torch.nn.Parameter(weight.clone())
    hidden = hidden.clone().requires_grad_(True)
    out = lm_head(hidden, target_ids, temperature=torch.ones(B, S))
    assert set(out) == {"logprobs", "entropy", "topk_logprobs"}
    assert out["topk_logprobs"] is None
    loss = -out["logprobs"][loss_mask].sum()
    (loss * UPSTREAM).backward()

    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, ref_hidden_grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lm_head.weight.grad, ref_weight_grad, rtol=1e-5, atol=1e-5)
