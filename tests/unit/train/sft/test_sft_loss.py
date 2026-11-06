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


def test_fused_lm_head_confident_targets_keep_loss_and_gradient():
    # Targets lead logsumexp of the others by ~23 to ~73, past the ~16.6 where fp32 rounds their loss and target
    # gradient to exactly 0; the last row is ignored
    vocab = 1000
    generator = torch.Generator().manual_seed(0)
    logits = 0.1 * torch.randn(4, vocab, generator=generator)
    target_ids = torch.tensor([3, 500, 997, 42])
    logits[torch.arange(3), target_ids[:3]] = torch.tensor([30.0, 50.0, 80.0])
    # One-hot hidden states make the head's logits exactly weight.T, and the weight gradient the logit gradient
    lm_head = FusedOutputLinear(4, vocab, chunk_size=2)
    lm_head.weight = torch.nn.Parameter(logits.t().contiguous())
    out = lm_head(torch.eye(4).unsqueeze(0), target_ids.masked_fill(torch.arange(4) == 3, IGNORE_INDEX).unsqueeze(0))
    out["loss"].backward()
    grad = lm_head.weight.grad.t()

    # fp64 reference: loss softplus(logsumexp_{j != target} - z_target), gradient softmax - onehot
    logits = logits[:3].double()
    target_logits = logits.gather(-1, target_ids[:3, None])
    nll = F.softplus(logits.scatter(-1, target_ids[:3, None], -torch.inf).logsumexp(-1, keepdim=True) - target_logits)
    expected_grad = torch.exp(logits - target_logits - nll).scatter_(-1, target_ids[:3, None], torch.expm1(-nll))

    assert out["loss"] > 0
    torch.testing.assert_close(out["loss"].double(), nll.sum(), rtol=1e-5, atol=0)
    torch.testing.assert_close(grad[:3].double(), expected_grad, rtol=1e-4, atol=0)
    assert torch.all(grad[3] == 0)
