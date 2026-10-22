import pytest
import torch

from prime_rl.inference.vllm.sampling_order import apply_top_p_then_top_k, patch_top_p_then_top_k


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_full_nucleus_then_cap(device):
    probs = torch.tensor([[0.50, 0.15, 0.10, 0.03] + [0.022] * 10], device=device)
    logits = probs.log()
    k = torch.tensor([4], device=device)
    p = torch.tensor([0.95], device=device)
    result = apply_top_p_then_top_k(logits.clone(), k, p)
    assert torch.isfinite(result).nonzero()[:, 1].tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(result.softmax(-1)[0, :4], probs[0, :4] / 0.78)
    # A nucleus that fits inside k is unchanged by the cap.
    p = torch.tensor([0.6], device=device)
    result = apply_top_p_then_top_k(logits.clone(), k, p)
    assert torch.isfinite(result).nonzero()[:, 1].tolist() == [0, 1]
    # The hard cap also applies when every candidate has the same score.
    tied = apply_top_p_then_top_k(torch.zeros(2, 8, device=device), torch.tensor([2, 4], device=device), None)
    assert torch.isfinite(tied).sum(-1).tolist() == [2, 4]
    assert torch.isfinite(tied[0]).nonzero().flatten().tolist() == [6, 7]
    assert apply_top_p_then_top_k(logits, None, None) is logits


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_sampling_order_matches_independent_nucleus_reference(device):
    generator = torch.Generator().manual_seed(4176)
    logits = torch.randn(16, 257, generator=generator).to(device)
    k = torch.tensor([1, 2, 16, 128] * 4, device=device)
    p = torch.tensor([0.01, 0.5, 0.95, 1.0] * 4, device=device)
    result = apply_top_p_then_top_k(logits.clone(), k, p)
    for row in range(len(logits)):
        order = logits[row].argsort(descending=True)
        cumulative = logits[row].softmax(-1)[order].cumsum(0)
        size = min(int((cumulative < p[row]).sum()) + 1, int(k[row]), len(order))
        expected = torch.zeros_like(logits[row], dtype=torch.bool)
        expected[order[:size]] = True
        assert torch.equal(torch.isfinite(result[row]), expected)
        torch.testing.assert_close(result[row, expected], logits[row, expected])
    assert torch.isfinite(apply_top_p_then_top_k(logits.clone(), None, torch.ones(16, device=device))).all()


def test_patch_routes_v2_sampling_and_preserves_mask_probabilities(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.worker.gpu.sample import sampler, states

    for module in (topk_topp_sampler, sampler, states):
        monkeypatch.setattr(module, "apply_top_k_top_p", module.apply_top_k_top_p)
    patch_top_p_then_top_k()
    for module in (topk_topp_sampler, sampler, states):
        assert module.apply_top_k_top_p is apply_top_p_then_top_k
    logits = torch.tensor([[0.5, 0.15, 0.1, 0.03] + [0.022] * 10]).log()
    processed = states.apply_top_k_top_p(logits, torch.tensor([4]), torch.tensor([0.95]))
    support = torch.isfinite(processed)
    logprobs = processed.log_softmax(-1)
    assert support.sum() == 4
    torch.testing.assert_close(logprobs[support].exp().sum(), torch.tensor(1.0))


@pytest.mark.gpu
def test_final_support_capture_on_gpu():
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

    logits = torch.tensor([[0.5, 0.15, 0.1, 0.03] + [0.022] * 10], device="cuda").log()
    processed = apply_top_p_then_top_k(logits, torch.tensor([4], device="cuda"), torch.tensor([0.95], device="cuda"))
    masks = SamplingMaskTensors.from_logits(
        processed,
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        max_num_kept=4,
    )
    assert masks.counts.tolist() == [4]
    ids = masks.token_ids[0].long()
    assert sorted(ids.tolist()) == [0, 1, 2, 3]
    logprobs = processed.log_softmax(-1)[0, ids]
    torch.testing.assert_close(logprobs.exp().sum(), torch.tensor(1.0, device="cuda"))
