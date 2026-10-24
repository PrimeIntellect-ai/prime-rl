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
    for nucleus in (None, p):
        torch.testing.assert_close(
            apply_top_p_then_top_k(logits.clone(), torch.tensor([-1], device=device), nucleus),
            apply_top_p_then_top_k(logits.clone(), None, nucleus),
        )
    assert apply_top_p_then_top_k(logits, None, None) is logits


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("top_k", [[1, 2, 16, 128] * 4, [-1] * 16, [-1, 2, 16, 128] * 4])
def test_sampling_order_matches_independent_nucleus_reference(device, top_k):
    generator = torch.Generator().manual_seed(4176)
    logits = torch.randn(16, 257, generator=generator).to(device)
    k = torch.tensor(top_k, device=device)
    p = torch.tensor([0.01, 0.5, 0.95, 1.0] * 4, device=device)
    result = apply_top_p_then_top_k(logits.clone(), k, p)
    for row in range(len(logits)):
        order = logits[row].argsort(descending=True)
        cumulative = logits[row].softmax(-1)[order].cumsum(0)
        cap = len(order) if k[row] == -1 else int(k[row])
        size = min(int((cumulative < p[row]).sum()) + 1, cap, len(order))
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


@pytest.mark.gpu
def test_mixed_batch_captures_only_opted_in_supports():
    import numpy as np
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

    from prime_rl.inference.vllm.request_replay import _capture, capture_selected_supports

    logits = torch.randn(4, 257, device="cuda")
    # Normalization can round distinct logits to equal logprobs.
    logits[0] *= 1e-7
    # Bounded training, unbounded nucleus eval, greedy eval, top-k eval.
    processed = apply_top_p_then_top_k(
        logits.clone(),
        torch.tensor([4, 257, 257, 8], device="cuda"),
        torch.tensor([0.95, 0.95, 1.0, 1.0], device="cuda"),
    )
    cu = torch.arange(5, dtype=torch.int32, device="cuda")
    sampled = torch.ones(4, dtype=torch.int32, device="cuda")
    original = SamplingMaskTensors.from_logits
    token = _capture.set((np.array([True, False, False, False]), np.array([4, 257, 257, 8])))
    try:
        masks = capture_selected_supports(original, processed, cu, sampled, 257)
        assert masks.token_ids.shape == (4, 4)
        assert masks.counts.tolist() == [4, 0, 0, 0]
        assert set(masks.token_ids[0].tolist()) == set(torch.isfinite(processed[0]).nonzero().flatten().tolist())
        # Capture does not alter sampling or committed-token counts.
        assert sampled.tolist() == [1, 1, 1, 1]
        assert torch.isfinite(processed[1]).sum() > 4
        _capture.set((np.zeros(4, dtype=bool), np.full(4, 257)))
        assert capture_selected_supports(original, processed, cu, sampled, 257) is None
    finally:
        _capture.reset(token)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_bounded_sampler_matches_full_sort(device):
    from prime_rl.inference.vllm.sampling_order import reference_top_p_then_top_k

    gen = torch.Generator().manual_seed(4210)
    for scale in (0.1, 1.0, 10.0):
        logits = torch.randn(64, 2048, generator=gen).to(device) * scale
        k = torch.tensor([1, 8, 32, 128] * 16, device=device)
        p = torch.tensor([0.01, 0.5, 0.95, 1.0] * 16, device=device)
        expected = reference_top_p_then_top_k(logits.clone(), k, p)
        actual = apply_top_p_then_top_k(logits.clone(), k, p)
        torch.testing.assert_close(actual, expected)
    # Boundary ties must retain the same token IDs as the stable reference.
    logits = torch.zeros(4, 256, device=device)
    logits[:, :8] = 1
    k = torch.tensor([4, 8, 16, 128], device=device)
    p = torch.tensor([0.001, 0.05, 0.95, 1.0], device=device)
    torch.testing.assert_close(
        apply_top_p_then_top_k(logits.clone(), k, p), reference_top_p_then_top_k(logits.clone(), k, p)
    )


@pytest.mark.gpu
def test_paired_support_matches_processed_logprobs():
    import numpy as np
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

    from prime_rl.inference.vllm.paired_capture import pack_support_logprobs, serialize_paired_support

    logits = torch.randn(4, 257, device="cuda")
    processed = apply_top_p_then_top_k(
        logits, torch.tensor([4, 8, 16, 32], device="cuda"), torch.tensor([0.95] * 4, device="cuda")
    )
    cu = torch.arange(5, device="cuda", dtype=torch.int32)
    masks = SamplingMaskTensors.from_logits(processed, cu, torch.ones(4, device="cuda", dtype=torch.int32), 32)
    packed = pack_support_logprobs(masks, processed, cu, np.ones(4, dtype=bool))
    counts = masks.counts.cpu().numpy()
    rows = packed.token_ids.cpu().numpy()
    flat = np.concatenate([row[:n] for row, n in zip(rows, counts)])
    ids, wire = serialize_paired_support(flat, counts)
    import base64

    raw = base64.b64decode(wire["data"])
    values = np.frombuffer(raw, dtype="<f4", offset=len(ids) * 4)
    offsets = wire["offsets"]
    reference = processed.log_softmax(-1).cpu().numpy()
    for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
        np.testing.assert_allclose(values[start:end], reference[i, ids[start:end]], atol=2e-6, rtol=1e-6)
        np.testing.assert_allclose(np.exp(values[start:end]).sum(), 1, atol=1e-6)
        assert np.all(np.diff(values[start:end]) <= 0)
        ties = np.diff(values[start:end]) == 0
        assert np.all(np.diff(ids[start:end])[ties] > 0)
    mixed = pack_support_logprobs(masks, processed, cu, np.array([True, False, True, False]))
    torch.testing.assert_close(mixed.token_ids[1, : counts[1]], masks.token_ids[1, : counts[1]].long())
    with pytest.raises(ValueError, match="Incomplete"):
        serialize_paired_support(ids.astype(np.int64), counts)
