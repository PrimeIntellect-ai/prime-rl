import torch
import pytest

from prime_rl.trainer.models.layers.moe_ops import (
    _torch_histogram,
    combine_expert_outputs,
    histogram,
    index_compute,
    moe_scatter,
    moe_weighted_gather,
    resolve_moe_backend_settings,
    scatter_tokens_by_expert,
)


def test_histogram_basic():
    top_k_rank = torch.tensor([0, 1, 1, 3, 3, 3], dtype=torch.int64)
    out = histogram(top_k_rank, expert_num=4, backend="torch")
    assert out.dtype == torch.int32
    assert out.tolist() == [1, 2, 0, 3]


def test_histogram_negative_indices():
    top_k_rank = torch.tensor([0, -1, 2, -3, 2], dtype=torch.int64)
    out = histogram(top_k_rank, expert_num=4, backend="torch")
    assert out.tolist() == [1, 0, 2, 0]


def test_histogram_out_of_range():
    top_k_rank = torch.tensor([0, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="out-of-range"):
        _torch_histogram(top_k_rank, expert_num=2)


def test_index_compute_basic():
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64)
    expert_histogram = torch.tensor([3, 3], dtype=torch.int32)
    out = index_compute(indices, expert_histogram, backend="torch")
    assert out.dtype == torch.int32
    assert out.tolist() == [[0, 3], [4, 1], [2, 5]]


def test_index_compute_negative_indices():
    indices = torch.tensor([[0, -1], [1, -2], [0, 1]], dtype=torch.int64)
    expert_histogram = torch.tensor([2, 2], dtype=torch.int32)
    out = index_compute(indices, expert_histogram, backend="torch")
    assert out.tolist() == [[0, -1], [2, -2], [1, 3]]


def test_moe_scatter_and_weighted_gather_roundtrip():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    idx = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    weights = torch.ones_like(idx, dtype=torch.float32)

    scattered = moe_scatter(x, idx, backend="torch")
    gathered = moe_weighted_gather(scattered, idx, weights, backend="torch")
    expected = x * idx.shape[1]
    torch.testing.assert_close(gathered, expected)


def test_moe_weighted_gather_backward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    idx = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
    weights = torch.tensor([[0.2, 0.8], [1.0, 0.5]], requires_grad=True)

    out = moe_weighted_gather(x, idx, weights, backend="torch")
    out.sum().backward()

    torch.testing.assert_close(x.grad, torch.tensor([[0.8, 0.8], [1.0, 1.0], [0.7, 0.7]]))
    torch.testing.assert_close(weights.grad, torch.tensor([[11.0, 3.0], [7.0, 11.0]]))


def test_scatter_tokens_by_expert_matches_manual_path():
    x = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    expert_ids = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int64)
    backends = resolve_moe_backend_settings(
        use_grouped_mm=False,
        routing="torch",
        scatter="torch",
        gather="torch",
        routed_ffn="torch",
    )

    scattered, scatter_index, expert_histogram = scatter_tokens_by_expert(x, expert_ids, num_experts=2, backends=backends)
    assert expert_histogram.tolist() == [3, 3]
    assert scatter_index.tolist() == [[3, 0], [1, 4], [5, 2]]

    weights = torch.ones_like(expert_ids, dtype=torch.float32)
    combined = combine_expert_outputs(scattered, scatter_index, weights, backend="torch")
    expected = x * expert_ids.shape[1]
    torch.testing.assert_close(combined, expected)


def test_resolve_moe_backend_settings_uses_legacy_grouped_mm_flag():
    grouped_mm = resolve_moe_backend_settings(
        use_grouped_mm=True,
        routing="torch",
        scatter="torch",
        gather="torch",
        routed_ffn="torch",
    )
    assert grouped_mm.grouped_gemm == "torch_grouped_mm"

    loop = resolve_moe_backend_settings(
        use_grouped_mm=False,
        routing="torch",
        scatter="torch",
        gather="torch",
        routed_ffn="torch",
    )
    assert loop.grouped_gemm == "loop"
