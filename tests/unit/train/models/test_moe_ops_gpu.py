import copy
import importlib.util

import pytest
import torch

from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, MoEArgs
from prime_rl.trainer.models.layers.moe_ops import (
    MoEBackendSettings,
    combine_expert_outputs,
    histogram,
    index_compute,
    moe_scatter,
    moe_weighted_gather,
    run_grouped_experts,
    run_routed_experts,
    scatter_token_weights,
    scatter_tokens_by_expert,
)

pytestmark = [pytest.mark.gpu]


TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None


def _make_topk_ids(
    token_num: int,
    topk: int,
    num_experts: int,
    device: torch.device,
    *,
    allow_dupe: bool = False,
    invalid_frac: float = 0.0,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1 and not allow_dupe:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts
    if invalid_frac > 0:
        mask = torch.rand((token_num, topk), device=device) < invalid_frac
        topk_ids = topk_ids.masked_fill(mask, -1)
    return topk_ids.to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_histogram_triton_matches_torch():
    torch.manual_seed(0)
    topk_ids = _make_topk_ids(
        token_num=4096,
        topk=2,
        num_experts=64,
        device=torch.device("cuda"),
        allow_dupe=True,
        invalid_frac=0.2,
        dtype=torch.int64,
    )
    ref = histogram(topk_ids, expert_num=64, backend="torch")
    out = histogram(topk_ids, expert_num=64, backend="triton")
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_index_compute_triton_matches_torch():
    torch.manual_seed(0)
    topk_ids = _make_topk_ids(
        token_num=4096,
        topk=2,
        num_experts=64,
        device=torch.device("cuda"),
        invalid_frac=0.2,
        dtype=torch.int64,
    )
    expert_hist = histogram(topk_ids, expert_num=64, backend="torch")
    ref = index_compute(topk_ids, expert_hist, backend="torch")
    out = index_compute(topk_ids, expert_hist, backend="triton")
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_moe_scatter_triton_matches_torch():
    torch.manual_seed(0)
    x = torch.randn((2048, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    topk_ids = _make_topk_ids(
        token_num=2048,
        topk=2,
        num_experts=32,
        device=torch.device("cuda"),
        invalid_frac=0.2,
        dtype=torch.int64,
    )
    expert_hist = histogram(topk_ids, expert_num=32, backend="torch")
    scatter_index = index_compute(topk_ids, expert_hist, backend="torch")

    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = moe_scatter(x_ref, scatter_index, backend="torch")
    out_ref.sum().backward()

    x_opt = x.detach().clone().requires_grad_(True)
    out_opt = moe_scatter(x_opt, scatter_index, backend="triton")
    out_opt.sum().backward()

    torch.testing.assert_close(out_opt, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_moe_weighted_gather_triton_matches_torch():
    torch.manual_seed(0)
    x = torch.randn((4096, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    idx = _make_topk_ids(
        token_num=2048,
        topk=2,
        num_experts=32,
        device=torch.device("cuda"),
        invalid_frac=0.1,
        dtype=torch.int64,
    )
    expert_hist = histogram(idx, expert_num=32, backend="torch")
    scatter_index = index_compute(idx, expert_hist, backend="torch")
    weights = torch.rand((2048, 2), device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weights.detach().clone().requires_grad_(True)
    out_ref = moe_weighted_gather(x_ref, scatter_index, w_ref, backend="torch")
    out_ref.sum().backward()

    x_opt = x.detach().clone().requires_grad_(True)
    w_opt = weights.detach().clone().requires_grad_(True)
    out_opt = moe_weighted_gather(x_opt, scatter_index, w_opt, backend="triton")
    out_opt.sum().backward()

    torch.testing.assert_close(out_opt, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(w_opt.grad, w_ref.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_torch_grouped_mm_matches_loop():
    torch.manual_seed(0)
    experts_ref = GroupedExperts(dim=64, hidden_dim=96, num_experts=4, use_grouped_mm=False).cuda()
    experts_ref.init_weights(0.02)
    experts_opt = copy.deepcopy(experts_ref).cuda()
    experts_opt.set_compute_backend("torch_grouped_mm")

    num_tokens = torch.tensor([11, 7, 13, 5], device="cuda", dtype=torch.int32)
    x_ref = torch.randn((int(num_tokens.sum().item()), 64), device="cuda", dtype=torch.float32, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    out_ref = experts_ref(x_ref, num_tokens)
    out_ref.sum().backward()
    out_opt = experts_opt(x_opt, num_tokens)
    out_opt.sum().backward()

    torch.testing.assert_close(out_opt, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-2, atol=1e-2)


def _run_routed_experts_manual(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    score_before_experts: bool,
    backends: MoEBackendSettings,
) -> torch.Tensor:
    routed_input, scatter_index, counts = scatter_tokens_by_expert(
        x,
        expert_ids,
        num_experts=w1.shape[0],
        backends=backends,
    )
    if score_before_experts:
        expert_weights = scatter_token_weights(token_weights, scatter_index)
        routed_input = (routed_input.to(torch.float32) * expert_weights.reshape(-1, 1)).to(x.dtype)
        combine_weights = torch.ones_like(token_weights)
    else:
        combine_weights = token_weights

    expert_output = run_grouped_experts(w1, w2, w3, routed_input, counts, backends.grouped_gemm)
    return combine_expert_outputs(expert_output, scatter_index, combine_weights, backend=backends.gather)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("score_before_experts", [False, True])
def test_run_routed_experts_matches_manual_path(score_before_experts: bool):
    torch.manual_seed(0)
    x = torch.randn((6, 4), device="cuda", dtype=torch.float32, requires_grad=True)
    expert_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]], device="cuda", dtype=torch.int64)
    token_weights = torch.rand((6, 2), device="cuda", dtype=torch.float32, requires_grad=True)
    backends = MoEBackendSettings(grouped_gemm="loop")
    w1 = torch.randn((3, 5, 4), device="cuda", dtype=torch.float32, requires_grad=True)
    w2 = torch.randn((3, 4, 5), device="cuda", dtype=torch.float32, requires_grad=True)
    w3 = torch.randn((3, 5, 4), device="cuda", dtype=torch.float32, requires_grad=True)

    manual_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (w1, w2, w3, x, token_weights)]
    helper_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (w1, w2, w3, x, token_weights)]

    manual_out = _run_routed_experts_manual(
        manual_inputs[0],
        manual_inputs[1],
        manual_inputs[2],
        manual_inputs[3],
        expert_ids,
        manual_inputs[4],
        score_before_experts=score_before_experts,
        backends=backends,
    )
    manual_out.sum().backward()

    helper_out = run_routed_experts(
        helper_inputs[0],
        helper_inputs[1],
        helper_inputs[2],
        helper_inputs[3],
        expert_ids,
        helper_inputs[4],
        num_experts=3,
        score_before_experts=score_before_experts,
        backends=backends,
    )
    helper_out.sum().backward()

    torch.testing.assert_close(helper_out, manual_out, rtol=1e-4, atol=1e-4)
    for helper_tensor, manual_tensor in zip(helper_inputs, manual_inputs, strict=True):
        torch.testing.assert_close(helper_tensor.grad, manual_tensor.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_moe_fused_routed_path_matches_composed():
    torch.manual_seed(0)
    moe_args = MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        score_func="sigmoid",
        route_norm=True,
        route_scale=1.0,
        score_before_experts=False,
        top_k=2,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    moe_ref = MoE(moe_args, dim=32, hidden_dim=48).cuda()
    moe_ref.init_weights(0.02, buffer_device=torch.device("cuda"))
    moe_opt = copy.deepcopy(moe_ref).cuda()

    moe_ref.configure_backends(
        MoEBackendSettings(
            routing="torch",
            scatter="torch",
            gather="torch",
            grouped_gemm="loop",
            routed_ffn="torch",
        )
    )
    moe_opt.configure_backends(
        MoEBackendSettings(
            routing="torch",
            scatter="torch",
            gather="torch",
            grouped_gemm="loop",
            routed_ffn="fused",
        )
    )

    x_ref = torch.randn((2, 16, 32), device="cuda", dtype=torch.float32, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    routed_experts = _make_topk_ids(
        token_num=32,
        topk=2,
        num_experts=8,
        device=torch.device("cuda"),
        allow_dupe=False,
        invalid_frac=0.0,
        dtype=torch.int64,
    ).reshape(2, 16, 2)

    out_ref = moe_ref(x_ref, routed_experts=routed_experts)
    out_ref.sum().backward()
    out_opt = moe_opt(x_opt, routed_experts=routed_experts)
    out_opt.sum().backward()

    torch.testing.assert_close(out_opt, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-4, atol=1e-4)

    for (name_ref, param_ref), (name_opt, param_opt) in zip(
        moe_ref.named_parameters(),
        moe_opt.named_parameters(),
        strict=True,
    ):
        assert name_ref == name_opt
        if param_ref.grad is None or param_opt.grad is None:
            continue
        torch.testing.assert_close(param_opt.grad, param_ref.grad, rtol=1e-4, atol=1e-4)
