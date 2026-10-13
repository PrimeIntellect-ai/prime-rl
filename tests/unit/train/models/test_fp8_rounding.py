import pytest
import torch

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.models.fp8_rounding import set_fp8_rounding_step, setup_fp8_rounding
from prime_rl.trainer.models.kernels.fp8_utils import (
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_token_cast_to_fp8_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_triton,
    ue8m0_for_device,
    unpack_rows_triton,
)
from prime_rl.trainer.models.layers.expert_compute import DeepGemmFP8ExpertCompute
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm
from prime_rl.trainer.models.layers.fp8_linear import _fp8_blockwise_mm, replace_linear_with_fp8_blockwise_linear
from prime_rl.trainer.models.layers.moe import GroupedExperts

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
def test_dense_backward_uses_forward_weights(compiled, stochastic):
    import deep_gemm

    torch.manual_seed(16)
    x = torch.randn(137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(137, 512, device="cuda", dtype=torch.bfloat16)
    seed = torch.tensor(2**50 + 31, device="cuda") if stochastic else None
    fn = torch.compile(_fp8_blockwise_mm, fullgraph=True) if compiled else _fp8_blockwise_mm
    y = fn(x, w, 128, seed, 42)
    y.backward(dy)
    mode = ue8m0_for_device(x.device)
    q, s = per_block_cast_to_fp8_triton(w, mode, rounding_seed=seed, rounding_salt=42)
    expected = torch.empty_like(x)
    deep_gemm.fp8_gemm_nt(per_token_cast_to_fp8_triton(dy, mode), (q.T.contiguous(), s.T.contiguous()), expected)
    assert torch.equal(x.grad, expected)
    assert torch.isfinite(w.grad).all()
    # SR changes only weights: dW retains exactly the existing FP8 activation/gradient calculation.
    xr, wr = x.detach().requires_grad_(), w.detach().requires_grad_()
    _fp8_blockwise_mm(xr, wr, 128).backward(dy)
    assert torch.equal(w.grad, wr.grad)


@pytest.mark.parametrize("compiled", [False, True])
def test_grouped_backward_uses_forward_weights(compiled):
    import deep_gemm

    torch.manual_seed(24)
    x = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    dy = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([128, 128, 384], device="cuda", dtype=torch.int32)
    seed = torch.tensor(12345, device="cuda")
    fn = torch.compile(grouped_fp8_gemm, fullgraph=True) if compiled else grouped_fp8_gemm
    fn(x, w, offs, seed, 83, 4 * 128 * 256).backward(dy)
    mode = ue8m0_for_device(x.device)
    q, s = grouped_per_block_cast_to_fp8_triton(
        w.transpose(1, 2), mode, rounding_seed=seed, rounding_salt=83, rounding_offset=4 * 128 * 256
    )
    total, padded, layout, blocks, _, starts, sizes, block_starts = build_grouped_layout(offs, total_m=x.shape[0])
    dyq = grouped_per_token_cast_to_fp8_triton(dy, padded, blocks, starts, sizes, block_starts, mode)
    expected_padded = torch.empty(padded, 128, device="cuda", dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        dyq,
        (q.transpose(1, 2).contiguous(), s.transpose(1, 2).contiguous()),
        expected_padded,
        layout,
        use_psum_layout=False,
    )
    expected = unpack_rows_triton(expected_padded, total, blocks, starts, sizes, block_starts)
    assert torch.equal(x.grad, expected)
    assert torch.count_nonzero(w.grad[1]) == 0
    xr, wr = x.detach().requires_grad_(), w.detach().requires_grad_()
    grouped_fp8_gemm(xr, wr, offs).backward(dy)
    assert torch.equal(w.grad, wr.grad)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("checkpoint_mode", ["full", "selective"])
def test_step_checkpoint_replay_and_tied_weights(compiled, checkpoint_mode):
    torch.manual_seed(43)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16),
        torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    model[1].weight = model[0].weight
    replace_linear_with_fp8_blockwise_linear(model, [], stochastic_rounding=True)
    setup_fp8_rounding(model, 123)
    assert model[0].rounding_salt == model[1].rounding_salt
    assert model[0].fp8_rounding is model[1].fp8_rounding
    fn = torch.compile(model, fullgraph=True) if compiled else model
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    set_fp8_rounding_step(model, 9)
    y = fn(x)
    set_fp8_rounding_step(model, 10)
    y.sum().backward()
    dx, dw = x.grad.clone(), model[0].weight.grad.clone()
    x.grad = None
    model.zero_grad(set_to_none=True)
    set_fp8_rounding_step(model, 9)
    replay = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode=checkpoint_mode))(fn)(x)
    replay.sum().backward()
    assert torch.equal(y, replay)
    assert torch.equal(dx, x.grad)
    assert torch.equal(dw, model[0].weight.grad)
    set_fp8_rounding_step(model, 10)
    assert not torch.equal(y, fn(x))
    set_fp8_rounding_step(model, 9)
    assert torch.equal(y, fn(x))


def test_expert_runtime_rounding():
    experts = GroupedExperts(128, 256, 3, compute=DeepGemmFP8ExpertCompute(True)).to("cuda", torch.bfloat16)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std=0.02)
    setup_fp8_rounding(experts, 77)
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([128, 0, 128], device="cuda")
    y = experts(x, counts)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in experts.parameters())
    set_fp8_rounding_step(experts, 1)
    assert not torch.equal(y, experts(x, counts))
