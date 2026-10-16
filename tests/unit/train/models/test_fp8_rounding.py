import pytest
import torch

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.models.fp8_rounding import (
    FROZEN_STEP,
    _stream_id,
    seed_for_step,
    set_fp8_rounding_step,
    setup_fp8_rounding,
)
from prime_rl.trainer.models.fusions import fuse_qkv_projections
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
    salt = torch.tensor(42, device="cuda") if stochastic else None
    fn = torch.compile(_fp8_blockwise_mm, fullgraph=True) if compiled else _fp8_blockwise_mm
    y = fn(x, w, 128, seed, salt)
    y.backward(dy)
    mode = ue8m0_for_device(x.device)
    q, s = per_block_cast_to_fp8_triton(w, mode, rounding_seed=seed, rounding_salt=salt)
    expected = torch.empty_like(x)
    deep_gemm.fp8_gemm_nt(per_token_cast_to_fp8_triton(dy, mode), (q.T.contiguous(), s.T.contiguous()), expected)
    assert torch.equal(x.grad, expected)
    assert torch.isfinite(w.grad).all()
    # SR changes only weights: dW retains exactly the existing FP8 activation/gradient calculation.
    xr, wr = x.detach().requires_grad_(), w.detach().requires_grad_()
    _fp8_blockwise_mm(xr, wr, 128).backward(dy)
    assert torch.equal(w.grad, wr.grad)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("requires_x,requires_weight", [(True, True), (True, False), (False, True)])
def test_grouped_backward_uses_forward_weights(compiled, requires_x, requires_weight):
    import deep_gemm

    torch.manual_seed(24)
    x = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16, requires_grad=requires_x)
    w = torch.randn(3, 256, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_(requires_weight)
    dy = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([128, 128, 384], device="cuda", dtype=torch.int32)
    seed = torch.tensor(12345, device="cuda")
    salt = torch.tensor(83, device="cuda")
    fn = torch.compile(grouped_fp8_gemm, fullgraph=True) if compiled else grouped_fp8_gemm
    fn(x, w, offs, seed, salt, 4 * 128 * 256).backward(dy)
    mode = ue8m0_for_device(x.device)
    q, s = grouped_per_block_cast_to_fp8_triton(
        w.transpose(1, 2), mode, rounding_seed=seed, rounding_salt=salt, rounding_offset=4 * 128 * 256
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
    if requires_x:
        assert torch.equal(x.grad, expected)
    else:
        assert x.grad is None
    xr, wr = x.detach().requires_grad_(), w.detach().requires_grad_()
    grouped_fp8_gemm(xr, wr, offs).backward(dy)
    if requires_weight:
        assert torch.count_nonzero(w.grad[1]) == 0
        assert torch.equal(w.grad, wr.grad)
    else:
        assert w.grad is None


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


def test_fused_qkv_rounds_only_the_packed_projection():
    attention = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(attention, name, torch.nn.Linear(256, 128, bias=False, device="cuda", dtype=torch.bfloat16))
    fuse_qkv_projections(attention)
    replace_linear_with_fp8_blockwise_linear(attention, [], stochastic_rounding=True)
    setup_fp8_rounding(attention, 5)
    assert attention.qkv_proj.rounding_salt.item() == _stream_id("qkv_proj.weight")
    assert attention.qkv_proj.fp8_rounding is attention.fp8_rounding
    for name in ("q_proj", "k_proj", "v_proj"):
        assert getattr(attention, name).rounding_salt is None
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    y = attention.qkv_proj(x)
    set_fp8_rounding_step(attention, 1)
    assert not torch.equal(y, attention.qkv_proj(x))


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


@pytest.mark.parametrize("grouped", [False, True])
def test_rounding_streams_share_compiled_graph(grouped):
    torch._dynamo.reset()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    if grouped:
        model = torch.nn.ModuleList(
            [GroupedExperts(128, 256, 3, compute=DeepGemmFP8ExpertCompute(True)) for _ in range(3)]
        ).to("cuda", torch.bfloat16)
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    else:
        model = torch.nn.ModuleList(
            [torch.nn.Linear(128, 128, bias=False, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        )
        replace_linear_with_fp8_blockwise_linear(model, [], stochastic_rounding=True)
    setup_fp8_rounding(model, 123)
    for module in model:
        module.compile(backend=backend, fullgraph=True)
    for step, rows in enumerate((128, 128, 256)):
        set_fp8_rounding_step(model, step)
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        inputs = (x, torch.tensor([rows, 0, 0], device="cuda")) if grouped else (x,)
        for module in model:
            assert torch.equal(module(*inputs), module.forward(*inputs))
        assert len(graphs) == (2 if step == 2 else 1)


def _frozen_linear(weight: torch.Tensor, frozen: bool) -> torch.nn.Sequential:
    model = torch.nn.Sequential(torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="cuda"))
    model[0].weight = torch.nn.Parameter(weight.clone())
    replace_linear_with_fp8_blockwise_linear(model, [], stochastic_rounding=True)
    setup_fp8_rounding(model, 3, frozen=frozen)
    return model


def test_frozen_rounding_keeps_the_first_step_stream():
    torch.manual_seed(31)
    weight = torch.randn(256, 384, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 384, device="cuda", dtype=torch.bfloat16)
    frozen = _frozen_linear(weight, frozen=True)
    stepwise = _frozen_linear(weight, frozen=False)
    assert frozen.fp8_rounding.seed.item() == seed_for_step(3, FROZEN_STEP)
    set_fp8_rounding_step(frozen, 1)
    y = frozen(x)
    for step in (2, 9, 1000):
        set_fp8_rounding_step(frozen, step)
        assert torch.equal(frozen(x), y)
    set_fp8_rounding_step(stepwise, FROZEN_STEP)
    assert torch.equal(stepwise(x), y)
    set_fp8_rounding_step(stepwise, FROZEN_STEP + 1)
    assert not torch.equal(stepwise(x), y)


def test_frozen_cast_differs_from_rtn_on_the_expected_share(monkeypatch):
    monkeypatch.setenv("PRIME_RL_FP8_WEIGHT_UE8M0", "1")
    torch.manual_seed(32)
    weight = 0.02 * torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    seed = torch.tensor(seed_for_step(3, FROZEN_STEP), device="cuda")
    salt = torch.tensor(11, device="cuda")
    q_frozen, scale = per_block_cast_to_fp8_triton(weight, True, rounding_seed=seed, rounding_salt=salt)
    q_rtn, scale_rtn = per_block_cast_to_fp8_triton(weight, True)
    assert torch.equal(scale, scale_rtn)
    scaled = weight.float() / scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    spacing = torch.exp2(torch.clamp(torch.floor(torch.log2(scaled.abs().clamp_min(2.0**-12))) - 3, min=-9))
    # Stochastic rounding takes the farther neighbour with probability |x - RTN(x)| / spacing.
    probability = ((scaled - q_rtn.float()).abs() / spacing).clamp(max=1.0)
    observed = (q_frozen.float() != q_rtn.float()).float().sum().item()
    expected = probability.sum().item()
    tolerance = 5 * (probability * (1 - probability)).sum().sqrt().item()
    assert abs(observed - expected) < tolerance, (observed, expected, tolerance)


def test_frozen_rounding_backward_reuses_the_forward_cast(monkeypatch):
    import deep_gemm

    monkeypatch.setenv("PRIME_RL_FP8_WEIGHT_UE8M0", "1")
    torch.manual_seed(33)
    weight = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    model = _frozen_linear(weight, frozen=True)
    x = torch.randn(137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(137, 512, device="cuda", dtype=torch.bfloat16)
    set_fp8_rounding_step(model, 6)
    model(x).backward(dy)
    seed = torch.tensor(seed_for_step(3, FROZEN_STEP), device="cuda")
    q, s = per_block_cast_to_fp8_triton(weight, True, rounding_seed=seed, rounding_salt=model[0].rounding_salt)
    expected = torch.empty_like(x)
    mode = ue8m0_for_device(x.device)
    deep_gemm.fp8_gemm_nt(per_token_cast_to_fp8_triton(dy, mode), (q.T.contiguous(), s.T.contiguous()), expected)
    assert torch.equal(x.grad, expected)


def test_frozen_rounding_requires_stochastic_rounding_everywhere():
    from prime_rl.configs.trainer import DeepGemmFP8MoEComputeConfig, FP8Config, ModelConfig, MoERuntimeConfig

    stochastic_experts = MoERuntimeConfig(compute=DeepGemmFP8MoEComputeConfig(stochastic_rounding=True))
    ModelConfig(fp8_rounding_frozen=True, quantization=FP8Config(stochastic_rounding=True), moe=stochastic_experts)
    with pytest.raises(ValueError, match="fp8_rounding_frozen"):
        ModelConfig(fp8_rounding_frozen=True, quantization=FP8Config(), moe=stochastic_experts)
    with pytest.raises(ValueError, match="fp8_rounding_frozen"):
        ModelConfig(fp8_rounding_frozen=True, quantization=FP8Config(stochastic_rounding=True))
