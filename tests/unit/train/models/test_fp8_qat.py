import hashlib

import pytest
import torch

from prime_rl.trainer.models.fp8_rounding import FP8RoundingState, set_fp8_rounding_step, setup_fp8_rounding
from prime_rl.trainer.models.kernels.fp8_qat import cast_qat_weight
from prime_rl.trainer.models.layers.fp8_qat import fp8_qat_grouped_mm, fp8_qat_linear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="FP8 QAT requires Hopper or Blackwell",
    ),
]


def _sr_reference(scaled, nearest, uniform):
    # Independent E4M3 byte-neighbor reference with explicitly supplied uniforms.
    near = nearest.float()
    bits = nearest.contiguous().view(torch.uint8).to(torch.int16)
    magnitude = bits & 0x7F
    farther = scaled.abs() > near.abs()
    neighbor_bits = (bits & 0x80) | torch.where(farther, magnitude + 1, magnitude - 1).clamp(0, 0x7E)
    neighbor = neighbor_bits.to(torch.uint8).view(torch.float8_e4m3fn).float()
    lower, upper = torch.minimum(near.abs(), neighbor.abs()), torch.maximum(near.abs(), neighbor.abs())
    gap = upper - lower
    probability = torch.where(gap != 0, (scaled.abs() - lower) / gap, torch.zeros_like(gap)).clamp(0, 1)
    return torch.copysign(torch.where(uniform < probability, upper, lower), scaled).to(torch.float8_e4m3fn)


def _step_seed(seed, step):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{step}".encode(), digest_size=8).digest(), "little") & (
        (1 << 63) - 1
    )


def _rounding_args(sr, step):
    return dict(
        rounding_seed=torch.tensor(_step_seed(1234, step), device="cuda") if sr else None,
        rounding_salt=torch.tensor(19, device="cuda") if sr else None,
    )


def _philox_uniform(shape, seed, step, key, expert_start=0):
    # Independent integer reference for Philox4x32-10 and Triton's float mapping.
    rows, cols = shape[-2:]
    offset = torch.arange(torch.Size(shape).numel(), device="cuda", dtype=torch.int64)
    offset += expert_start * rows * cols
    mask = (1 << 32) - 1
    stream = _step_seed(seed, step) ^ key
    c0, c1 = offset & mask, (offset >> 32) & mask
    c2, c3 = torch.zeros_like(c0), torch.zeros_like(c0)
    k0, k1 = stream & mask, stream >> 32
    for _ in range(10):
        a, b = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = ((b >> 32) & mask) ^ c1 ^ k0, b & mask, ((a >> 32) & mask) ^ c3 ^ k1, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    signed = c0.to(torch.int32).to(torch.int64)
    unsigned_magnitude = torch.where(signed < 0, -signed - 1, signed)
    return (unsigned_magnitude.float() * 4.6566127342e-10).reshape(shape)


def _decoded_weight(weight, sr, step):
    q, scales = cast_qat_weight(weight, True, **_rounding_args(sr, step))
    expanded = scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return (q.float() * expanded[..., : weight.shape[-2], : weight.shape[-1]]).bfloat16()


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("shape,scale", [((256, 384), 0.02), ((2176, 256), 1e-5), ((129, 255), 1.0)])
def test_qat_cast_matches_vllm_bytes(ue8m0, sr, shape, scale):
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    torch.manual_seed(47)
    x = torch.randn(shape[::-1], device="cuda", dtype=torch.bfloat16).T * scale
    x[0] = 0
    x[1] = -0.0
    reference, scales = per_block_cast_to_fp8(x, use_ue8m0=False)
    expert_start = 1 << 20 if shape == (129, 255) else 0
    uniform = _philox_uniform(shape, 1234, 7, 19, expert_start)
    if sr:
        reference = reference.clone()
        for start in range(0, shape[0], 2048):
            end = min(start + 2048, shape[0])
            expanded = scales[start // 128 : (end + 127) // 128].repeat_interleave(128, 0)[: end - start]
            expanded = expanded.repeat_interleave(128, 1)[:, : shape[1]]
            reference[start:end] = _sr_reference(
                x[start:end].float() * (1.0 / expanded), reference[start:end], uniform[start:end]
            )
    if ue8m0:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import requant_weight_ue8m0_inplace

        requant_weight_ue8m0_inplace(reference, scales)
    actual, actual_scales = cast_qat_weight(
        x, ue8m0, rounding_offset=expert_start * shape[0] * shape[1], **_rounding_args(sr, 7)
    )
    torch.testing.assert_close(actual_scales, scales, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("groups", [1, 8])
@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("inverse_rope", [False, True])
def test_qat_linear_backward_is_bf16(groups, sr, compiled, inverse_rope):
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import dsv4_rope

    if compiled:
        torch.compiler.reset()
    torch.manual_seed(42)
    shape = (2, 64, 256) if groups == 1 else (2, 64, groups, 256)
    if inverse_rope:
        shape = (2, 64, groups * 2, 128)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((groups * 128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    op = torch.compile(fp8_qat_linear, fullgraph=True) if compiled else fp8_qat_linear
    state = FP8RoundingState(1234, torch.empty((), device="cuda", dtype=torch.int64))
    state.set_step(7)
    cache, positions = None, None
    if inverse_rope:
        angles = torch.randn((128, 32), device="cuda")
        cache = torch.cat((angles.cos(), angles.sin()), -1)
        positions = torch.arange(128, device="cuda")
    out = op(x, w, groups, state.seed if sr else None, torch.tensor(19, device="cuda"), cache, positions)
    rounded = _decoded_weight(w.detach(), sr, 7)
    state.set_step(8)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    reference_x = dsv4_rope(x, cache, positions, inverse=True).reshape(2, 64, groups, 256) if inverse_rope else x
    if groups == 1:
        reference = torch.nn.functional.linear(reference_x, w + (rounded - w).detach())
    else:
        reference = torch.bmm(
            reference_x.reshape(-1, groups, 256).transpose(0, 1),
            (w + (rounded - w).detach()).view(groups, 128, 256).transpose(1, 2),
        )
        reference = reference.transpose(0, 1).reshape_as(out)
    expected = torch.autograd.grad(reference, (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert (out.float() - reference.float()).norm() / reference.float().norm() < 0.06


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_qat_expert_backward_is_bf16(sr, compiled):
    if compiled:
        torch.compiler.reset()
    torch.manual_seed(44)
    x = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    offs = torch.tensor([64, 64, 192, 256], dtype=torch.int32, device="cuda")
    op = torch.compile(fp8_qat_grouped_mm, fullgraph=True) if compiled else fp8_qat_grouped_mm
    out = op(x, w, offs, **_rounding_args(sr, 7))
    rounded = _decoded_weight(w.detach().transpose(1, 2), sr, 7).transpose(1, 2)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    reference = torch._grouped_mm(x, w + (rounded - w).detach(), offs=offs)
    expected = torch.autograd.grad(reference, (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert (out.float() - reference.float()).norm() / reference.float().norm() < 0.06


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_qat_checkpoint_replays_rounding(mode, compiled):
    from prime_rl.configs.trainer import ActivationCheckpointConfig
    from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
    from prime_rl.trainer.models.layers.fp8_qat import FP8QATLinear

    if compiled:
        torch.compiler.reset()
    model = (
        torch.nn.Sequential(
            FP8QATLinear(128, 128, bias=False, stochastic_rounding=True),
            torch.nn.SiLU(),
            FP8QATLinear(128, 128, bias=False, stochastic_rounding=True),
        )
        .cuda()
        .bfloat16()
    )
    setup_fp8_rounding(model, 1234)
    x = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eager = torch.compile(model, fullgraph=True) if compiled else model
    torch.cuda.manual_seed(77)
    expected = eager(x)
    expected_grads = torch.autograd.grad(expected.sum(), (x, *model.parameters()))
    wrapped = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode=mode))(model)
    if compiled:
        wrapped = torch.compile(wrapped, fullgraph=True)
    torch.cuda.manual_seed(77)
    actual = wrapped(x)
    actual_grads = torch.autograd.grad(actual.sum(), (x, *model.parameters()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("sr", [False, True])
def test_qat_rounding_is_invariant_to_expert_shards_and_rng(sr):
    weight = torch.randn((4, 384, 256), device="cuda", dtype=torch.bfloat16) * 0.02
    kwargs = _rounding_args(sr, 13)
    whole, scales = cast_qat_weight(weight, True, **kwargs)
    torch.rand(1357, device="cuda")
    for start in (0, 2):
        shard, shard_scales = cast_qat_weight(
            weight[start : start + 2], True, rounding_offset=start * 384 * 256, **kwargs
        )
        assert torch.equal(shard.view(torch.uint8), whole[start : start + 2].view(torch.uint8))
        assert torch.equal(shard_scales, scales[start : start + 2])
    decoded, _ = cast_qat_weight(weight, True, dequantize=True, **kwargs)
    assert torch.equal(decoded.view(torch.int16), _decoded_weight(weight, sr, 13).view(torch.int16))
    if sr:
        next_weight, _ = cast_qat_weight(weight, True, **_rounding_args(sr, 14))
        assert not torch.equal(whole.view(torch.uint8), next_weight.view(torch.uint8))


def test_qat_shared_weights_share_identity():
    from prime_rl.trainer.models.layers.fp8_qat import replace_linear_with_fp8_qat

    first = torch.nn.Linear(128, 128, bias=False)
    second = torch.nn.Linear(128, 128, bias=False)
    second.weight = first.weight
    model = torch.nn.ModuleDict({"first": first, "second": second, "alias": first})
    replace_linear_with_fp8_qat(model, [], True)
    model.cuda()
    setup_fp8_rounding(model, 1234)
    assert model["first"].rounding_salt == model["second"].rounding_salt == model["alias"].rounding_salt
    assert model["first"].fp8_rounding is model["second"].fp8_rounding is model.fp8_rounding
    assert model["first"] is model["alias"]
    assert model["first"].weight is model["second"].weight


@pytest.mark.parametrize("grouped", [False, True])
def test_qat_and_compute_fp8_share_stochastic_mask(grouped):
    from prime_rl.trainer.models.kernels.fp8_utils import (
        grouped_per_block_cast_to_fp8_triton,
        per_block_cast_to_fp8_triton,
    )

    shape = (4, 256, 256) if grouped else (256, 256)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    # Exact power-of-two scales isolate the shared rounding from serving requantization.
    weight[..., ::128, ::128] = 7.0
    weight[..., 1, :] = -0.0
    kwargs = _rounding_args(True, 13)
    if grouped:
        kwargs["rounding_offset"] = (1 << 32) + 256 * 256
    cast = grouped_per_block_cast_to_fp8_triton if grouped else per_block_cast_to_fp8_triton
    expected, expected_scales = cast(weight, True, **kwargs)
    actual, actual_scales = cast_qat_weight(weight, True, **kwargs)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scales, expected_scales)


@pytest.fixture(scope="module")
def qat_process_group():
    import os
    from datetime import timedelta

    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with uv run torchrun --nproc-per-node=2 -m pytest -k distributed")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    yield dist.get_rank()
    dist.destroy_process_group()


@pytest.mark.parametrize("split_tokens", [False, True], ids=["fsdp", "cp"])
@pytest.mark.parametrize("quantization", ["fp8", "mxfp8"])
def test_distributed_qat_shared_weight(qat_process_group, split_tokens, quantization):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from prime_rl.trainer.models.layers.fp8_qat import replace_linear_with_fp8_qat
    from prime_rl.trainer.models.layers.mxfp8_qat import mxfp8_qat_linear, replace_linear_with_mxfp8_qat

    if quantization == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 QAT requires Blackwell")
    replace = replace_linear_with_mxfp8_qat if quantization == "mxfp8" else replace_linear_with_fp8_qat
    linear = mxfp8_qat_linear if quantization == "mxfp8" else fp8_qat_linear

    rank = qat_process_group
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(256, 128, bias=False)).cuda()
    replace(model, [], True)
    setup_fp8_rounding(model, 1234)
    original_weight = model[0].weight.detach().bfloat16().clone()
    salt = model[0].rounding_salt
    mesh = init_device_mesh("cuda", (2,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model[0], mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    whole_input = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    for step_number in (7, 8):
        set_fp8_rounding_step(model, step_number)
        torch.rand((rank + 1) * 139, device="cuda")
        x = (whole_input.chunk(2)[rank] if split_tokens else whole_input).detach().requires_grad_()
        actual = model(x)
        reference = linear(x, original_weight, 1, model.fp8_rounding.seed, salt)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        actual.sum().backward()
        actual_dx = x.grad.clone()
        expected_dx = torch.autograd.grad(reference.sum(), x)[0]
        torch.testing.assert_close(actual_dx, expected_dx, atol=0, rtol=0)
        model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("quantization", ["fp8", "mxfp8"])
def test_distributed_qat_expert_shard(qat_process_group, quantization):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module

    from prime_rl.trainer.distributed.expert_parallel import ExpertWeightParallel
    from prime_rl.trainer.models.layers.fp8_qat import FP8QATExpertCompute
    from prime_rl.trainer.models.layers.moe import GroupedExperts
    from prime_rl.trainer.models.layers.mxfp8_qat import MXFP8QATExpertCompute

    if quantization == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 QAT requires Blackwell")
    compute = MXFP8QATExpertCompute if quantization == "mxfp8" else FP8QATExpertCompute

    rank = qat_process_group
    torch.manual_seed(45)
    experts = GroupedExperts(256, 128, 4, compute=compute(True)).cuda().bfloat16()
    experts.init_weights(0.02)
    setup_fp8_rounding(experts, 1234)
    reference = GroupedExperts(256, 128, 4, compute=compute(True)).cuda().bfloat16()
    reference.load_state_dict(experts.state_dict())
    setup_fp8_rounding(reference, 1234)
    mesh = init_device_mesh("cuda", (2,))
    parallelize_module(experts, device_mesh=mesh, parallelize_plan=ExpertWeightParallel())
    experts.fp8_expert_offset = rank * 2
    set_fp8_rounding_step(experts, 9)
    set_fp8_rounding_step(reference, 9)
    torch.rand((rank + 1) * 157, device="cuda")
    tokens_per_expert = 128 if quantization == "mxfp8" else 64
    x = torch.randn((2 * tokens_per_expert, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([tokens_per_expert, tokens_per_expert], device="cuda", dtype=torch.int64)
    whole_counts = torch.zeros(4, device="cuda", dtype=torch.int64)
    whole_counts[rank * 2 : rank * 2 + 2] = counts
    actual = experts(x, counts)
    expected = reference(x, whole_counts)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_dx = torch.autograd.grad(actual.sum(), x)[0]
    expected_dx = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_dx, expected_dx, atol=0, rtol=0)


@pytest.mark.parametrize("input_scale", [1.0, 1e-8, 1e-12])
def test_inverse_rope_fp8_cast_matches_vllm(input_scale):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import dsv4_rope, inverse_rope_cast_to_fp8
    from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_triton

    torch.manual_seed(59)
    x = torch.randn((257, 64, 512), device="cuda", dtype=torch.bfloat16) * input_scale
    x[0] = 0.0
    x[1] = -0.0
    angles = torch.randn((257, 32), device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    positions = torch.arange(257, device="cuda")
    expected, expected_scales = fused_inv_rope_fp8_quant(x, positions, cache, 8, 8)
    actual, actual_scales = inverse_rope_cast_to_fp8(x, cache, positions, 8)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scales, expected_scales[:257])
    if input_scale == 1.0:
        rounded = dsv4_rope(x, cache, positions, inverse=True)
        old, _ = per_token_cast_to_fp8_triton(rounded.reshape(-1, 4096), True)
        assert not torch.equal(old.view(torch.uint8).reshape_as(actual), actual.view(torch.uint8))
