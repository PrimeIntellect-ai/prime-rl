import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_qat import cast_qat_weight
from prime_rl.trainer.models.layers.fp8_qat import fp8_qat_grouped_mm, fp8_qat_linear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="FP8 QAT requires Hopper or Blackwell",
    ),
]


def _vllm_sr_reference(scaled, nearest, uniform):
    # PR #3627's stochastic probability rule with explicitly supplied uniforms.
    near = nearest.float()
    bits = nearest.contiguous().view(torch.uint8).to(torch.int16)
    magnitude = bits & 0x7F
    farther = scaled.abs() > near.abs()
    neighbor_bits = (bits & 0x80) | torch.where(farther, magnitude + 1, magnitude - 1).clamp(0, 0x7E)
    neighbor = neighbor_bits.to(torch.uint8).view(torch.float8_e4m3fn).float()
    gap = neighbor - near
    probability = torch.where(gap != 0, (scaled - near) / gap, torch.zeros_like(gap)).clamp(0, 1)
    return torch.where(uniform < probability, neighbor, near).to(torch.float8_e4m3fn)


def _philox_uniform(shape, seed, step, key, expert_start=0):
    # Independent integer reference for Philox4x32-10 and Triton's float mapping.
    rows, cols = shape[-2:]
    offset = torch.arange(torch.Size(shape).numel(), device="cuda", dtype=torch.int64)
    offset += expert_start * rows * cols
    mask = (1 << 32) - 1
    stream = (seed ^ key ^ (step * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
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


def _decoded_weight(weight, sr, step, key=19, expert_start=0):
    q, scales = cast_qat_weight(weight, True, sr, step=step, seed=1234, weight_key=key, expert_start=expert_start)
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
    step = torch.tensor(7, device="cuda", dtype=torch.int64)
    expert_start = 1 << 20 if shape == (129, 255) else 0
    uniform = _philox_uniform(shape, 1234, 7, 19, expert_start)
    if sr:
        reference = reference.clone()
        for start in range(0, shape[0], 2048):
            end = min(start + 2048, shape[0])
            expanded = scales[start // 128 : (end + 127) // 128].repeat_interleave(128, 0)[: end - start]
            expanded = expanded.repeat_interleave(128, 1)[:, : shape[1]]
            reference[start:end] = _vllm_sr_reference(
                x[start:end].float() * (1.0 / expanded), reference[start:end], uniform[start:end]
            )
    if ue8m0:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import requant_weight_ue8m0_inplace

        requant_weight_ue8m0_inplace(reference, scales)
    actual, actual_scales = cast_qat_weight(
        x, ue8m0, sr, step=step, seed=1234, weight_key=19, expert_start=expert_start
    )
    torch.testing.assert_close(actual_scales, scales, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("groups", [1, 8])
@pytest.mark.parametrize("sr", [False, True])
def test_qat_linear_backward_is_bf16(groups, sr, compiled):
    torch.manual_seed(42)
    shape = (2, 64, 256) if groups == 1 else (2, 64, groups, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((groups * 128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    op = torch.compile(fp8_qat_linear, fullgraph=True) if compiled else fp8_qat_linear
    step = torch.tensor(7, device="cuda", dtype=torch.int64)
    out = op(x, w, groups, sr, step, 1234, 19)
    rounded = _decoded_weight(w.detach(), sr, step)
    step.add_(1)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    if groups == 1:
        reference = torch.nn.functional.linear(x, w + (rounded - w).detach())
    else:
        reference = torch.bmm(
            x.reshape(-1, groups, 256).transpose(0, 1),
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
    torch.manual_seed(44)
    x = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    offs = torch.tensor([64, 64, 192, 256], dtype=torch.int32, device="cuda")
    op = torch.compile(fp8_qat_grouped_mm, fullgraph=True) if compiled else fp8_qat_grouped_mm
    step = torch.tensor(7, device="cuda", dtype=torch.int64)
    out = op(x, w, offs, sr, step, 1234, 19, 0)
    rounded = _decoded_weight(w.detach().transpose(1, 2), sr, step).transpose(1, 2)
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
    from prime_rl.trainer.models.layers.fp8_qat import FP8QATLinear, initialize_qat_rounding

    model = (
        torch.nn.Sequential(
            FP8QATLinear(128, 128, bias=False, stochastic_rounding=True),
            torch.nn.SiLU(),
            FP8QATLinear(128, 128, bias=False, stochastic_rounding=True),
        )
        .cuda()
        .bfloat16()
    )
    initialize_qat_rounding(model)
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
    step = torch.tensor(13, device="cuda", dtype=torch.int64)
    kwargs = dict(step=step, seed=1234, weight_key=19)
    whole, scales = cast_qat_weight(weight, True, sr, **kwargs)
    torch.rand(1357, device="cuda")
    for start in (0, 2):
        shard, shard_scales = cast_qat_weight(weight[start : start + 2], True, sr, expert_start=start, **kwargs)
        assert torch.equal(shard.view(torch.uint8), whole[start : start + 2].view(torch.uint8))
        assert torch.equal(shard_scales, scales[start : start + 2])
    decoded, _ = cast_qat_weight(weight, True, sr, dequantize=True, **kwargs)
    assert torch.equal(decoded.view(torch.int16), _decoded_weight(weight, sr, step).view(torch.int16))
    if sr:
        next_weight, _ = cast_qat_weight(weight, True, sr, **(kwargs | {"step": step + 1}))
        assert not torch.equal(whole.view(torch.uint8), next_weight.view(torch.uint8))


def test_qat_shared_weights_share_identity():
    from prime_rl.trainer.models.layers.fp8_qat import replace_linear_with_fp8_qat

    first = torch.nn.Linear(128, 128, bias=False)
    second = torch.nn.Linear(128, 128, bias=False)
    second.weight = first.weight
    model = torch.nn.ModuleDict({"first": first, "second": second, "alias": first})
    replace_linear_with_fp8_qat(model, [], True, 1234)
    assert model["first"].weight_key == model["second"].weight_key == model["alias"].weight_key
    assert model["first"] is model["alias"]
    assert model["first"].weight is model["second"].weight


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
def test_distributed_qat_shared_weight(qat_process_group, split_tokens):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from prime_rl.trainer.model import set_fp8_qat_step
    from prime_rl.trainer.models.layers.fp8_qat import initialize_qat_rounding, replace_linear_with_fp8_qat

    rank = qat_process_group
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(256, 128, bias=False)).cuda()
    replace_linear_with_fp8_qat(model, [], True, 1234)
    original_weight = model[0].weight.detach().bfloat16().clone()
    key = model[0].weight_key
    mesh = init_device_mesh("cuda", (2,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model[0], mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    initialize_qat_rounding(model)
    whole_input = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    for step_number in (7, 8):
        set_fp8_qat_step(model, step_number)
        torch.rand((rank + 1) * 139, device="cuda")
        x = (whole_input.chunk(2)[rank] if split_tokens else whole_input).detach().requires_grad_()
        actual = model(x)
        reference = fp8_qat_linear(x, original_weight, 1, True, model._fp8_rounding_step, 1234, key)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        actual.sum().backward()
        actual_dx = x.grad.clone()
        expected_dx = torch.autograd.grad(reference.sum(), x)[0]
        torch.testing.assert_close(actual_dx, expected_dx, atol=0, rtol=0)
        model.zero_grad(set_to_none=True)


def test_distributed_qat_expert_shard(qat_process_group):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module

    from prime_rl.trainer.distributed.expert_parallel import ExpertWeightParallel
    from prime_rl.trainer.models.layers.fp8_qat import (
        FP8QATExpertCompute,
        initialize_qat_rounding,
        logical_weight_keys,
    )
    from prime_rl.trainer.models.layers.moe import GroupedExperts

    rank = qat_process_group
    torch.manual_seed(45)
    experts = GroupedExperts(256, 128, 4, compute=FP8QATExpertCompute(True, 1234)).cuda().bfloat16()
    experts.init_weights(0.02)
    keys = logical_weight_keys(experts)
    experts.fp8_qat_keys = {name: keys[id(weight)] for name, weight in experts.named_parameters()}
    experts.fp8_qat_expert_start = 0
    initialize_qat_rounding(experts)
    reference = GroupedExperts(256, 128, 4, compute=FP8QATExpertCompute(True, 1234)).cuda().bfloat16()
    reference.load_state_dict(experts.state_dict())
    reference.fp8_qat_keys = experts.fp8_qat_keys.copy()
    reference.fp8_qat_expert_start = 0
    initialize_qat_rounding(reference)
    mesh = init_device_mesh("cuda", (2,))
    parallelize_module(experts, device_mesh=mesh, parallelize_plan=ExpertWeightParallel())
    experts.fp8_qat_expert_start = rank * 2
    experts.rounding_step.fill_(9)
    reference.rounding_step.fill_(9)
    torch.rand((rank + 1) * 157, device="cuda")
    x = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([64, 64], device="cuda", dtype=torch.int64)
    whole_counts = torch.zeros(4, device="cuda", dtype=torch.int64)
    whole_counts[rank * 2 : rank * 2 + 2] = counts
    actual = experts(x, counts)
    expected = reference(x, whole_counts)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_dx = torch.autograd.grad(actual.sum(), x)[0]
    expected_dx = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_dx, expected_dx, atol=0, rtol=0)
