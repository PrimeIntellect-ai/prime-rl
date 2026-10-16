import copy
import importlib

import pytest
import torch
from torchao.prototype.mx_formats.config import MXFP8Dim0CastKernelChoice, ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common import KernelPreference

from prime_rl.configs.trainer import ModelConfig, MXFP8Config
from prime_rl.trainer.model import apply_quantization
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.fusions import fuse_gate_up_projections
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.expert_compute import (
    BF16ExpertCompute,
    GroupedGemmExpertCompute,
    MXFP8ExpertCompute,
)
from prime_rl.trainer.models.layers.grouped_linear import GroupedLinear
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear
from prime_rl.trainer.moe_runtime import configure_moe_runtime
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models import deepseek_v4_eager_reference as eager_reference
from tests.unit.train.models.test_deepseek_v4 import MODEL, MODEL_CONFIG, _randomize, _single_doc


def _mxfp8_moe_unavailable_reason() -> str | None:
    import prime_kernels

    if "mxfp8_moe" not in prime_kernels.KERNELS:
        return "this prime-kernels install does not ship mxfp8_moe"
    return prime_kernels.unavailable_reason("mxfp8_moe")


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 GEMMs need Blackwell (SM100 or newer)",
    ),
]

requires_mxfp8_moe = pytest.mark.skipif(
    torch.cuda.is_available() and _mxfp8_moe_unavailable_reason() is not None,
    reason=f"prime-kernels mxfp8_moe is unusable: {_mxfp8_moe_unavailable_reason() if torch.cuda.is_available() else ''}",
)

BLOCK = 32
E4M3_MAX = 448.0

# MXFP8 against bfloat16, relative to the reference's own scale. e4m3 keeps 3 mantissa bits, so each
# quantized operand carries up to 2^-4 relative rounding; 0.1 leaves room for two operands and a
# SwiGLU between GEMMs, while a transposed weight, a wrong group or a dropped expert moves the output
# by the order of its whole scale.
BF16_RTOL = 0.1

# MXFP8 GEMM against a float32 oracle that multiplies the same dequantized operands. Only
# accumulation order and the bfloat16 output rounding (2^-8 relative) separate them.
ORACLE_RTOL = 2.0**-7

# Tokens per expert. Every count is a multiple of the 32-row group alignment the dispatcher pads to,
# and some groups are empty. 32 groups is DeepSeek V4-Flash's local expert count at EP 8 (256 / 8), the
# most the CUDA scale rearrange handles; 48 is DeepSeek V4.1's (384 / 8), not a power of two, which
# takes the Triton rearrange with power-of-two padding.
GROUP_COUNTS = {
    "4": [64, 0, 160, 32],
    "32": [32 * ((i * 3) % 4) for i in range(32)],
    "48": [32 * ((i * 7) % 5) for i in range(48)],
}


def _assert_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    actual, reference = actual.float(), reference.float()
    deviation = (actual - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _dequantize(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Float32 values of an MXFP8 tensor with one E8M0 scale per 32 elements of its last axis."""
    exponents = scales.view(torch.uint8).float().reshape(*data.shape[:-1], data.shape[-1] // BLOCK, 1)
    blocks = data.float().reshape(*data.shape[:-1], data.shape[-1] // BLOCK, BLOCK)
    return (blocks * torch.exp2(exponents - 127)).reshape(data.shape)


def _torchao_dense_cast(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight cast `MXFP8Linear` runs in its forward (torchao `mx_mm`, dim0 Triton, RCEIL)."""
    mx = MXTensor.to_mx(
        weight,
        torch.float8_e4m3fn,
        BLOCK,
        ScaleCalculationMode.RCEIL,
        KernelPreference.AUTO,
        mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
    )
    return mx.qdata, mx.scale


def _expert_weight_cast(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight cast prime-kernels' MXFP8 grouped GEMM forward runs on an `[E, N, K]` expert weight.

    prime-rl hands the GEMM `weight.transpose(-2, -1)`, and the GEMM quantizes that view's transpose.
    """
    grouped_gemm = importlib.import_module("prime_kernels.mxfp8_moe.grouped_gemm")
    return grouped_gemm._quantize_rows(weight.transpose(-2, -1).transpose(-2, -1))


def _all_finite_bf16(rows_of: int) -> torch.Tensor:
    magnitudes = torch.arange(0, 0x7F80, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    values = torch.cat([magnitudes, -magnitudes])
    return values[: values.numel() // rows_of * rows_of].reshape(-1, rows_of)


def _one_value_per_block() -> torch.Tensor:
    magnitudes = torch.arange(0, 0x7F80, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    blocks = torch.zeros(magnitudes.numel(), BLOCK, device="cuda", dtype=torch.bfloat16)
    blocks[:, 0] = magnitudes
    return blocks


def _amax_at(mantissa: float) -> torch.Tensor:
    """Blocks whose amax is `mantissa * 2^k`; at 448 and 1.75 the RCEIL scale sits exactly on a power of two."""
    exponents = torch.arange(-20, 20, device="cuda", dtype=torch.float32)
    amax = mantissa * 2.0**exponents
    blocks = torch.rand(exponents.numel(), BLOCK, device="cuda") * amax.unsqueeze(1)
    blocks[:, 0] = amax
    return blocks.bfloat16()


def _on_e4m3_grid() -> torch.Tensor:
    """A DeepSeek V4 style BF16 weight: e4m3 values times one power-of-two scale per 128 elements."""
    values = (torch.randn(256, 512, device="cuda") * 100).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    scales = 2.0 ** torch.randint(-14, -4, (256, 512 // 128), device="cuda").float()
    return (values * scales.repeat_interleave(128, dim=1)).bfloat16()


WEIGHT_CASES = {
    "randn": lambda: (torch.randn(256, 512, device="cuda") * 0.02).bfloat16(),
    "all_finite_bf16": lambda: _all_finite_bf16(512),
    "one_value_per_block": _one_value_per_block,
    "amax_448_pow2": lambda: _amax_at(E4M3_MAX),
    "amax_1.75_pow2": lambda: _amax_at(1.75),
    "on_e4m3_grid": _on_e4m3_grid,
    "zeros": lambda: torch.zeros(8, 64, device="cuda", dtype=torch.bfloat16),
}


@pytest.mark.parametrize("case", WEIGHT_CASES)
def test_dense_weight_cast_matches_vllm_bitwise(case):
    vllm_mxfp8 = pytest.importorskip("vllm.model_executor.layers.quantization.utils.mxfp8_utils")
    torch.manual_seed(0)
    weight = WEIGHT_CASES[case]()

    vllm_data, vllm_scales = vllm_mxfp8.mxfp8_e4m3_quantize(weight)
    data, scales = _torchao_dense_cast(weight)

    assert torch.equal(data.view(torch.uint8), vllm_data.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8).reshape(vllm_scales.shape), vllm_scales.view(torch.uint8))


@requires_mxfp8_moe
@pytest.mark.parametrize("case", WEIGHT_CASES)
def test_expert_weight_cast_matches_vllm_bitwise(case):
    vllm_mxfp8 = pytest.importorskip("vllm.model_executor.layers.quantization.utils.mxfp8_utils")
    torch.manual_seed(0)
    dense = WEIGHT_CASES[case]()
    rows = dense.shape[0] // 2 * 2
    weight = dense[:rows].reshape(2, rows // 2, dense.shape[1])

    data, scales = _expert_weight_cast(weight)

    for expert in range(weight.shape[0]):
        vllm_data, vllm_scales = vllm_mxfp8.mxfp8_e4m3_quantize(weight[expert])
        assert torch.equal(data[expert].view(torch.uint8), vllm_data.view(torch.uint8)), f"expert {expert}"
        assert torch.equal(
            scales[expert].view(torch.uint8).reshape(vllm_scales.shape), vllm_scales.view(torch.uint8)
        ), f"expert {expert}"


def test_on_grid_weights_round_trip_exactly():
    """A weight already on the e4m3 grid with power-of-two scales comes back unchanged from MXFP8."""
    vllm_mxfp8 = pytest.importorskip("vllm.model_executor.layers.quantization.utils.mxfp8_utils")
    torch.manual_seed(0)
    weight = _on_e4m3_grid()

    data, scales = vllm_mxfp8.mxfp8_e4m3_quantize(weight)

    assert torch.equal(_dequantize(data, scales), weight.float())


def _grouped_inputs(counts: list[int], dim: int, out_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens, an `[E, out_dim, dim]` expert weight laid out as prime-rl stores it, and group end offsets."""
    tokens = sum(counts)
    x = (torch.randn(tokens, dim, device="cuda") * 0.5).bfloat16()
    weight = (torch.randn(len(counts), out_dim, dim, device="cuda") * 0.02).bfloat16()
    offsets = torch.cumsum(torch.tensor(counts, device="cuda"), dim=0, dtype=torch.int32)
    return x, weight, offsets


def _grouped_reference(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    outputs, start = [], 0
    for expert, end in enumerate(offsets.tolist()):
        outputs.append(x[start:end] @ weight[expert].t())
        start = end
    return torch.cat(outputs)


@requires_mxfp8_moe
@pytest.mark.parametrize("groups", GROUP_COUNTS)
@pytest.mark.parametrize(("dim", "out_dim"), [(512, 256), (256, 512)], ids=["k512-n256", "k256-n512"])
def test_grouped_gemm_forward_multiplies_the_quantized_operands(groups, dim, out_dim):
    import prime_kernels

    grouped_gemm = importlib.import_module("prime_kernels.mxfp8_moe.grouped_gemm")
    torch.manual_seed(0)
    x, weight, offsets = _grouped_inputs(GROUP_COUNTS[groups], dim, out_dim)

    output = prime_kernels.load("mxfp8_moe").grouped_gemm(x, weight.transpose(-2, -1), offsets)

    x_q = _dequantize(*grouped_gemm.quantize_rows(x))
    weight_q = _dequantize(*_expert_weight_cast(weight))
    oracle = _grouped_reference(x_q, weight_q, offsets)
    _assert_relative(output, oracle, ORACLE_RTOL, "forward vs quantized oracle")
    unquantized = _grouped_reference(x.float(), weight.float(), offsets)
    deviation = (unquantized - oracle).abs().max()
    assert deviation > ORACLE_RTOL * oracle.abs().max(), "quantization should be visible above the oracle bound"


@requires_mxfp8_moe
@pytest.mark.parametrize("high_precision_wgrad", [False, True], ids=["mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp"])
@pytest.mark.parametrize("groups", GROUP_COUNTS)
def test_grouped_gemm_backward_matches_bf16(groups, high_precision_wgrad):
    import prime_kernels

    torch.manual_seed(0)
    x, weight, offsets = _grouped_inputs(GROUP_COUNTS[groups], 512, 256)
    grad_output = torch.randn(x.shape[0], 256, device="cuda").bfloat16()

    x_mx, weight_mx = x.clone().requires_grad_(), weight.clone().requires_grad_()
    prime_kernels.load("mxfp8_moe").grouped_gemm(
        x_mx, weight_mx.transpose(-2, -1), offsets, high_precision_wgrad=high_precision_wgrad
    ).backward(grad_output)
    x_ref, weight_ref = x.float().requires_grad_(), weight.float().requires_grad_()
    _grouped_reference(x_ref, weight_ref, offsets).backward(grad_output.float())

    _assert_relative(x_mx.grad, x_ref.grad, BF16_RTOL, "dgrad")
    _assert_relative(weight_mx.grad, weight_ref.grad, BF16_RTOL, "wgrad")
    empty_experts = [e for e, count in enumerate(GROUP_COUNTS[groups]) if count == 0]
    assert torch.all(weight_mx.grad[empty_experts] == 0), "experts with no tokens must get zero gradient"


@requires_mxfp8_moe
@pytest.mark.parametrize("groups", GROUP_COUNTS)
def test_clamped_swiglu_experts_match_bf16(groups):
    """DeepSeek V4's routed experts (fused gate/up, SwiGLU clamped at 10) under MXFP8 compute."""
    import prime_kernels

    torch.manual_seed(0)
    counts = GROUP_COUNTS[groups]
    with torch.device("cuda"):
        experts = GroupedExperts(dim=128, hidden_dim=64, num_experts=len(counts), activation=ClampedSilu(10.0))
    with torch.no_grad():
        for param in experts.parameters():
            param.normal_(std=0.05)
    fuse_gate_up_projections(experts)
    experts_mx = copy.deepcopy(experts)
    experts.set_compute(BF16ExpertCompute())
    experts_mx.set_compute(MXFP8ExpertCompute(prime_kernels.load("mxfp8_moe"), high_precision_wgrad=False))

    num_tokens = torch.tensor(counts, device="cuda")
    x = torch.randn(sum(counts), 128, device="cuda").bfloat16() * 4
    grad_output = torch.randn(sum(counts), 128, device="cuda").bfloat16()
    x_ref, x_mx = x.clone().requires_grad_(), x.clone().requires_grad_()
    output = experts(x_ref, num_tokens)
    output_mx = experts_mx(x_mx, num_tokens)
    output.backward(grad_output)
    output_mx.backward(grad_output)

    _assert_relative(output_mx, output, BF16_RTOL, "experts forward")
    _assert_relative(x_mx.grad, x_ref.grad, BF16_RTOL, "experts input grad")
    for (name, param), param_mx in zip(experts.named_parameters(), experts_mx.parameters()):
        _assert_relative(param_mx.grad, param.grad, BF16_RTOL, f"experts {name} grad")


def test_mxfp8_linear_forward_multiplies_the_quantized_operands_and_backward_matches_bf16():
    torch.manual_seed(0)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        linear = torch.nn.Linear(512, 256, bias=False)
    with torch.no_grad():
        linear.weight.normal_(std=0.02)
    linear_mx = MXFP8Linear.from_linear(copy.deepcopy(linear))
    x = torch.randn(4, 64, 512, device="cuda").bfloat16()
    grad_output = torch.randn(4, 64, 256, device="cuda").bfloat16()

    x_mx = x.clone().requires_grad_()
    output_mx = linear_mx(x_mx)
    output_mx.backward(grad_output)
    x_ref = x.float().requires_grad_()
    weight_ref = linear.weight.detach().float().requires_grad_()
    (x_ref @ weight_ref.t()).backward(grad_output.float())

    x_q = _dequantize(*_torchao_dense_cast(x.reshape(-1, 512))).reshape(x.shape)
    weight_q = _dequantize(*_torchao_dense_cast(linear.weight.detach()))
    _assert_relative(output_mx, x_q @ weight_q.t(), ORACLE_RTOL, "forward vs quantized oracle")
    _assert_relative(x_mx.grad, x_ref.grad, BF16_RTOL, "input grad")
    _assert_relative(linear_mx.weight.grad, weight_ref.grad, BF16_RTOL, "weight grad")


requires_dsv4_kernels = pytest.mark.skipif(
    dsv4_attention.dsv4_sparse_attn is None,
    reason="the fused sparse attention kernel did not import; tilelang ships in the `gpu` extra, on linux only",
)


ATTENTION_PROJECTIONS = {"q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"}


@requires_dsv4_kernels
@pytest.mark.parametrize("ignore_indexer", [False, True], ids=["plain", "indexer-ignored"])
def test_deepseek_v4_mxfp8_converts_exactly_what_vllm_serves_in_mxfp8(ignore_indexer):
    """vLLM's online MXFP8 serves `wo_a` (never quantizable) and every compressor's projections in BF16, and the indexer as its ignore list says."""
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = DeepseekV4ForCausalLM(MODEL_CONFIG)
    quantization = {"type": "mxfp8"}
    if ignore_indexer:
        quantization["ignore_patterns"] = [*MXFP8Config().ignore_patterns, r"indexer\."]

    apply_quantization(model, ModelConfig.model_validate({"quantization": quantization}))

    linears = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    quantized = {name for name, module in linears.items() if isinstance(module, MXFP8Linear)}
    expected = {
        name
        for name in linears
        if (name.split(".")[-1] in ATTENTION_PROJECTIONS and ".compressor." not in name)
        or ".shared_expert." in name
        or (name.endswith(".indexer.q_b_proj") and not ignore_indexer)
    }
    assert any(name.endswith(".indexer.q_b_proj") for name in linears)
    assert all(isinstance(linears[name], GroupedLinear) for name in linears if name.endswith("o_a_proj"))
    assert quantized == expected


def _tiny_deepseek_v4(config: ModelConfig, kernel_preference: KernelPreference = KernelPreference.AUTO):
    torch.manual_seed(0)
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = DeepseekV4ForCausalLM(MODEL_CONFIG)
    _randomize(model)
    eager_reference.use_eager_attention(model)
    inject_prime_lm_head(model)
    apply_quantization(model, config)
    configure_moe_runtime(model, config, ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1))
    for module in model.modules():
        if isinstance(module, MXFP8Linear):
            module.kernel_preference = kernel_preference
    return model


def _tiny_deepseek_v4_logits(model: DeepseekV4ForCausalLM) -> torch.Tensor:
    torch.manual_seed(1)
    input_ids = torch.randint(0, MODEL["vocab_size"], (1, 32), device="cuda")
    position_ids, seq_lens = _single_doc(input_ids)
    return model(input_ids, position_ids=position_ids, seq_lens=seq_lens)["logits"]


def _emulated_mxfp8_grouped_gemm(x: torch.Tensor, weight_t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """The MXFP8 grouped GEMM's forward in float32 on dequantized operands, with the rows past the last group zeroed."""
    grouped_gemm = importlib.import_module("prime_kernels.mxfp8_moe.grouped_gemm")
    x_q = _dequantize(*grouped_gemm.quantize_rows(x))
    weight_q = _dequantize(*grouped_gemm._quantize_rows(weight_t.transpose(-2, -1)))
    output = torch.zeros(x.shape[0], weight_q.shape[1], device=x.device)
    output[: offsets[-1]] = _grouped_reference(x_q, weight_q, offsets)
    return output.bfloat16()


MXFP8_EVERYWHERE = {"quantization": {"type": "mxfp8"}, "moe": {"compute": {"type": "mxfp8"}}}


@requires_dsv4_kernels
@requires_mxfp8_moe
def test_deepseek_v4_mxfp8_kernels_match_emulated_mxfp8():
    """Inside the full model, MXFP8 linears and experts compute exactly what MXFP8 emulation computes, and a swapped expert does not."""
    config = ModelConfig.model_validate(MXFP8_EVERYWHERE)

    def emulated_model(swap_experts: bool) -> DeepseekV4ForCausalLM:
        model = _tiny_deepseek_v4(config, KernelPreference.EMULATED)
        for module in model.modules():
            if isinstance(module, GroupedExperts):
                module.set_compute(GroupedGemmExpertCompute(_emulated_mxfp8_grouped_gemm, token_group_alignment=BLOCK))
                if swap_experts:
                    module.up_proj.data[[0, 1]] = module.up_proj.data[[1, 0]]
        return model

    with torch.no_grad():
        logits = _tiny_deepseek_v4_logits(_tiny_deepseek_v4(config))
        emulated = _tiny_deepseek_v4_logits(emulated_model(swap_experts=False))
        swapped = _tiny_deepseek_v4_logits(emulated_model(swap_experts=True))

    _assert_relative(logits, emulated, ORACLE_RTOL, "logits vs emulated MXFP8")
    with pytest.raises(AssertionError):
        _assert_relative(logits, swapped, ORACLE_RTOL, "logits vs emulated MXFP8 with two experts swapped")


@requires_dsv4_kernels
@requires_mxfp8_moe
def test_deepseek_v4_mxfp8_backward_produces_finite_gradients_where_bf16_does():
    model = _tiny_deepseek_v4(ModelConfig())
    model_mx = _tiny_deepseek_v4(ModelConfig.model_validate(MXFP8_EVERYWHERE))

    _tiny_deepseek_v4_logits(model).float().sum().backward()
    _tiny_deepseek_v4_logits(model_mx).float().sum().backward()

    for (name, param), param_mx in zip(model.named_parameters(), model_mx.parameters()):
        if param.grad is None:
            assert param_mx.grad is None, name
            continue
        assert param_mx.grad is not None and torch.isfinite(param_mx.grad).all(), name


@requires_mxfp8_moe
def test_grouped_gemm_backward_accepts_an_expanded_gradient():
    """A reduction's backward hands the GEMM a stride-0 gradient, which the MXFP8 casts cannot read directly."""
    import prime_kernels

    torch.manual_seed(0)
    x, weight, offsets = _grouped_inputs(GROUP_COUNTS["4"], 512, 256)
    x_mx, weight_mx = x.clone().requires_grad_(), weight.clone().requires_grad_()
    prime_kernels.load("mxfp8_moe").grouped_gemm(x_mx, weight_mx.transpose(-2, -1), offsets).sum().backward()
    x_ref, weight_ref = x.float().requires_grad_(), weight.float().requires_grad_()
    _grouped_reference(x_ref, weight_ref, offsets).sum().backward()

    _assert_relative(x_mx.grad, x_ref.grad, BF16_RTOL, "dgrad")
    _assert_relative(weight_mx.grad, weight_ref.grad, BF16_RTOL, "wgrad")
