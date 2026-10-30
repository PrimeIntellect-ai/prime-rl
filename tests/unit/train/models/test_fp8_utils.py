import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.kernels.fp8_torch_quant import quant_1x128, quant_128x1_tp, quant_grouped_128x1_kmajor
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    ceil_div,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="block-fp8 cast kernels use Triton fp8e4nv (e4m3), only supported on Hopper (SM90) and newer",
    ),
]


@pytest.mark.parametrize("rows,cols", [(256, 256), (256, 512), (512, 256), (1024, 768), (384, 128)])
def test_block_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is *bit-identical* to unfused."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_block_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_block_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)


@pytest.mark.parametrize("rows,cols", [(256, 512), (512, 256), (128, 1024), (1024, 768), (384, 512)])
def test_token_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is *bit-identical* to unfused."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_token_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_token_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)


@pytest.mark.parametrize(
    "rows,cols,input_scale",
    [(256, 256, 1.0), (256, 512, 0.3), (512, 256, 0.02), (1024, 768, 0.02), (384, 128, 0.005)],
)
def test_token_cast_matches_vllm_cuda_quant(rows, cols, input_scale):
    """The per-token cast is *bit-identical* to vLLM's production CUDA quantizer."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * input_scale
    assert x.is_contiguous()

    ref_q, ref_s = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    q, s = per_token_cast_to_fp8_triton(x, False)

    assert q.shape == ref_q.shape == (rows, cols)
    assert s.shape == ref_s.shape == (rows, cols // 128)
    assert torch.equal(s, ref_s)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


@pytest.mark.parametrize("transposed_view", [False, True], ids=["contiguous", "transposed_view"])
@pytest.mark.parametrize(
    "rows,cols,input_scale",
    [(256, 256, 1.0), (256, 512, 0.02), (512, 256, 0.002), (384, 320, 0.02), (256, 384, 0.0005)],
)
def test_block_cast_matches_vllm_online_quant(rows, cols, input_scale, transposed_view):
    """The grouped weight cast is *bit-identical* to the block quantizer vLLM's ``fp8_per_block`` serves with."""
    pytest.importorskip("vllm")
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    torch.manual_seed(rows + cols)
    groups = 3
    if transposed_view:
        x = (torch.randn(groups, cols, rows, device="cuda", dtype=torch.bfloat16) * input_scale).transpose(1, 2)
    else:
        x = torch.randn(groups, rows, cols, device="cuda", dtype=torch.bfloat16) * input_scale

    q, s = grouped_per_block_cast_to_fp8_triton(x, False)

    for group in range(groups):
        ref_q, ref_s = per_block_cast_to_fp8(x[group].contiguous(), use_ue8m0=False)
        assert s[group].shape == ref_s.shape
        assert torch.equal(s[group], ref_s)
        assert torch.equal(q[group].view(torch.uint8), ref_q.view(torch.uint8))


# Two token counts per quantizer, so the compiled quantizers run one dynamic-shape graph on both. The dense
# counts are not multiples of 128, which the 128x1 cast pads. Each grouped case has empty groups and a tail
# past `offs[-1]`.
EXACT_QUANT_TOKENS = (1000, 1537)
EXACT_QUANT_GROUP_COUNTS = ([256, 0, 384, 128, 0, 512], [128, 640, 0, 256, 128, 0])
EXACT_QUANT_TAIL_ROWS = 128
BELOW_AMAX_FLOOR = 1e-14


def _exact_quant_input(tokens: int, channels: int) -> torch.Tensor:
    """bf16 spanning 1e-4 to 1e4 across tokens, with the first 128 tokens all zero and the next 128 below the amax floor."""
    generator = torch.Generator(device="cuda").manual_seed(tokens + channels)
    magnitudes = torch.logspace(-4, 4, tokens, device="cuda").unsqueeze(-1)
    x = torch.randn(tokens, channels, device="cuda", generator=generator) * magnitudes
    x[:GROUP_ALIGNMENT] = 0
    x[GROUP_ALIGNMENT : 2 * GROUP_ALIGNMENT] = (
        torch.randn(GROUP_ALIGNMENT, channels, device="cuda", generator=generator) * BELOW_AMAX_FLOOR
    )
    return x.to(torch.bfloat16)


def _exact_torch_1x128(x: torch.Tensor, offs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    return quant_1x128(x, "exact")


def _triton_1x128(x: torch.Tensor, offs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_cast_to_fp8_triton(x, False)


def _exact_torch_128x1_tp(x: torch.Tensor, offs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    return quant_128x1_tp(x, "exact")


def _triton_128x1_tp(x: torch.Tensor, offs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    padded_tokens = ceil_div(x.size(0), GROUP_ALIGNMENT) * GROUP_ALIGNMENT
    return per_token_cast_to_fp8_tp_triton(F.pad(x, (0, 0, 0, padded_tokens - x.size(0))), False)


def _exact_torch_grouped_128x1_kmajor(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quant_grouped_128x1_kmajor(x, offs, "exact")


def _triton_grouped_128x1_kmajor(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, block_to_group, ks_tensor, starts, actual_ms, block_starts = build_grouped_layout(offs, x.size(0))
    grouped_rows = int(ks_tensor.sum())
    return grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
        x,
        grouped_rows,
        block_to_group[: grouped_rows // GROUP_ALIGNMENT],
        starts,
        actual_ms,
        ks_tensor,
        block_starts,
        False,
    )


# The torch-op quantizer, the Triton kernel it replaces, and the channel count. 130 channels pad the 128x1
# scales to DeepGEMM's TMA alignment.
EXACT_QUANTIZERS = {
    "1x128": (_exact_torch_1x128, _triton_1x128, 384),
    "128x1_tp": (_exact_torch_128x1_tp, _triton_128x1_tp, 130),
    "grouped_128x1_kmajor": (_exact_torch_grouped_128x1_kmajor, _triton_grouped_128x1_kmajor, 256),
}


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("layout", list(EXACT_QUANTIZERS))
def test_exact_torch_quant_matches_triton(layout, compiled):
    """The torch-op quantizers with correctly rounded divisions are *bit-identical* to the Triton kernels, eager and compiled with dynamic tokens."""
    quantize, reference, channels = EXACT_QUANTIZERS[layout]
    if layout.startswith("grouped"):
        cases = [
            (
                _exact_quant_input(sum(counts) + EXACT_QUANT_TAIL_ROWS, channels),
                torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32),
            )
            for counts in EXACT_QUANT_GROUP_COUNTS
        ]
    else:
        cases = [(_exact_quant_input(tokens, channels), None) for tokens in EXACT_QUANT_TOKENS]
    if compiled:
        torch._dynamo.reset()
        torch._dynamo.mark_dynamic(cases[0][0], 0)
        quantize = torch.compile(quantize, fullgraph=True)

    for call, (x, offs) in enumerate(cases):
        with torch._dynamo.config.patch(error_on_recompile=call > 0):
            fp8, scale = quantize(x, offs)
        ref_fp8, ref_scale = reference(x, offs)

        # The grouped torch-op cast also covers the rows past `offs[-1]`, which the Triton kernel leaves out.
        fp8 = fp8.reshape(-1)[: ref_fp8.numel()]
        scale = scale[:, : ref_scale.size(1)]
        assert scale.shape == ref_scale.shape and scale.stride() == ref_scale.stride()
        assert torch.equal(fp8.view(torch.uint8), ref_fp8.reshape(-1).view(torch.uint8))
        assert torch.equal(scale.view(torch.int32), ref_scale.view(torch.int32))
