import pytest
import torch
from prime_rl_kernels import grouped_gemm
from prime_rl_kernels.nvfp4.quantize import quantize_activations, quantize_weights


def _blackwell_nvfp4_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 0)
        and hasattr(torch, "float4_e2m1fn_x2")
    )


@pytest.mark.skipif(not _blackwell_nvfp4_available(), reason="NVFP4 grouped GEMM requires Blackwell")
@pytest.mark.parametrize("backward", ["bf16", "bf16_dequantized"])
def test_grouped_nvfp4_forward_and_backward(backward: str) -> None:
    torch.manual_seed(42)
    sizes = [7, 33, 0, 88]
    groups = len(sizes)
    rows = sum(sizes)
    contraction_size = 256
    output_size = 128
    offsets = torch.cumsum(
        torch.tensor(sizes, device="cuda", dtype=torch.int32),
        dim=0,
        dtype=torch.int32,
    )
    matrix = (torch.randn(rows, contraction_size, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    weight = (
        torch.randn(
            groups,
            contraction_size,
            output_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    ).requires_grad_()

    quantized_matrix = quantize_activations(matrix.detach(), offsets)
    quantized_weight = quantize_weights(weight.detach())
    dequantized_matrix = quantized_matrix.dequantize()
    dequantized_weight = quantized_weight.dequantize()

    output = grouped_gemm(matrix, weight, offsets, backward=backward)
    reference = torch._grouped_mm(
        dequantized_matrix,
        dequantized_weight,
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    backward_matrix = matrix.detach() if backward == "bf16" else dequantized_matrix
    backward_weight = weight.detach() if backward == "bf16" else dequantized_weight
    reference_grad_matrix = torch._grouped_mm(
        grad_output,
        backward_weight.transpose(-2, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    reference_grad_weight = torch._grouped_mm(
        backward_matrix.transpose(0, 1),
        grad_output,
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(matrix.grad, reference_grad_matrix, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, reference_grad_weight, atol=0, rtol=0)


@pytest.mark.skipif(not _blackwell_nvfp4_available(), reason="NVFP4 grouped GEMM requires Blackwell")
def test_grouped_nvfp4_ignores_physical_tail_rows() -> None:
    """TorchTitan pads its permute buffer beyond the final logical offset."""

    torch.manual_seed(7)
    sizes = [7, 33, 0, 88]
    logical_rows = sum(sizes)
    physical_rows = logical_rows + 32
    contraction_size = 256
    output_size = 128
    offsets = torch.cumsum(
        torch.tensor(sizes, device="cuda", dtype=torch.int32),
        dim=0,
        dtype=torch.int32,
    )
    matrix = torch.randn(physical_rows, contraction_size, device="cuda", dtype=torch.bfloat16) * 0.1
    matrix[logical_rows:] = 1_000
    weight = (
        torch.randn(
            len(sizes),
            contraction_size,
            output_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )

    padded_output = grouped_gemm(matrix, weight, offsets)
    logical_output = grouped_gemm(matrix[:logical_rows].contiguous(), weight, offsets)
    torch.testing.assert_close(
        padded_output[:logical_rows],
        logical_output,
        atol=0,
        rtol=0,
    )
