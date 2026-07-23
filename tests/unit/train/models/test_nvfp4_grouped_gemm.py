import copy

import pytest
import torch

from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.nvfp4_grouped_gemm import apply_nvfp4_moe_grouped_gemm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
        reason="NVFP4 grouped GEMM requires Blackwell",
    ),
]


def test_nvfp4_grouped_experts_forward_and_backward():
    torch.manual_seed(42)
    reference = (
        GroupedExperts(
            dim=256,
            hidden_dim=128,
            num_experts=4,
            use_grouped_mm=True,
        )
        .cuda()
        .bfloat16()
    )
    reference.init_weights(0.02)
    nvfp4 = copy.deepcopy(reference)
    apply_nvfp4_moe_grouped_gemm(nvfp4)

    counts = torch.tensor([7, 33, 0, 88], device="cuda", dtype=torch.int32)
    matrix = torch.randn(int(counts.sum().item()), 256, device="cuda", dtype=torch.bfloat16) * 0.1
    matrix_reference = matrix.detach().clone().requires_grad_()
    matrix_nvfp4 = matrix.detach().clone().requires_grad_()

    output_reference = reference(matrix_reference, counts)
    output_nvfp4 = nvfp4(matrix_nvfp4, counts)
    torch.testing.assert_close(output_nvfp4, output_reference, atol=0.08, rtol=0.2)

    grad_output = torch.randn_like(output_reference)
    output_reference.backward(grad_output)
    output_nvfp4.backward(grad_output)
    for tensor in (
        output_nvfp4,
        matrix_nvfp4.grad,
        nvfp4.w1.grad,
        nvfp4.w2.grad,
        nvfp4.w3.grad,
    ):
        assert torch.isfinite(tensor).all()
