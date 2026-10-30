import copy

import pytest
import torch
from torch import nn
from torch._inductor.utils import run_and_get_code

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.fp8_quant_pass import Fp8QuantPass
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
        reason="the pass's pre-quantized DeepGEMM ops take fp32 scales, which DeepGEMM accepts on SM90 only",
    ),
]

HIDDEN = 512
INTERMEDIATE = 256
SWIGLU_LIMIT = 10.0
# Token counts run through one dynamic-shape graph. 1000 is not a multiple of 128, which the weight gradient pads.
TOKENS = (1000, 1536)
# Rounds a fused producer's output to bfloat16 before its quantization, as the unfused graph stores it.
INDUCTOR_OPTIONS = {"emulate_precision_casts": True}


@pytest.fixture(autouse=True)
def _require_deep_gemm():
    pytest.importorskip("deep_gemm")


class GatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = Float8BlockwiseLinear(HIDDEN, 2 * INTERMEDIATE, bias=False)
        self.down_proj = Float8BlockwiseLinear(INTERMEDIATE, HIDDEN, bias=False)
        self.activation = ClampedSilu(SWIGLU_LIMIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.activation.apply(gate, up))


def _compiled_block(block: nn.Module, options: dict) -> nn.Module:
    block = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="full"))(copy.deepcopy(block))
    block.compile(fullgraph=True, options=options)
    return block


def _steps(block: nn.Module) -> list[dict[str, torch.Tensor]]:
    results = []
    for tokens in TOKENS:
        generator = torch.Generator(device="cuda").manual_seed(tokens)
        x = torch.randn(tokens, HIDDEN, device="cuda", generator=generator).to(torch.bfloat16).requires_grad_()
        grad_output = torch.randn(tokens, HIDDEN, device="cuda", generator=generator).to(torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        block.zero_grad(set_to_none=True)
        out = block(x)
        out.backward(grad_output)
        results.append(
            {"out": out.detach(), "x.grad": x.grad}
            | {f"{name}.grad": param.grad.clone() for name, param in block.named_parameters()}
        )
    return results


@pytest.mark.parametrize("wgrad_quant", ["triton", "torch"])
def test_pass_with_exact_division_is_bitwise_equal_to_no_pass(wgrad_quant):
    """Under full AC and `emulate_precision_casts`, the pass with exact division changes no bit of the output or the gradients."""
    torch.manual_seed(0)
    block = GatedMLP()
    for linear in (block.gate_up_proj, block.down_proj):
        nn.init.trunc_normal_(linear.weight, std=0.02)
    block = block.cuda().to(torch.bfloat16)

    pass_options = INDUCTOR_OPTIONS | {"post_grad_custom_pre_pass": Fp8QuantPass(wgrad_quant, ["linear"], "exact")}
    with_pass, codes = run_and_get_code(_steps, _compiled_block(block, pass_options))
    without_pass = _steps(_compiled_block(block, INDUCTOR_OPTIONS))

    backward_codes = [code for code in codes if "torch.ops.prime_rl.fp8_blockwise_mm_dgrad_prequant.default(" in code]
    forward_codes = [code for code in codes if code not in backward_codes]
    assert len(backward_codes) == 1 and len(forward_codes) == 1, (
        "expected one forward and one backward graph for all token counts"
    )
    assert "torch.ops.prime_rl.fp8_blockwise_mm_prequant.default(" in forward_codes[0]
    for code in codes:
        assert "torch.ops.prime_rl.fp8_blockwise_mm.default(" not in code
        assert "torch.ops.prime_rl.fp8_blockwise_mm_backward.default(" not in code
    for tokens, got, want in zip(TOKENS, with_pass, without_pass):
        assert got.keys() == want.keys()
        for name in want:
            assert torch.equal(got[name].view(torch.int16), want[name].view(torch.int16)), f"{name} at {tokens} tokens"
