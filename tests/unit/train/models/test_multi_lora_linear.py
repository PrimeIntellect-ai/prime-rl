import pytest
import torch
from torch import nn

from prime_rl.trainer.models.layers.lora import base as lora_base
from prime_rl.trainer.models.layers.lora.base import set_lora_num_tokens, set_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def reset_multilora_globals():
    yield
    lora_base.LORA_NUM_TOKENS = None
    lora_base.SCALING_FACTORS = None


def reference_forward(module: MultiLoRALinear, x: torch.Tensor) -> torch.Tensor:
    outputs = []
    start = 0
    for idx, count in enumerate(module._lora_num_tokens.tolist()):
        x_i = x[start : start + count]
        delta = x_i @ module.lora_A[idx].T @ module.lora_B[idx].T
        outputs.append(module.base_layer(x_i) + module._scaling_factors[idx] * delta)
        start += count
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("use_grouped_mm", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_multi_lora_linear_matches_reference(use_grouped_mm: bool):
    dtype = torch.bfloat16 if use_grouped_mm else torch.float32
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.manual_seed(0)

    num_tokens = torch.tensor([5, 0, 11], dtype=torch.int32, device=DEVICE)
    set_lora_num_tokens(num_tokens, reset_reference=True)
    set_multilora_scaling(torch.tensor([0.5, 1.0, 2.0], dtype=dtype, device=DEVICE), reset_reference=True)

    base = nn.Linear(32, 48, bias=False, device=DEVICE, dtype=dtype)
    module = MultiLoRALinear(base, rank=8, n_adapters=3, use_grouped_mm=use_grouped_mm)
    with torch.no_grad():
        for lora_B in module.lora_B:
            lora_B.normal_()

    x = torch.randn(int(num_tokens.sum()), 32, device=DEVICE, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(x.shape[0], 48, device=DEVICE, dtype=dtype)
    params = [x, *module.lora_A, *module.lora_B]

    out = module(x)
    grads = torch.autograd.grad(out, params, grad_out)
    ref_out = reference_forward(module, x)
    ref_grads = torch.autograd.grad(ref_out, params, grad_out)

    torch.testing.assert_close(out, ref_out, atol=atol, rtol=atol)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, atol=atol, rtol=atol)
