import math

import pytest
import torch
from torch import nn

from prime_rl.trainer.models.layers.lora.base import LoRAModule


@pytest.mark.parametrize("shape", [(16, 128), (64, 96), (4, 16, 128), (4, 64, 96)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_initialization_matches_independent_linear_matrices(shape, dtype):
    module = LoRAModule(nn.Identity(), rank=shape[-2], alpha=32, dropout=0)
    module.proj_lora_A = nn.Parameter(torch.empty(shape, dtype=dtype))
    module.proj_lora_B = nn.Parameter(torch.ones((*shape[:-2], 32, shape[-2]), dtype=dtype))
    lora_a, lora_b = module.proj_lora_A, module.proj_lora_B
    expected = torch.empty_like(lora_a)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        module.reset_parameters()
        torch.random.default_generator.manual_seed(0)
        for matrix in expected.reshape(-1, *shape[-2:]):
            nn.init.kaiming_uniform_(matrix, a=math.sqrt(5))

    torch.testing.assert_close(lora_a, expected, rtol=0, atol=0)
    assert torch.count_nonzero(lora_b) == 0
    assert module.proj_lora_A is lora_a
    assert module.proj_lora_B is lora_b
