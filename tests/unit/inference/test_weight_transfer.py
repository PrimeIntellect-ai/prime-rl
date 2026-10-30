import pytest
import torch

from prime_rl.inference.vllm.worker.weight_transfer import check_fp8_source


def test_fp8_params_only_take_fp8_tensors():
    check_fp8_source("w", torch.float8_e4m3fn, torch.float8_e4m3fn)
    check_fp8_source("w", torch.bfloat16, torch.bfloat16)
    with pytest.raises(ValueError, match="fp8_per_block"):
        check_fp8_source("w", torch.float8_e4m3fn, torch.bfloat16)
