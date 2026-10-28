import pytest
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from prime_rl.inference.vllm.worker.weight_transfer import check_weight_update_supported


def test_weight_updates_refuse_fp8_checkpoints():
    check_weight_update_supported(None)
    with pytest.raises(ValueError, match="fp8_per_block"):
        check_weight_update_supported(Fp8Config(weight_block_size=[128, 128]))
