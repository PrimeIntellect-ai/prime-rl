import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from prime_rl.trainer.perf import PerfCounter


@pytest.mark.parametrize(
    "model_name, active_params, flops_per_token",
    [("Qwen/Qwen3-0.6B", 595_984_384, 3_928_227_840), ("Jackmin108/debug-moe-0.5B", 256_442_368, 1_689_649_152)],
)
def test_perf_counter(model_name: str, active_params: int, flops_per_token: int):
    # This speeds up the model loading as its a fake device
    with torch.device("meta"):
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
    perf_counter = PerfCounter(model, seq_len=1024, window_size=10)

    assert perf_counter.get_active_mm_params(config) == active_params, (
        f"Expected {active_params:,} active parameters, got {perf_counter.get_active_mm_params(config):,} active parameters"
    )
    assert perf_counter.num_flop_per_token == flops_per_token, (
        f"Expected {flops_per_token:,} FLOPS per token, got {perf_counter.num_flop_per_token:,} FLOPS per token"
    )
