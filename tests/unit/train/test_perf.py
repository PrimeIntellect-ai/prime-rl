import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from prime_rl.trainer.perf import PerfCounter
from prime_rl.utils.flops import forward_flops


@pytest.mark.parametrize(
    "model_name, active_params, flops_per_token",
    [("Qwen/Qwen3-0.6B", 595_984_384, 4_280_549_376), ("Jackmin108/debug-moe-0.5B", 256_442_368, 1_840_644_096)],
)
def test_perf_counter(model_name: str, active_params: int, flops_per_token: int):
    # This speeds up the model loading as its a fake device
    with torch.device("meta"):
        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "pad_token_id", None) is None:
            gen_config = GenerationConfig.from_model_config(config)
            # Use `is not None` instead of truthiness: token ID 0 is valid.
            config.pad_token_id = next(
                (
                    v
                    for v in [gen_config.pad_token_id, gen_config.eos_token_id, getattr(config, "eos_token_id", None)]
                    if v is not None
                ),
                None,
            )
        model = AutoModelForCausalLM.from_config(config)
    perf_counter = PerfCounter(model, seq_len=1024)

    assert forward_flops(config)[0] == 2 * active_params
    assert perf_counter.num_flop_per_token == flops_per_token, (
        f"Expected {flops_per_token:,} FLOPS per token, got {perf_counter.num_flop_per_token:,} FLOPS per token"
    )
