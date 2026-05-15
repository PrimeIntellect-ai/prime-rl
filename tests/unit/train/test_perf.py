import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from prime_rl.trainer.perf import PerfCounter


@pytest.mark.parametrize(
    "model_name, active_params, flops_per_token",
    [("Qwen/Qwen3-0.6B", 595_984_384, 3_928_227_840), ("Jackmin108/debug-moe-0.5B", 256_442_368, 1_689_649_152)],
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

    assert perf_counter.get_active_mm_params(config) == active_params, (
        f"Expected {active_params:,} active parameters, got {perf_counter.get_active_mm_params(config):,} active parameters"
    )
    assert perf_counter.num_flop_per_token == flops_per_token, (
        f"Expected {flops_per_token:,} FLOPS per token, got {perf_counter.num_flop_per_token:,} FLOPS per token"
    )


def test_perf_counter_estimate_batch_flops_uses_actual_sequence_lengths():
    with torch.device("meta"):
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        if getattr(config, "pad_token_id", None) is None:
            gen_config = GenerationConfig.from_model_config(config)
            config.pad_token_id = next(
                (
                    v
                    for v in [gen_config.pad_token_id, gen_config.eos_token_id, getattr(config, "eos_token_id", None)]
                    if v is not None
                ),
                None,
            )
        model = AutoModelForCausalLM.from_config(config)

    perf_counter = PerfCounter(model, seq_len=8, window_size=10)
    sequence_lengths = [5, 3]
    processed_tokens = sum(sequence_lengths)
    expected = perf_counter.non_attention_flop_per_token * processed_tokens + perf_counter.attention_flop_per_seq_sq * (
        5 * 5 + 3 * 3
    )
    assert perf_counter.estimate_batch_flops(processed_tokens, sequence_lengths) == expected
