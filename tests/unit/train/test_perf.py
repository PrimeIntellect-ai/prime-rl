import pytest
import torch
from transformers import AutoConfig

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.perf import PerfCounter


# Expected values are torchtitan's `get_nparams_and_flops(model, seq_len=1024)` for its matching flavors.
@pytest.mark.parametrize(
    "model_name, flops_per_token",
    [("Qwen/Qwen3-0.6B", 4_280_942_592), ("Qwen/Qwen3-30B-A3B", 20_667_125_760)],
)
def test_flops_per_token_matches_torchtitan(model_name: str, flops_per_token: int):
    config = AutoConfig.from_pretrained(model_name, attn_implementation="flash_attention_2")
    config.pad_token_id = config.eos_token_id
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(config)
    assert PerfCounter(model, seq_len=1024).num_flop_per_token == flops_per_token
