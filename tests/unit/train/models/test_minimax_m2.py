import pytest
import torch
from transformers import MiniMaxM2Config as HFMiniMaxM2Config
from transformers import MiniMaxM2ForCausalLM as HFMiniMaxM2ForCausalLM

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.parity import assert_matches_hf, hf_checkpoint_state_dict

pytestmark = [pytest.mark.gpu]


def test_minimax_m2_matches_hf(tmp_path):
    torch.manual_seed(0)
    config_kwargs = dict(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rotary_dim=32,
        max_position_embeddings=512,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    hf_config = HFMiniMaxM2Config(**config_kwargs)
    hf_config._attn_implementation = "flash_attention_2"
    prime_config = MiniMaxM2Config(**config_kwargs)
    prime_config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_model = HFMiniMaxM2ForCausalLM._from_config(hf_config)
        prime_model = MiniMaxM2ForCausalLM._from_config(prime_config)
    with torch.no_grad():
        state_dict = hf_checkpoint_state_dict(hf_model, tmp_path)
        prime_model.load_state_dict(prime_model.convert_to_prime(state_dict))
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert_matches_hf(hf_model, prime_model, tol=2e-2)
