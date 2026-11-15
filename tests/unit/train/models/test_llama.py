import pytest
import torch
from torch import nn
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.llama import LlamaForCausalLM as PrimeRLLlamaForCausalLM
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.parity import assert_matches_hf, hf_checkpoint_state_dict

pytestmark = [pytest.mark.gpu]

TOL = 2e-2


def get_model_pairs(tmp_path):
    torch.manual_seed(0)
    hf_config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    hf_config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_model = HFLlamaForCausalLM._from_config(hf_config)
        prime_model = PrimeRLLlamaForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_checkpoint_state_dict(hf_model, tmp_path)
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    # Training code wraps the LM head; tests should mirror that (so forward can accept labels/temperature).
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_llama_attn_only(tmp_path):
    hf_model, prime_model = get_model_pairs(tmp_path)
    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()
    for layer in prime_model.model.layers:
        layer.mlp = nn.Identity()
    assert_matches_hf(hf_model, prime_model, TOL)


def test_llama_mlp_only(tmp_path):
    hf_model, prime_model = get_model_pairs(tmp_path)

    def foo(hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = foo
    for layer in prime_model.model.layers:
        layer.self_attn.forward = foo
    assert_matches_hf(hf_model, prime_model, TOL)


def test_llama(tmp_path):
    hf_model, prime_model = get_model_pairs(tmp_path)
    assert_matches_hf(hf_model, prime_model, TOL)

    converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
    hf_state_dict = hf_model.state_dict()
    assert converted_state_dict.keys() == hf_state_dict.keys()
    for name, tensor in hf_state_dict.items():
        torch.testing.assert_close(converted_state_dict[name], tensor, rtol=0, atol=0)
