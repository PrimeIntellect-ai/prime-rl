import pytest
import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFGptOssForCausalLM

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM


def _config() -> GptOssConfig:
    return GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=4,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_experts_per_tok=2,
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
    )


def test_gpt_oss_checkpoint_conversion_roundtrip():
    config = _config()
    with torch.device("meta"):
        hf_model = HFGptOssForCausalLM(config)
        prime_model = PrimeRLGptOssForCausalLM(config)

    hf_state_dict = {name: torch.randn(tensor.shape) for name, tensor in hf_model.state_dict().items()}
    expected_hf = {name: tensor.clone() for name, tensor in hf_state_dict.items()}
    prime_state_dict = prime_model.convert_to_prime(hf_state_dict)

    assert prime_model.is_prime_state_dict(prime_state_dict)
    assert not prime_model.is_hf_state_dict(prime_state_dict)
    assert set(prime_state_dict) == set(prime_model.state_dict())
    for name, tensor in prime_state_dict.items():
        assert tensor.shape == prime_model.state_dict()[name].shape, name

    roundtrip = prime_model.convert_to_hf(prime_state_dict)
    assert roundtrip.keys() == expected_hf.keys()
    for name, tensor in roundtrip.items():
        torch.testing.assert_close(tensor, expected_hf[name])


@pytest.mark.gpu
def test_gpt_oss_moe_matches_hf():
    config = _config()
    with torch.device("cuda"):
        hf_model = HFGptOssForCausalLM(config).to(torch.bfloat16)
        prime_model = PrimeRLGptOssForCausalLM(config).to(torch.bfloat16)

    state_dict = hf_model.state_dict()
    prime_model.convert_to_prime(state_dict)
    prime_model.load_state_dict(state_dict)

    hidden_states = torch.randn(2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    expected, _ = hf_model.model.layers[0].mlp(hidden_states)
    actual = prime_model.model.layers[0].mlp(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
