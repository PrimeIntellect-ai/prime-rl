import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFGptOssForCausalLM

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM


def test_gpt_oss_checkpoint_format_matches_hf():
    config = GptOssConfig(
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

    with torch.device("meta"):
        hf_model = HFGptOssForCausalLM(config)
        prime_model = PrimeRLGptOssForCausalLM(config)

    hf_state_dict = hf_model.state_dict()
    prime_state_dict = prime_model.state_dict()
    assert set(prime_state_dict) == set(hf_state_dict)
    for name, tensor in prime_state_dict.items():
        assert tensor.shape == hf_state_dict[name].shape, name
