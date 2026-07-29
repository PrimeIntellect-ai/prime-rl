import pytest
import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFGptOssForCausalLM

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM
from prime_rl.trainer.models.layers.moe import broadcast_expert_bias


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


@pytest.mark.gpu
def test_gpt_oss_expert_bias_broadcast_accepts_router_counts():
    bias = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda")
    selected_experts = torch.tensor([0, 0, 2], device="cuda")
    num_tokens_per_expert = torch.histc(selected_experts, bins=3, min=0, max=3)

    result = broadcast_expert_bias(bias, num_tokens_per_expert, target_rows=4)

    expected = torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [0.0, 0.0]], device="cuda")
    torch.testing.assert_close(result, expected)
