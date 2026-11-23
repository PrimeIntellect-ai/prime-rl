import pytest
import torch
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen2 import Qwen2ForCausalLM as PrimeRLQwen2ForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs(tie_word_embeddings: bool):
    hf_config = Qwen2Config(
        hidden_size=1024,
        intermediate_size=2048,
        max_position_embeddings=4096,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=3,
        vocab_size=32000,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=tie_word_embeddings,
    )
    hf_config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_model = HFQwen2ForCausalLM._from_config(hf_config)
        prime_model = PrimeRLQwen2ForCausalLM._from_config(hf_config)
    with torch.no_grad():
        # HF initialises biases to zero, which would hide a misplaced q/k/v bias.
        for layer in hf_model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.bias.normal_(std=0.5)
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_qwen2(tie_word_embeddings: bool):
    hf_model, prime_model = get_model_pairs(tie_word_embeddings)

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids, seq_lens=torch.tensor([input_ids.shape[1]], device="cuda"))
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-0), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    q_bias_grad_diff = (
        hf_model.model.layers[0].self_attn.q_proj.bias.grad - prime_model.model.layers[0].self_attn.q_proj.bias.grad
    )
    assert torch.allclose(q_bias_grad_diff, torch.zeros_like(q_bias_grad_diff), atol=1000), (
        f"Max q_proj bias grad diff: {q_bias_grad_diff.abs().max()}"
    )

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        hf_from_prime_model = HFQwen2ForCausalLM._from_config(hf_model.config)
        hf_from_prime_model.load_state_dict(prime_model.convert_to_hf(prime_model.state_dict()))

    hf_from_prime_output = hf_from_prime_model(input_ids, position_ids)
    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-0), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
