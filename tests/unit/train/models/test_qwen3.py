import pytest
import torch
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM as PrimeRLQwen3ForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    hf_config = Qwen3Config(
        head_dim=32,
        hidden_size=256,
        intermediate_size=512,
        max_position_embeddings=4096,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        attention_bias=True,
    )
    hf_config._attn_implementation = "sdpa"
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3ForCausalLM._from_config(hf_config)
        prime_model = PrimeRLQwen3ForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_qwen3():
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 32))
        position_ids = torch.arange(1, 33).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"
