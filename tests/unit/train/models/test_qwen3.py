import importlib.util

import pytest
import torch
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, supports_custom_impl
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.layers.rms_norm import RMSNorm
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM as PrimeRLQwen3ForCausalLM
from prime_rl.utils.utils import default_dtype

QUACK_INSTALLED = importlib.util.find_spec("quack") is not None
requires_quack_rms = pytest.mark.skipif(
    not (QUACK_INSTALLED and torch.cuda.is_available()),
    reason="quack RMSNorm requires quack-kernels and CUDA",
)


def make_qwen3_config() -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=2048,
        max_position_embeddings=4096,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=3,
        head_dim=128,
        vocab_size=32000,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        attention_bias=False,
        use_sliding_window=False,
    )
    config._attn_implementation = "sdpa"
    return config


def get_model_pairs():
    hf_config = make_qwen3_config()
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFQwen3ForCausalLM._from_config(hf_config)
        prime_model = PrimeRLQwen3ForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_qwen3_supports_custom_impl():
    config = make_qwen3_config()
    assert supports_custom_impl(config)

    model = AutoModelForCausalLMPrimeRL.from_config(config)
    assert isinstance(model, PrimeRLQwen3ForCausalLM)


@pytest.mark.gpu
def test_qwen3():
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


@pytest.mark.gpu
@requires_quack_rms
def test_qwen3_quack_rms_norm_propagates():
    config = make_qwen3_config()
    config.rms_norm_impl = "quack"

    with torch.device("cuda"), default_dtype(torch.float32):
        model = PrimeRLQwen3ForCausalLM._from_config(config)

    assert isinstance(model.model.layers[0].input_layernorm, RMSNorm)
    assert model.model.layers[0].input_layernorm.impl == "quack"
    assert model.model.layers[0].post_attention_layernorm.impl == "quack"
    assert model.model.layers[0].self_attn.q_norm.impl == "quack"
    assert model.model.layers[0].self_attn.k_norm.impl == "quack"
    assert model.model.norm.impl == "quack"
