import pytest
import torch
from torch import nn
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.models.llama import LlamaForCausalLM as PrimeRLLlamaForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    hf_config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=2048,
        max_position_embeddings=4096,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=3,
        vocab_size=32000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )
    hf_config._attn_implementation = "sdpa"
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFLlamaForCausalLM._from_config(hf_config)
        prime_model = PrimeRLLlamaForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_llama_attn_only():
    hf_model, prime_model = get_model_pairs()
    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()
    for layer in prime_model.model.layers:
        layer.mlp = nn.Identity()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_llama_mlp_only():
    hf_model, prime_model = get_model_pairs()

    def foo(hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = foo
    for layer in prime_model.model.layers:
        layer.self_attn.forward = foo

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_llama():
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids)
    prime_output = prime_model(input_ids, position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFLlamaForCausalLM._from_config(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_from_prime_output = hf_from_prime_model(input_ids, position_ids)
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


if __name__ == "__main__":
    test_llama_mlp_only()
