import pytest
import torch
from torch import nn

from tests.unit.train.models.afmoe_hf_modeling.configuration_afmoe import AfmoeConfig
from tests.unit.train.models.afmoe_hf_modeling.modeling_afmoe import (
    AfmoeForCausalLM as HFAfmoeForCausalLM,
)
from prime_rl.trainer.models.afmoe import AfmoeForCausalLM as PrimeRLAfmoeForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    hf_config = AfmoeConfig(
        architectures=["AfmoeForCausalLM"],
        attention_dropout=0.0,
        global_attn_every_n_layers=2,
        head_dim=64,
        hidden_act="silu",
        hidden_size=256,
        initializer_range=0.02,
        intermediate_size=512,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        load_balance_coeff=0.001,
        max_position_embeddings=512,
        moe_intermediate_size=128,
        mup_enabled=True,
        n_group=1,
        num_attention_heads=4,
        num_dense_layers=2,
        num_expert_groups=1,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        num_key_value_heads=2,
        num_limited_groups=1,
        num_shared_experts=1,
        rms_norm_eps=1e-5,
        rope_scaling=None,
        rope_theta=10000.0,
        route_norm=True,
        route_scale=2.826,
        score_func="sigmoid",
        sliding_window=128,
        tie_word_embeddings=False,
        topk_group=1,
        use_cache=True,
        use_grouped_mm=True,
        vocab_size=256,
    )
    hf_config._attn_implementation = "sdpa"
    hf_config.use_grouped_mm = False
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFAfmoeForCausalLM._from_config(hf_config)
        prime_model = PrimeRLAfmoeForCausalLM._from_config(hf_config)
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_afmoe_attn_only() -> None:
    hf_model, prime_model = get_model_pairs()
    for layer in hf_model.model.layers:
        layer.mlp = nn.Identity()
    for layer in prime_model.model.layers:
        layer.mlp = nn.Identity()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_afmoe_mlp_only() -> None:
    hf_model, prime_model = get_model_pairs()

    # HF AfmoeAttention.forward() returns a single tensor
    def identity_attn_hf(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states

    # PrimeRL attention returns a tuple (output, attn_weights)
    def identity_attn_prime(hidden_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, None]:
        return hidden_states, None

    for layer in hf_model.model.layers:
        layer.self_attn.forward = identity_attn_hf
    for layer in prime_model.model.layers:
        layer.self_attn.forward = identity_attn_prime

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


def test_afmoe() -> None:
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, hf_model.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output.logits.sum().backward()

    logits_diff = prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embed_tokens.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_from_prime_model = HFAfmoeForCausalLM._from_config(hf_model.config)
        converted_state_dict = prime_model.convert_to_hf(prime_model.state_dict())
        hf_from_prime_model.load_state_dict(converted_state_dict)

    hf_from_prime_output = hf_from_prime_model(input_ids, position_ids=position_ids)
    hf_from_prime_output.logits.sum().backward()

    logits_diff = hf_from_prime_output.logits - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_from_prime_model.model.embed_tokens.weight.grad - hf_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


if __name__ == "__main__":
    test_afmoe()
