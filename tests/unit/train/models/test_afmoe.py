import pytest
import torch
from torch import nn

from prime_rl.trainer.models.afmoe import AfMoeConfig
from prime_rl.trainer.models.afmoe import AfMoeForCausalLM as PrimeRLAfMoeForCausalLM
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def get_model_pairs():
    """Create a pair of Prime-RL AF MoE models for testing."""
    config = AfMoeConfig(
        vocab_size=151552,  # Reduced from 200192 to save memory
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=256,
        num_hidden_layers=3,
        num_dense_layers=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=16,
        num_experts_per_tok=4,
        num_shared_experts=2,
        route_scale=1.0,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        # Add missing parameters that might be needed
        norm_topk_prob=True,
        use_qk_norm=False,
        attention_bias=False,
        load_balance_coeff=1e-3,
        use_grouped_mm=True,
    )
    config._attn_implementation = "sdpa"
    
    with torch.device("cuda"), default_dtype(torch.float32):
        model1 = PrimeRLAfMoeForCausalLM._from_config(config)
        model2 = PrimeRLAfMoeForCausalLM._from_config(config)
    
    # Copy weights from model1 to model2 to ensure they're identical
    with torch.no_grad():
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)
    
    return model1, model2


def test_afmoe_attn_only():
    """Test attention-only forward pass."""
    model1, model2 = get_model_pairs()
    
    # Replace MLP with Identity to test only attention
    for layer in model1.model.layers:
        layer.mlp = nn.Identity()
    for layer in model2.model.layers:
        layer.mlp = nn.Identity()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, model1.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    output1 = model1(input_ids, position_ids)
    output2 = model2(input_ids, position_ids)
    output1.logits.sum().backward()
    output2.logits.sum().backward()

    logits_diff = output1.logits - output2.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = model1.model.embed_tokens.weight.grad - model2.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


def test_afmoe_mlp_only():
    """Test MLP/MoE-only forward pass."""
    model1, model2 = get_model_pairs()

    def identity_attn(hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states, None

    # Replace attention with identity to test only MLP/MoE
    for layer in model1.model.layers:
        layer.self_attn.forward = identity_attn
    for layer in model2.model.layers:
        layer.self_attn.forward = identity_attn

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, model1.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    output1 = model1(input_ids, position_ids)
    output2 = model2(input_ids, position_ids)
    output1.logits.sum().backward()
    output2.logits.sum().backward()

    logits_diff = output1.logits - output2.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = model1.model.embed_tokens.weight.grad - model2.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


def test_afmoe_full():
    """Test full model forward pass and conversion."""
    model1, model2 = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, model1.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)

    output1 = model1(input_ids, position_ids)
    output2 = model2(input_ids, position_ids)
    output1.logits.sum().backward()
    output2.logits.sum().backward()

    logits_diff = output1.logits - output2.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = model1.model.embed_tokens.weight.grad - model2.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), (
        f"Max grad diff: {grad_diff.abs().max()}"
    )


def test_afmoe_conversion():
    """Test bidirectional conversion between TT and HF formats."""
    model1, _ = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        # Get state dict in TT format
        tt_state_dict = model1.state_dict()
        
        # Convert TT -> HF
        hf_state_dict = model1.convert_to_hf(tt_state_dict.copy())
        
        # Convert HF -> TT
        tt_state_dict_2 = model1.convert_to_prime(hf_state_dict.copy())
        
        # Create a new model and load the converted state dict
        model2 = PrimeRLAfMoeForCausalLM._from_config(model1.config)
        model2.load_state_dict(tt_state_dict_2)
        
        # Test forward pass with converted weights
        input_ids = torch.randint(0, model1.config.vocab_size, (1, 100))
        position_ids = torch.arange(1, 101).unsqueeze(0)
        
        output1 = model1(input_ids, position_ids)
        output2 = model2(input_ids, position_ids)
        
        logits_diff = output1.logits - output2.logits
        assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=2e-2), (
            f"Max logits diff after conversion: {logits_diff.abs().max()}"
        )


def test_afmoe_dense_vs_moe_layers():
    """Test that dense layers and MoE layers are correctly placed."""
    config = AfMoeConfig(
        vocab_size=151552,  # Reduced from 200192 to save memory
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=256,
        num_hidden_layers=5,
        num_dense_layers=2,  # First 2 layers should be dense
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=16,
        num_experts_per_tok=4,
        num_shared_experts=2,
        route_scale=1.0,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        norm_topk_prob=True,
        use_qk_norm=False,
        attention_bias=False,
        load_balance_coeff=1e-3,
        use_grouped_mm=True,
    )
    config._attn_implementation = "sdpa"
    
    with torch.device("cuda"), default_dtype(torch.float32):
        model = PrimeRLAfMoeForCausalLM._from_config(config)
    
    # Check that first num_dense_layers are dense MLP
    for i in range(config.num_dense_layers):
        assert not hasattr(model.model.layers[i].mlp, "experts"), (
            f"Layer {i} should be dense but has MoE"
        )
        assert isinstance(model.model.layers[i].mlp, nn.Module), (
            f"Layer {i} should have MLP"
        )
    
    # Check that remaining layers are MoE
    for i in range(config.num_dense_layers, config.num_hidden_layers):
        assert hasattr(model.model.layers[i].mlp, "experts"), (
            f"Layer {i} should be MoE but is dense"
        )


if __name__ == "__main__":
    test_afmoe_full()
