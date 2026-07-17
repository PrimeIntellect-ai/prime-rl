import pytest
import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig as HFNemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
)
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM as HFNemotronHForCausalLM,
)

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHAttentionLayer
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

# Shared small-model hyperparams (satisfy mamba_expand * hidden_size == mamba_num_heads * mamba_head_dim)
_BASE = dict(
    vocab_size=256,
    hidden_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=128,
    intermediate_size=512,
    mamba_expand=2,
    mamba_num_heads=8,
    mamba_head_dim=64,
    ssm_state_size=64,
    mamba_n_groups=1,
    mamba_d_conv=4,
    mamba_chunk_size=64,
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=256,
    moe_shared_expert_intermediate_size=256,
    moe_latent_size=128,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
)


def get_model_pairs():
    """Create an HF model and a PrimeRL model with shared weights."""
    hf_config = HFNemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    hf_config._attn_implementation = "sdpa"

    prime_config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    prime_config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFNemotronHForCausalLM._from_config(hf_config)
        prime_model = NemotronHForCausalLM._from_config(prime_config)

    with torch.no_grad():
        state_dict = hf_model.state_dict()
        prime_state_keys = prime_model.state_dict().keys()
        NemotronHForCausalLM.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)

    inject_prime_lm_head(prime_model, chunk_size=None)
    assert set(prime_state_keys) - set(state_dict.keys()) == set()
    return hf_model, prime_model


def test_nemotron_h_mamba_moe_only():
    """Test Mamba and MoE layers produce identical outputs (attention bypassed)."""
    hf_model, prime_model = get_model_pairs()

    # Bypass attention layers in both models so only Mamba+MoE are exercised
    for layer in hf_model.model.layers:
        if isinstance(layer.mixer, NemotronHAttention):
            layer.forward = lambda hidden_states, **kwargs: hidden_states

    for layer in prime_model.model.layers:
        if isinstance(layer, NemotronHAttentionLayer):
            layer.forward = lambda hidden_states, **kwargs: hidden_states

    torch.manual_seed(42)
    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, 256, (1, 32))
        position_ids = torch.arange(0, 32).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    hf_output.logits.sum().backward()
    prime_output["logits"].sum().backward()

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )
    grad_diff = hf_model.model.embeddings.weight.grad - prime_model.model.embed_tokens.weight.grad
    assert torch.allclose(grad_diff, torch.zeros_like(grad_diff), atol=2), f"Max grad diff: {grad_diff.abs().max()}"


@pytest.mark.xfail(reason="HF NemotronH now uses fused expert tensors; convert_to_hf produces individual expert format")
def test_nemotron_h_reverse():
    """Test reverse: PrimeRL weights loaded into HF model produce identical outputs."""
    prime_config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    prime_config._attn_implementation = "sdpa"

    hf_config = HFNemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    hf_config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        prime_model = NemotronHForCausalLM._from_config(prime_config)
        hf_model = HFNemotronHForCausalLM._from_config(hf_config)

    inject_prime_lm_head(prime_model, chunk_size=None)

    with torch.no_grad():
        sd = prime_model.state_dict()
        NemotronHForCausalLM.convert_to_hf(sd)
        # convert_to_hf produces checkpoint format with "backbone." prefix;
        # the HF model uses "model." prefix for its state dict
        keys_to_rename = [k for k in sd if k.startswith("backbone.")]
        for key in keys_to_rename:
            sd["model." + key[len("backbone.") :]] = sd.pop(key)
        hf_model.load_state_dict(sd)

    # Bypass attention to isolate Mamba+MoE matching
    for layer in hf_model.model.layers:
        if isinstance(layer.mixer, NemotronHAttention):
            layer.forward = lambda hidden_states, **kwargs: hidden_states
    for layer in prime_model.model.layers:
        if isinstance(layer, NemotronHAttentionLayer):
            layer.forward = lambda hidden_states, **kwargs: hidden_states

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, 256, (1, 32))
        position_ids = torch.arange(0, 32).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)

    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=1e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )


def test_nemotron_h():
    """Test full model (Mamba + MoE + Attention) produces close outputs."""
    hf_model, prime_model = get_model_pairs()

    with torch.device("cuda"), default_dtype(torch.float32):
        input_ids = torch.randint(0, 256, (1, 32))
        position_ids = torch.arange(0, 32).unsqueeze(0)

    hf_output = hf_model(input_ids, position_ids=position_ids)
    prime_output = prime_model(input_ids, position_ids=position_ids)
    # Slightly larger tolerance due to different SDPA attention implementations
    logits_diff = prime_output["logits"] - hf_output.logits
    assert torch.allclose(logits_diff, torch.zeros_like(logits_diff), atol=5e-2), (
        f"Max logits diff: {logits_diff.abs().max()}"
    )


def test_nemotron_h_backward():
    """Verify all parameters receive non-zero gradients."""
    prime_config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    model = NemotronHForCausalLM(prime_config).to("cuda")
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    output = model(input_ids)
    output["logits"].sum().backward()

    zero_grads = []
    for name, p in model.named_parameters():
        if p.numel() == 0:
            continue
        if p.grad is None or p.grad.norm().item() == 0:
            zero_grads.append(name)
    assert not zero_grads, f"Parameters with zero/no gradients: {zero_grads}"


def test_nemotron_h_weight_conversion_roundtrip():
    """Verify PrimeRL -> HF -> PrimeRL conversion preserves all weights."""
    prime_config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    model = NemotronHForCausalLM(prime_config).to("cuda")
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}

    sd = model.state_dict()
    NemotronHForCausalLM.convert_to_hf(sd)
    assert NemotronHForCausalLM.is_hf_state_dict(sd)
    NemotronHForCausalLM.convert_to_prime(sd)
    assert NemotronHForCausalLM.is_prime_state_dict(sd)

    for key in original_sd:
        assert key in sd, f"Missing key after roundtrip: {key}"
        assert torch.equal(original_sd[key], sd[key]), f"Value mismatch for {key}"


def test_nemotron_h_hybrid_override_pattern():
    """Verify hybrid_override_pattern correctly maps to layers_block_type."""
    config = NemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    assert config.layers_block_type == ["mamba", "moe", "attention", "moe"]
    assert config.num_hidden_layers == 4


def test_nemotron_h_no_latent_projection():
    """Verify model works without latent projections (moe_latent_size=None)."""
    prime_config = NemotronHConfig(
        **{**_BASE, "moe_latent_size": None},
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    model = NemotronHForCausalLM(prime_config).to("cuda")
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    output = model(input_ids)
    assert output["logits"].shape == (2, 16, 256)

    output["logits"].sum().backward()
    for name, p in model.named_parameters():
        if "experts.w1" in name and p.numel() > 0:
            assert p.grad is not None and p.grad.norm().item() > 0, f"Zero grad for {name}"
