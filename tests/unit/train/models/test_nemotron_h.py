from unittest.mock import MagicMock

import pytest
import torch

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from prime_rl.trainer.models.nemotron_h.mamba import NemotronHMamba2
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

_BASE = dict(
    attn_implementation="flash_attention_2",
    vocab_size=256,
    hidden_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=128,
    intermediate_size=512,
    expand=2,
    mamba_num_heads=8,
    mamba_head_dim=64,
    ssm_state_size=64,
    n_groups=1,
    conv_kernel=4,
    chunk_size=64,
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=256,
    moe_shared_expert_intermediate_size=256,
    moe_latent_size=128,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
)


def _seq_lens(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.tensor([input_ids.shape[1]], device=input_ids.device)


def test_nemotron_h_reverse():
    """PrimeRL weights convert back to the source checkpoint layout."""
    config = NemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    with torch.device("meta"):
        prime_model = NemotronHForCausalLM(config)
    converted = prime_model.convert_to_hf(dict(prime_model.state_dict()))

    assert prime_model.is_hf_state_dict(converted)
    assert "backbone.layers.1.mixer.experts.0.up_proj.weight" in converted
    assert not any("mlp.experts.up_proj" in name for name in converted)


def test_nemotron_h_backward():
    """Verify all parameters receive non-zero gradients."""
    prime_config = NemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = NemotronHForCausalLM(prime_config)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 16), device="cuda")
    output = model(input_ids, seq_lens=_seq_lens(input_ids))
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
    prime_config = NemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    model = NemotronHForCausalLM(prime_config).to("cuda")
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}

    sd = model.state_dict()
    model.convert_to_hf(sd)
    assert NemotronHForCausalLM.is_hf_state_dict(sd)
    model.convert_to_prime(sd)
    assert NemotronHForCausalLM.is_prime_state_dict(sd)

    for key in original_sd:
        assert key in sd, f"Missing key after roundtrip: {key}"
        assert torch.equal(original_sd[key], sd[key]), f"Value mismatch for {key}"


def test_nemotron_h_layer_types():
    expected = ["mamba", "moe", "attention", "moe"]
    pattern_config = NemotronHConfig(**_BASE, hybrid_override_pattern="ME*E")
    list_config = NemotronHConfig(**_BASE, layers_block_type=expected)

    assert pattern_config.layer_types == expected
    assert list_config.layer_types == expected
    assert pattern_config.num_hidden_layers == list_config.num_hidden_layers == 4


def test_nemotron_h_context_parallel_setup_finds_wrapped_mamba_layer():
    config = NemotronHConfig(
        **(_BASE | {"n_groups": 2}),
        hybrid_override_pattern="ME*E",
    )
    with torch.device("meta"):
        model = NemotronHForCausalLM(config)

    mamba_layer = model.model.layers[0]
    assert isinstance(mamba_layer.mamba, NemotronHMamba2)
    model.model.layers[0] = torch.nn.Sequential(mamba_layer)

    cp_group = MagicMock()

    for module in model.modules():
        if hasattr(module, "setup_context_parallel"):
            module.setup_context_parallel(cp_group, 1, 2, "ulysses")

    assert mamba_layer.mamba.cp_group is cp_group
    assert mamba_layer.mamba.cp_rank == 1
    assert mamba_layer.mamba.cp_world_size == 2


def test_nemotron_h_no_latent_projection():
    """Verify model works without latent projections (moe_latent_size=None)."""
    prime_config = NemotronHConfig(
        **{**_BASE, "moe_latent_size": None},
        hybrid_override_pattern="ME*E",
    )
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = NemotronHForCausalLM(prime_config)
    inject_prime_lm_head(model)

    input_ids = torch.randint(0, 256, (1, 16), device="cuda")
    output = model(input_ids, seq_lens=_seq_lens(input_ids))
    assert output["logits"].shape == (1, 16, 256)

    output["logits"].sum().backward()
    for name, p in model.named_parameters():
        if "experts.up_proj" in name and p.numel() > 0:
            assert p.grad is not None and p.grad.norm().item() > 0, f"Zero grad for {name}"
