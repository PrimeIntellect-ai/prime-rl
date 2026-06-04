"""Test KL mismatch computation for NemotronH models.

Creates two small NemotronH models (policy + reference), perturbs
the policy to simulate SFT drift, computes logprobs from both, and
verifies KL mismatch through the loss pipeline.
"""

import pytest
import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from prime_rl.trainer.rl.loss import (
    LossInputs,
    default_loss_fn,
    selective_log_softmax,
    shift_tensor_right,
)
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

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


def _make_model(device="cuda"):
    config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    config._attn_implementation = "sdpa"
    with torch.device(device), default_dtype(torch.float32):
        model = NemotronHForCausalLM._from_config(config)
    inject_prime_lm_head(model, chunk_size=None)
    return model


def _get_logprobs_vanilla(model, input_ids):
    """Get logprobs using VanillaOutputLinear (returns logits, we compute logprobs)."""
    with torch.no_grad():
        out = model(input_ids)
    logits = out["logits"]
    labels = torch.cat(
        [input_ids[:, 1:], torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)], dim=1
    )
    logprobs = selective_log_softmax(logits, labels)
    logprobs = shift_tensor_right(logprobs, pad_value=torch.log(torch.tensor(1.0 / model.config.vocab_size)).item())
    return logprobs


def _perturb_model(model, scale=0.1):
    """Add small noise to model weights to simulate SFT drift."""
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() > 0:
                p.add_(torch.randn_like(p) * scale)


def test_kl_zero_when_identical():
    """Two identical models should have zero KL mismatch."""
    model = _make_model()

    input_ids = torch.randint(0, 256, (2, 32), device="cuda")
    logprobs = _get_logprobs_vanilla(model, input_ids)

    loss_mask = torch.ones(32, dtype=torch.bool, device="cuda")
    # mask out first token (shifted right padding)
    loss_mask[0] = False
    advantages = torch.ones(32, device="cuda")

    for i in range(logprobs.shape[0]):
        inputs = LossInputs(
            trainer_logprobs=logprobs[i],
            inference_logprobs=logprobs[i],
            teacher_logprobs=None,
            advantages=advantages,
            loss_mask=loss_mask,
        )
        result = default_loss_fn(inputs, DefaultLossConfig())

        assert result.metrics["unmasked_mismatch_kl"].item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected zero KL for identical models, got {result.metrics['unmasked_mismatch_kl'].item()}"
        )


def test_kl_positive_after_perturbation():
    """Perturbing the policy model should produce positive KL."""
    ref_model = _make_model()
    policy_model = _make_model()

    # Copy reference weights to policy, then perturb
    with torch.no_grad():
        policy_model.load_state_dict(ref_model.state_dict())
    _perturb_model(policy_model, scale=0.05)

    input_ids = torch.randint(0, 256, (2, 32), device="cuda")

    ref_logprobs = _get_logprobs_vanilla(ref_model, input_ids)
    policy_logprobs = _get_logprobs_vanilla(policy_model, input_ids)

    loss_mask = torch.ones(32, dtype=torch.bool, device="cuda")
    loss_mask[0] = False
    advantages = torch.ones(32, device="cuda")

    for i in range(ref_logprobs.shape[0]):
        inputs = LossInputs(
            trainer_logprobs=policy_logprobs[i],
            inference_logprobs=ref_logprobs[i],
            teacher_logprobs=None,
            advantages=advantages,
            loss_mask=loss_mask,
        )
        result = default_loss_fn(inputs, DefaultLossConfig())
        kl = result.metrics["unmasked_mismatch_kl"].item()

        assert kl > 0, f"Expected positive KL after perturbation, got {kl}"
        assert kl < 100, f"KL unexpectedly large: {kl}"


def test_kl_increases_with_larger_perturbation():
    """Larger perturbation should produce larger KL."""
    ref_model = _make_model()
    policy_small = _make_model()
    policy_large = _make_model()

    with torch.no_grad():
        policy_small.load_state_dict(ref_model.state_dict())
        policy_large.load_state_dict(ref_model.state_dict())

    _perturb_model(policy_small, scale=0.01)
    _perturb_model(policy_large, scale=0.1)

    input_ids = torch.randint(0, 256, (1, 32), device="cuda")

    ref_logprobs = _get_logprobs_vanilla(ref_model, input_ids)
    small_logprobs = _get_logprobs_vanilla(policy_small, input_ids)
    large_logprobs = _get_logprobs_vanilla(policy_large, input_ids)

    loss_mask = torch.ones(32, dtype=torch.bool, device="cuda")
    loss_mask[0] = False

    log_ratio_small = small_logprobs[0] - ref_logprobs[0]
    ratio_small = torch.exp(log_ratio_small)
    kl_small = ((ratio_small - log_ratio_small - 1) * loss_mask).sum() / loss_mask.sum()

    log_ratio_large = large_logprobs[0] - ref_logprobs[0]
    ratio_large = torch.exp(log_ratio_large)
    kl_large = ((ratio_large - log_ratio_large - 1) * loss_mask).sum() / loss_mask.sum()

    assert kl_large > kl_small, (
        f"Larger perturbation should produce larger KL: small={kl_small.item()}, large={kl_large.item()}"
    )


def test_kl_with_fused_lm_head():
    """FusedOutputLinear should produce same logprobs as VanillaOutputLinear."""
    config = NemotronHConfig(
        **_BASE,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        use_grouped_mm=False,
    )
    config._attn_implementation = "sdpa"

    with torch.device("cuda"), default_dtype(torch.float32):
        model = NemotronHForCausalLM._from_config(config)

    # Get logits from vanilla head
    inject_prime_lm_head(model, chunk_size=None)
    input_ids = torch.randint(0, 256, (1, 16), device="cuda")
    labels = torch.cat([input_ids[:, 1:], torch.zeros(1, 1, dtype=torch.long, device="cuda")], dim=1)

    with torch.no_grad():
        vanilla_out = model(input_ids)
    vanilla_logits = vanilla_out["logits"]
    temperature = torch.ones(1, 16, device="cuda")
    vanilla_logprobs = selective_log_softmax(vanilla_logits / temperature.unsqueeze(-1), labels)

    # Now switch to fused head and compare
    inject_prime_lm_head(model, chunk_size=16)
    with torch.no_grad():
        fused_out = model(input_ids, labels=labels, temperature=temperature)
    fused_logprobs = fused_out["logprobs"]

    diff = (vanilla_logprobs - fused_logprobs).abs().max()
    assert diff < 1e-3, f"Vanilla vs fused logprob diff: {diff.item()}"


def test_kl_logprob_alignment():
    """Verify logprobs are properly aligned between trainer and inference after shift."""
    model = _make_model()

    input_ids = torch.randint(0, 256, (1, 16), device="cuda")

    # Simulate what the training loop does
    with torch.no_grad():
        out = model(input_ids)
    logits = out["logits"]

    # Labels = shifted input_ids (predict next token)
    labels = torch.cat([input_ids[:, 1:], torch.zeros(1, 1, dtype=torch.long, device="cuda")], dim=1)

    logprobs = selective_log_softmax(logits, labels)
    logprobs_shifted = shift_tensor_right(
        logprobs, pad_value=torch.log(torch.tensor(1.0 / model.config.vocab_size)).item()
    )

    # Position 0 should be the pad value (uniform distribution logprob)
    expected_pad = torch.log(torch.tensor(1.0 / model.config.vocab_size)).item()
    assert logprobs_shifted[0, 0].item() == pytest.approx(expected_pad, abs=1e-5), (
        f"Position 0 should be pad value {expected_pad}, got {logprobs_shifted[0, 0].item()}"
    )

    # Positions 1..N-1 should contain actual logprobs (negative values)
    actual_logprobs = logprobs_shifted[0, 1:]
    assert (actual_logprobs < 0).all(), "Logprobs should be negative"
    assert (actual_logprobs > -20).all(), "Logprobs shouldn't be extremely negative"


def test_kl_backward_through_policy():
    """KL loss should produce gradients for the policy model."""
    ref_model = _make_model()
    policy_model = _make_model()

    with torch.no_grad():
        policy_model.load_state_dict(ref_model.state_dict())
    _perturb_model(policy_model, scale=0.05)

    input_ids = torch.randint(0, 256, (1, 16), device="cuda")
    labels = torch.cat([input_ids[:, 1:], torch.zeros(1, 1, dtype=torch.long, device="cuda")], dim=1)

    # Reference logprobs (detached)
    with torch.no_grad():
        ref_out = ref_model(input_ids)
    ref_logprobs = selective_log_softmax(ref_out["logits"], labels)
    ref_logprobs = shift_tensor_right(
        ref_logprobs, pad_value=torch.log(torch.tensor(1.0 / ref_model.config.vocab_size)).item()
    ).detach()

    # Policy logprobs (with grad)
    policy_out = policy_model(input_ids)
    policy_logprobs = selective_log_softmax(policy_out["logits"], labels)
    policy_logprobs = shift_tensor_right(
        policy_logprobs, pad_value=torch.log(torch.tensor(1.0 / policy_model.config.vocab_size)).item()
    )

    loss_mask = torch.ones(16, dtype=torch.bool, device="cuda")
    loss_mask[0] = False

    log_ratio = policy_logprobs[0] - ref_logprobs[0]
    kl_loss = (loss_mask * log_ratio**2).sum()
    kl_loss.backward()

    # Check gradients flow to policy model parameters
    has_grad = False
    for name, p in policy_model.named_parameters():
        if p.grad is not None and p.grad.norm().item() > 0:
            has_grad = True
            break

    assert has_grad, "KL loss should produce non-zero gradients for policy model"


def test_kl_mismatch_formula_sanity():
    """Verify the mismatch_kl formula: r - log(r) - 1 >= 0 for all r > 0."""
    model = _make_model()
    ref_model = _make_model()

    with torch.no_grad():
        ref_model.load_state_dict(model.state_dict())
    _perturb_model(model, scale=0.1)

    input_ids = torch.randint(0, 256, (2, 32), device="cuda")

    trainer_logprobs = _get_logprobs_vanilla(model, input_ids)
    inference_logprobs = _get_logprobs_vanilla(ref_model, input_ids)

    log_ratio = trainer_logprobs - inference_logprobs
    ratio = torch.exp(log_ratio)
    mismatch_kl = ratio - log_ratio - 1

    # KL divergence should always be non-negative (up to numerical precision)
    assert (mismatch_kl >= -1e-5).all(), f"KL should be non-negative, min={mismatch_kl.min().item()}"

    # For identical models (log_ratio=0, ratio=1), KL = 1 - 0 - 1 = 0
    # For divergent models, KL > 0
    loss_mask = torch.ones(32, dtype=torch.bool, device="cuda")
    loss_mask[0] = False
    mean_kl = mismatch_kl[0][loss_mask].mean()
    assert mean_kl > 0, f"Mean KL should be positive for perturbed model, got {mean_kl.item()}"
