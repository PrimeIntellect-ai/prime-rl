import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear, VanillaOutputLinear
from prime_rl.trainer.models.llama import LlamaForCausalLM as PrimeRLLlamaForCausalLM
from prime_rl.trainer.rl.loss import compute_entropy, selective_log_softmax
from prime_rl.utils.utils import default_dtype


def _baseline_logprobs_and_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, *, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden @ weight.t()
    logits = logits / float(temperature)
    logp = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    ent = compute_entropy(logits)
    return logp, ent


def test_fused_lm_head_matches_full_logits_forward_and_backward_cpu():
    torch.manual_seed(0)
    b, s, h, v = 2, 4, 8, 37
    temperature = 1.7
    chunk_size = 11

    hidden0 = torch.randn(b, s, h, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight0 = torch.randn(v, h, dtype=torch.float32, requires_grad=True)

    # Baseline
    logp0, ent0 = _baseline_logprobs_and_entropy(hidden0, weight0, labels, temperature=temperature)
    loss0 = logp0.sum()
    loss0.backward()
    grad_hidden0 = hidden0.grad.detach().clone()
    grad_weight0 = weight0.grad.detach().clone()

    # Fused
    hidden1 = hidden0.detach().clone().requires_grad_(True)
    weight1 = weight0.detach().clone().requires_grad_(True)
    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    lm.weight = torch.nn.Parameter(weight1)

    out = lm(hidden1, labels, temperature=temperature)
    assert out.logits is None
    assert out.logprobs is not None
    assert out.entropy is not None

    loss1 = out.logprobs.sum()
    loss1.backward()
    grad_hidden1 = hidden1.grad.detach().clone()
    grad_weight1 = lm.weight.grad.detach().clone()

    torch.testing.assert_close(out.logprobs, logp0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out.entropy, ent0, rtol=0, atol=1e-5)
    torch.testing.assert_close(grad_hidden1, grad_hidden0, rtol=0, atol=1e-5)
    torch.testing.assert_close(grad_weight1, grad_weight0, rtol=0, atol=1e-5)


def test_fused_lm_head_requires_labels():
    """Test that FusedOutputLinear raises assertion error when labels is None."""
    torch.manual_seed(0)
    b, s, h, v = 2, 3, 4, 9

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)

    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=5)
    lm.weight = torch.nn.Parameter(weight)

    with pytest.raises(AssertionError, match="FusedOutputLinear requires labels"):
        lm(hidden, labels=None, temperature=1.0)


def test_vanilla_lm_head_returns_logits():
    """Test that VanillaOutputLinear returns logits."""
    torch.manual_seed(0)
    b, s, h, v = 2, 3, 4, 9

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)

    lm = VanillaOutputLinear(in_features=h, out_features=v)
    lm.weight = torch.nn.Parameter(weight)

    out = lm(hidden, labels=None, temperature=1.0)
    assert out.logits is not None
    assert out.logprobs is None
    assert out.entropy is None

    logits_ref = hidden @ weight.t()
    torch.testing.assert_close(out.logits, logits_ref, rtol=0, atol=1e-6)


def test_fused_vs_vanilla_integration():
    """Integration test comparing fused and vanilla outputs after postprocessing."""
    torch.manual_seed(42)
    b, s, h, v = 2, 4, 8, 37
    temperature = 1.7
    chunk_size = 11

    hidden = torch.randn(b, s, h, dtype=torch.float16)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight = torch.randn(v, h, dtype=torch.float16)

    # Vanilla path: get logits, compute logprobs manually
    vanilla_lm = VanillaOutputLinear(in_features=h, out_features=v)
    vanilla_lm.weight = torch.nn.Parameter(weight.clone())
    vanilla_out = vanilla_lm(hidden, labels=None, temperature=temperature).cast_float_and_contiguous()

    assert vanilla_out.logits is not None
    logits = vanilla_out.logits / float(temperature)
    vanilla_logprobs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    vanilla_entropy = compute_entropy(logits)

    # Fused path: get logprobs and entropy directly
    fused_lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    fused_lm.weight = torch.nn.Parameter(weight.clone())
    fused_out = fused_lm(hidden, labels=labels, temperature=temperature).cast_float_and_contiguous()

    assert fused_out.logprobs is not None
    assert fused_out.entropy is not None

    # Compare: fused should match vanilla within tolerance
    torch.testing.assert_close(fused_out.logprobs, vanilla_logprobs, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fused_out.entropy, vanilla_entropy, rtol=1e-3, atol=1e-4)


@pytest.mark.gpu
def test_full_model_fused_vs_vanilla():
    """Full model integration test comparing fused vs vanilla LM head across multiple training steps."""
    torch.manual_seed(123)

    # Create tiny Llama model for fast testing
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
    )

    with torch.device("cuda"), default_dtype(torch.float32):
        # Create two identical models
        model_vanilla = PrimeRLLlamaForCausalLM._from_config(config)
        model_fused = PrimeRLLlamaForCausalLM._from_config(config)

        # Share weights between models
        model_fused.load_state_dict(model_vanilla.state_dict())

        # Wrap with different LM heads
        model_vanilla.wrap_lm_head(chunk_size=None)  # Vanilla
        model_fused.wrap_lm_head(chunk_size=256)  # Fused with chunking

    # Setup optimizers
    optimizer_vanilla = torch.optim.AdamW(model_vanilla.parameters(), lr=1e-4)
    optimizer_fused = torch.optim.AdamW(model_fused.parameters(), lr=1e-4)

    # Run a few training steps
    num_steps = 3
    batch_size, seq_len = 2, 64
    temperature = 1.5

    for step in range(num_steps):
        # Generate random batch
        with torch.device("cuda"):
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Vanilla forward (returns logits, compute logprobs/entropy using RL train functions)
        optimizer_vanilla.zero_grad()
        out_vanilla = model_vanilla(
            labels, position_ids, labels=labels, temperature=temperature
        ).cast_float_and_contiguous()
        if out_vanilla.logprobs is None:
            assert out_vanilla.logits is not None
            logits = out_vanilla.logits / float(temperature)
            out_vanilla.logprobs = selective_log_softmax(logits, labels)
            out_vanilla.entropy = compute_entropy(logits)
        loss_vanilla = -out_vanilla.logprobs.mean()
        loss_vanilla.backward()
        optimizer_vanilla.step()

        # Fused forward (returns logprobs and entropy directly)
        optimizer_fused.zero_grad()
        out_fused = model_fused(
            labels, position_ids, labels=labels, temperature=temperature
        ).cast_float_and_contiguous()
        if out_fused.logprobs is None:
            assert out_fused.logits is not None
            logits = out_fused.logits / float(temperature)
            out_fused.logprobs = selective_log_softmax(logits, labels)
            out_fused.entropy = compute_entropy(logits)
        loss_fused = -out_fused.logprobs.mean()
        loss_fused.backward()
        optimizer_fused.step()

        # Compare outputs (should be very close since models started identical)
        torch.testing.assert_close(out_fused.logprobs, out_vanilla.logprobs, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out_fused.entropy, out_vanilla.entropy, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(loss_fused, loss_vanilla, rtol=1e-4, atol=1e-5)

    # After training, weights should still be close (optimizer steps should be similar)
    for (name_v, param_v), (name_f, param_f) in zip(model_vanilla.named_parameters(), model_fused.named_parameters()):
        if "lm_head" not in name_v:  # Compare non-lm_head params
            torch.testing.assert_close(param_f, param_v, rtol=1e-3, atol=1e-4)
