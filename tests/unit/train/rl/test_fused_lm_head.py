import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.trainer.models import cast_float_and_contiguous
from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear, VanillaOutputLinear, inject_prime_lm_head
from prime_rl.trainer.models.llama import LlamaForCausalLM as PrimeRLLlamaForCausalLM
from prime_rl.trainer.rl.loss import compute_entropy, selective_log_softmax, shift_tensor_left, shift_tensor_right
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
    assert out.get("logits") is None
    assert out.get("logprobs") is not None
    assert out.get("entropy") is not None

    loss1 = out["logprobs"].sum()
    loss1.backward()
    grad_hidden1 = hidden1.grad.detach().clone()
    grad_weight1 = lm.weight.grad.detach().clone()

    torch.testing.assert_close(out["logprobs"], logp0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out["entropy"], ent0, rtol=0, atol=1e-5)
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
    assert out.get("logits") is not None
    assert out.get("logprobs") is None
    assert out.get("entropy") is None

    logits_ref = hidden @ weight.t()
    torch.testing.assert_close(out["logits"], logits_ref, rtol=0, atol=1e-6)


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
    vanilla_out = cast_float_and_contiguous(vanilla_lm(hidden, labels=None, temperature=temperature))

    assert vanilla_out.get("logits") is not None
    logits = vanilla_out["logits"] / float(temperature)
    vanilla_logprobs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    vanilla_entropy = compute_entropy(logits)

    # Fused path: get logprobs and entropy directly
    fused_lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    fused_lm.weight = torch.nn.Parameter(weight.clone())
    fused_out = cast_float_and_contiguous(fused_lm(hidden, labels=labels, temperature=temperature))

    assert fused_out.get("logprobs") is not None
    assert fused_out.get("entropy") is not None

    # Compare: fused should match vanilla within tolerance
    torch.testing.assert_close(fused_out["logprobs"], vanilla_logprobs, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fused_out["entropy"], vanilla_entropy, rtol=1e-3, atol=1e-4)


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
        inject_prime_lm_head(model_vanilla, chunk_size=None)  # Vanilla
        inject_prime_lm_head(model_fused, chunk_size=256)  # Fused with chunking

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
        out_vanilla = cast_float_and_contiguous(
            model_vanilla(labels, position_ids, labels=labels, temperature=temperature)
        )
        if out_vanilla.get("logprobs") is None:
            assert out_vanilla.get("logits") is not None
            logits = out_vanilla["logits"] / float(temperature)
            out_vanilla["logprobs"] = selective_log_softmax(logits, labels)
            out_vanilla["entropy"] = compute_entropy(logits)
        loss_vanilla = -out_vanilla["logprobs"].mean()
        loss_vanilla.backward()
        optimizer_vanilla.step()

        # Fused forward (returns logprobs and entropy directly)
        optimizer_fused.zero_grad()
        out_fused = cast_float_and_contiguous(model_fused(labels, position_ids, labels=labels, temperature=temperature))
        if out_fused.get("logprobs") is None:
            assert out_fused.get("logits") is not None
            logits = out_fused["logits"] / float(temperature)
            out_fused["logprobs"] = selective_log_softmax(logits, labels)
            out_fused["entropy"] = compute_entropy(logits)
        loss_fused = -out_fused["logprobs"].mean()
        loss_fused.backward()
        optimizer_fused.step()

        # Compare outputs (should be very close since models started identical)
        torch.testing.assert_close(out_fused["logprobs"], out_vanilla["logprobs"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out_fused["entropy"], out_vanilla["entropy"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(loss_fused, loss_vanilla, rtol=1e-4, atol=1e-5)

    # After training, weights should still be close (optimizer steps should be similar)
    for (name_v, param_v), (name_f, param_f) in zip(model_vanilla.named_parameters(), model_fused.named_parameters()):
        if "lm_head" not in name_v:  # Compare non-lm_head params
            torch.testing.assert_close(param_f, param_v, rtol=1e-3, atol=1e-4)


def test_fused_lm_head_correct_shift():
    """
    End-to-end test that the fused LM head with shifted labels, after shift_tensor_right,
    produces logprobs aligned with the inference convention.

    This simulates the full training loop behavior and verifies the importance ratio
    (trainer_logprobs - inference_logprobs) is ~0 for positions that matter in training.
    """
    torch.manual_seed(999)
    b, s, h, v = 2, 16, 32, 50
    temperature = 1.5
    chunk_size = 13

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)
    input_ids = torch.randint(0, v, (b, s), dtype=torch.long)

    # Create shifted labels as done in training
    labels = shift_tensor_left(input_ids)

    # === Fused path (as in training) ===
    fused_lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    fused_lm.weight = torch.nn.Parameter(weight.clone())
    fused_out = fused_lm(hidden, labels=labels, temperature=temperature)
    trainer_logprobs = shift_tensor_right(fused_out["logprobs"])

    # === Inference convention (baseline) ===
    logits = hidden @ weight.t()
    logits = logits / temperature
    # Shift logits right (prepend zeros, drop last) to get inference convention
    shifted_logits = torch.cat([torch.zeros(b, 1, v, dtype=logits.dtype), logits[:, :-1, :]], dim=1)
    inference_logprobs = (
        torch.log_softmax(shifted_logits, dim=-1).gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    )

    assert torch.all(trainer_logprobs[:, 0] == 0), "Position 0 should be 0 after shift_tensor_right"

    importance_ratio = trainer_logprobs[:, 1:] - inference_logprobs[:, 1:]
    torch.testing.assert_close(
        importance_ratio,
        torch.zeros(b, s - 1),
        rtol=0,
        atol=1e-4,
        msg="Importance ratio at positions 1 to s-1 should be ~0 (same token probs)",
    )


@pytest.mark.gpu
def test_inject_prime_lm_head_vanilla():
    """Test that inject_prime_lm_head correctly wraps HF model with VanillaOutputLinear."""
    torch.manual_seed(123)

    # Create tiny Llama model using HuggingFace AutoModelForCausalLM
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
        model = AutoModelForCausalLM.from_config(config)

    # Wrap with VanillaOutputLinear (chunk_size=None)
    inject_prime_lm_head(model, chunk_size=None)

    assert isinstance(model.lm_head, VanillaOutputLinear), "lm_head should be VanillaOutputLinear"

    # Test forward with labels and temperature
    batch_size, seq_len = 2, 64
    temperature = 1.5

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, position_ids=position_ids, labels=labels, temperature=temperature)

    # VanillaOutputLinear returns logits
    assert isinstance(out, dict), "Output should be PrimeLmOutput (dict)"
    assert out.get("logits") is not None, "Vanilla path should return logits"
    assert out.get("logprobs") is None, "Vanilla path should not return logprobs"
    assert out["logits"].shape == (batch_size, seq_len, config.vocab_size), "Logits shape mismatch"


@pytest.mark.gpu
def test_inject_prime_lm_head_fused():
    """Test that inject_prime_lm_head correctly wraps HF model with FusedOutputLinear."""
    torch.manual_seed(123)

    # Create tiny Llama model using HuggingFace AutoModelForCausalLM
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
        model = AutoModelForCausalLM.from_config(config)

    # Wrap with FusedOutputLinear (chunk_size=512)
    inject_prime_lm_head(model, chunk_size=512)

    assert isinstance(model.lm_head, FusedOutputLinear), "lm_head should be FusedOutputLinear"

    # Test forward with labels and temperature
    batch_size, seq_len = 2, 64
    temperature = 1.5

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, position_ids=position_ids, labels=labels, temperature=temperature)

    # FusedOutputLinear returns logprobs and entropy
    assert isinstance(out, dict), "Output should be PrimeLmOutput (dict)"
    assert out.get("logprobs") is not None, "Fused path should return logprobs"
    assert out.get("entropy") is not None, "Fused path should return entropy"
    assert out.get("logits") is None, "Fused path should not return logits"
    assert out["logprobs"].shape == (batch_size, seq_len), "Logprobs shape mismatch"
    assert out["entropy"].shape == (batch_size, seq_len), "Entropy shape mismatch"


@pytest.mark.gpu
def test_hf_model_fused_vs_vanilla_matches():
    """Test that fused and vanilla paths produce equivalent results for HF models."""
    torch.manual_seed(42)

    # Create tiny Llama model using HuggingFace AutoModelForCausalLM
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
        model_vanilla = AutoModelForCausalLM.from_config(config)
        model_fused = AutoModelForCausalLM.from_config(config)
        # Share weights
        model_fused.load_state_dict(model_vanilla.state_dict())

    # Wrap models
    inject_prime_lm_head(model_vanilla, chunk_size=None)
    inject_prime_lm_head(model_fused, chunk_size=512)

    # Test data
    batch_size, seq_len = 2, 64
    temperature = 1.5

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Run vanilla model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_vanilla = model_vanilla(
            input_ids=input_ids, position_ids=position_ids, labels=labels, temperature=temperature
        )

    # Compute logprobs and entropy from vanilla logits
    logits = out_vanilla["logits"].float() / float(temperature)
    vanilla_logprobs = selective_log_softmax(logits, labels)
    vanilla_entropy = compute_entropy(logits)

    # Run fused model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_fused = model_fused(input_ids=input_ids, position_ids=position_ids, labels=labels, temperature=temperature)

    fused_logprobs = out_fused["logprobs"].float()
    fused_entropy = out_fused["entropy"].float()

    # Compare results
    torch.testing.assert_close(fused_logprobs, vanilla_logprobs, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fused_entropy, vanilla_entropy, rtol=1e-3, atol=1e-4)
