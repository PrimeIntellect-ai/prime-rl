import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models import cast_float_and_contiguous
from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear, VanillaOutputLinear, inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.rl.loss import compute_entropy, selective_log_softmax, shift_tensor_left, shift_tensor_right
from prime_rl.utils.utils import default_dtype


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("head_only", [False, True])
def test_fused_head_top_logprobs_gradient(dtype, replay, head_only):
    from prime_rl.trainer.rl.loss import selective_log_softmax_with_sampling_mask, selective_topk_log_softmax

    torch.manual_seed(7)
    hidden = torch.randn(1, 7, 16, device="cuda", dtype=dtype, requires_grad=True)
    lm = FusedOutputLinear(16, 8203, chunk_size=3).to(device="cuda", dtype=dtype)
    labels = torch.tensor([[3, 8199, 12, 99, 0, 17, 8192]], device="cuda")
    head_ids = torch.stack([labels, (labels + 1) % 8203, (labels + 2) % 8203, torch.full_like(labels, -1)], dim=-1)
    head_ids[:, 2] = -1
    sampling_mask = head_ids if replay else None
    temperature = torch.linspace(0.6, 1.4, 7, device="cuda").unsqueeze(0)
    coefficient = torch.randn_like(head_ids, dtype=torch.float32)
    out = lm(hidden, labels, temperature, sampling_mask, head_ids)
    loss = (out["topk_logprobs"] * coefficient).sum()
    if not head_only:
        loss = loss + out["logprobs"].sum()
    actual = torch.autograd.grad(loss, (hidden, lm.weight))
    logits = (hidden @ lm.weight.t()).float() / temperature.unsqueeze(-1)
    head = selective_topk_log_softmax(logits, head_ids, sampling_mask, labels)
    reference = (head * coefficient).sum()
    if not head_only:
        sampled = (
            selective_log_softmax_with_sampling_mask(logits, labels, sampling_mask)
            if replay
            else selective_log_softmax(logits, labels)
        )
        reference = reference + sampled.sum()
    expected = torch.autograd.grad(reference, (hidden, lm.weight))
    torch.testing.assert_close(out["topk_logprobs"], head, atol=3e-6, rtol=1e-6)
    for grad, ref in zip(actual, expected, strict=True):
        atol, rtol = (1e-2, 3e-2) if dtype == torch.bfloat16 else (5e-6, 1e-4)
        torch.testing.assert_close(grad, ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
def test_ipo_score_centering_replay_matches_dense_support_gradient():
    from prime_rl.configs.trainer import IPOLossConfig
    from prime_rl.trainer.rl.loss import LossInputs, setup_rl_loss_fn

    hidden = torch.eye(2, device="cuda").unsqueeze(0).requires_grad_()
    lm = FusedOutputLinear(2, 7, chunk_size=1).cuda()
    # Large excluded logits expose accidental full-vocabulary normalization.
    lm.weight = torch.nn.Parameter(
        torch.tensor(
            [[9.0, 0.4], [0.4, 9.0], [8.0, -0.7], [-0.7, 8.0], [7.0, 7.0], [0.1, 6.0], [6.0, 0.1]], device="cuda"
        )
    )
    support = torch.tensor([[[1, 3, 5, -1], [0, 2, 6, -1]]], device="cuda")
    labels = torch.tensor([[1, 2]], device="cuda")
    temperature = torch.tensor([[0.7, 1.3]], device="cuda")
    q = torch.tensor([[[0.7, 0.2, 0.1, 0.0], [0.6, 0.3, 0.1, 0.0]]], device="cuda")
    q_sampled = torch.tensor([[0.7, 0.3]], device="cuda")
    advantage = torch.tensor([[1.3, -0.7]], device="cuda")
    weights = torch.tensor([[0.4, 2.0]], device="cuda")
    config = IPOLossConfig(eps=0.1, max_importance_ratio=1.2, score_centering=True)
    out = lm(hidden, labels, temperature, support, support)
    actual = setup_rl_loss_fn(config).loss(
        LossInputs(
            out["logprobs"],
            q_sampled.log(),
            None,
            advantage,
            torch.ones_like(labels, dtype=torch.bool),
            weights,
            out["topk_logprobs"],
            q.log().masked_fill(support < 0, 0.0),
            support >= 0,
        )
    )

    logits = (hidden @ lm.weight.t()) / temperature.unsqueeze(-1)
    dense_logp = torch.stack([logits[0, i, support[0, i, :3]].log_softmax(-1) for i in range(2)])
    p = dense_logp.exp().detach()
    q_dense = q[0, :, :3]
    w = (p / q_dense).clamp_max(config.max_importance_ratio) * ((p - q_dense).abs() <= config.eps)
    sampled_logp = dense_logp[[0, 1], [0, 1]]
    sampled_w = w[[0, 1], [0, 1]]
    assert sampled_w[0] == 0
    center = (q_dense * w * dense_logp).sum(-1)
    reference = (-advantage[0] * weights[0] * (sampled_w * sampled_logp - center)).sum()
    grads = torch.autograd.grad(actual.loss, (hidden, lm.weight), retain_graph=True)
    expected = torch.autograd.grad(reference, (hidden, lm.weight))
    for grad, ref in zip(grads, expected, strict=True):
        torch.testing.assert_close(grad, ref, atol=2e-7, rtol=2e-6)
    # No path through logits outside each frozen support, including the centering term.
    assert torch.count_nonzero(grads[1][[0, 2, 4, 6], 0]) == 0
    assert torch.count_nonzero(grads[1][[1, 3, 4, 5], 1]) == 0


def _baseline_logprobs_and_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, *, temperature: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline logprobs and entropy with per-token temperature tensor."""
    logits = hidden @ weight.t()
    # temperature is [b, s], logits is [b, s, v]
    logits = logits / temperature.unsqueeze(-1)
    logp = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    ent = compute_entropy(logits)
    return logp, ent


def test_fused_lm_head_matches_full_logits_forward_and_backward_cpu():
    torch.manual_seed(0)
    b, s, h, v = 2, 4, 8, 37
    temperature = torch.full((b, s), 1.7, dtype=torch.float32)
    chunk_size = 3

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


def test_fused_lm_head_frozen_weight_backward_cpu():
    """Frozen LM-head weight: hidden gradient matches baseline, no weight gradient is produced."""
    torch.manual_seed(0)
    b, s, h, v = 2, 4, 8, 37
    temperature = torch.full((b, s), 1.7, dtype=torch.float32)
    chunk_size = 3

    hidden0 = torch.randn(b, s, h, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight0 = torch.randn(v, h, dtype=torch.float32)

    # Baseline (weight frozen)
    logp0, _ = _baseline_logprobs_and_entropy(hidden0, weight0, labels, temperature=temperature)
    logp0.sum().backward()
    grad_hidden0 = hidden0.grad.detach().clone()

    # Fused (weight frozen)
    hidden1 = hidden0.detach().clone().requires_grad_(True)
    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    lm.weight = torch.nn.Parameter(weight0.clone(), requires_grad=False)

    out = lm(hidden1, labels, temperature=temperature)
    out["logprobs"].sum().backward()

    torch.testing.assert_close(hidden1.grad, grad_hidden0, rtol=0, atol=1e-5)
    assert lm.weight.grad is None


def test_fused_lm_head_requires_labels():
    """Test that FusedOutputLinear raises assertion error when labels is None."""
    torch.manual_seed(0)
    b, s, h, v = 2, 3, 4, 9

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)
    temperature = torch.full((b, s), 1.0, dtype=torch.float32)

    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=5)
    lm.weight = torch.nn.Parameter(weight)

    with pytest.raises(AssertionError, match="FusedOutputLinear requires labels"):
        lm(hidden, labels=None, temperature=temperature)


def test_vanilla_lm_head_returns_logits():
    """Test that VanillaOutputLinear returns logits."""
    torch.manual_seed(0)
    b, s, h, v = 2, 3, 4, 9

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)

    lm = VanillaOutputLinear(in_features=h, out_features=v)
    lm.weight = torch.nn.Parameter(weight)

    # VanillaOutputLinear doesn't use temperature - it just returns logits
    out = lm(hidden, labels=None, temperature=None)
    assert out.get("logits") is not None
    assert out.get("logprobs") is None
    assert out.get("entropy") is None

    logits_ref = hidden @ weight.t()
    torch.testing.assert_close(out["logits"], logits_ref, rtol=0, atol=1e-6)


def test_fused_vs_vanilla_integration():
    """Integration test comparing fused and vanilla outputs after postprocessing."""
    torch.manual_seed(42)
    b, s, h, v = 2, 4, 8, 37
    temp_value = 1.7
    temperature = torch.full((b, s), temp_value, dtype=torch.float32)
    chunk_size = 3

    hidden = torch.randn(b, s, h, dtype=torch.float16)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight = torch.randn(v, h, dtype=torch.float16)

    # Vanilla path: get logits, compute logprobs manually
    vanilla_lm = VanillaOutputLinear(in_features=h, out_features=v)
    vanilla_lm.weight = torch.nn.Parameter(weight.clone())
    vanilla_out = cast_float_and_contiguous(vanilla_lm(hidden, labels=None, temperature=None))

    assert vanilla_out.get("logits") is not None
    logits = vanilla_out["logits"] / temp_value
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


def test_fused_lm_head_correct_shift():
    """
    End-to-end test that the fused LM head with shifted labels, after shift_tensor_right,
    produces logprobs aligned with the inference convention.

    This simulates the full training loop behavior and verifies the importance ratio
    (trainer_logprobs - inference_logprobs) is ~0 for positions that matter in training.
    """
    torch.manual_seed(999)
    b, s, h, v = 2, 16, 32, 50
    temp_value = 1.5
    temperature = torch.full((b, s), temp_value, dtype=torch.float32)
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
    logits = logits / temp_value
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
    """Test that inject_prime_lm_head correctly wraps the model with VanillaOutputLinear."""
    torch.manual_seed(123)

    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
    )

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3ForCausalLM._from_config(config)

    # Wrap with VanillaOutputLinear (chunk_size=None)
    inject_prime_lm_head(model, chunk_size=None)

    assert isinstance(model.lm_head, VanillaOutputLinear), "lm_head should be VanillaOutputLinear"

    # Test forward with labels and temperature
    batch_size, seq_len = 2, 64

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        temperature = torch.full((batch_size, seq_len), 1.5, dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            temperature=temperature,
            seq_lens=torch.tensor([seq_len], device="cuda"),
        )

    # VanillaOutputLinear returns logits
    assert isinstance(out, dict), "Output should be PrimeLmOutput (dict)"
    assert out.get("logits") is not None, "Vanilla path should return logits"
    assert out.get("logprobs") is None, "Vanilla path should not return logprobs"
    assert out["logits"].shape == (batch_size, seq_len, config.vocab_size), "Logits shape mismatch"


@pytest.mark.gpu
def test_inject_prime_lm_head_fused():
    """Test that inject_prime_lm_head correctly wraps the model with FusedOutputLinear."""
    torch.manual_seed(123)

    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
    )

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3ForCausalLM._from_config(config)

    # Wrap with FusedOutputLinear
    inject_prime_lm_head(model, chunk_size=512)

    assert isinstance(model.lm_head, FusedOutputLinear), "lm_head should be FusedOutputLinear"

    # Test forward with labels and temperature
    batch_size, seq_len = 2, 64

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        temperature = torch.full((batch_size, seq_len), 1.5, dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            temperature=temperature,
            seq_lens=torch.tensor([seq_len], device="cuda"),
        )

    # FusedOutputLinear returns logprobs and entropy
    assert isinstance(out, dict), "Output should be PrimeLmOutput (dict)"
    assert out.get("logprobs") is not None, "Fused path should return logprobs"
    assert out.get("entropy") is not None, "Fused path should return entropy"
    assert out.get("logits") is None, "Fused path should not return logits"
    assert out["logprobs"].shape == (batch_size, seq_len), "Logprobs shape mismatch"
    assert out["entropy"].shape == (batch_size, seq_len), "Entropy shape mismatch"
