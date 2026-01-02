import pytest
import torch

from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts


@pytest.mark.gpu
def test_multi_lora_grouped_experts_shapes():
    """Test parameter shapes match base layer dimensions."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 3

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters)

    # Check w1 LoRA shapes (gate_proj: dim -> hidden_dim)
    assert lora.w1_lora_A[0].shape == (num_experts, rank, dim)
    assert lora.w1_lora_B[0].shape == (num_experts, hidden_dim, rank)

    # Check w2 LoRA shapes (down_proj: hidden_dim -> dim)
    assert lora.w2_lora_A[0].shape == (num_experts, rank, hidden_dim)
    assert lora.w2_lora_B[0].shape == (num_experts, dim, rank)

    # Check w3 LoRA shapes (up_proj: dim -> hidden_dim)
    assert lora.w3_lora_A[0].shape == (num_experts, rank, dim)
    assert lora.w3_lora_B[0].shape == (num_experts, hidden_dim, rank)

    # Check all adapters are created
    assert len(lora.w1_lora_A) == n_adapters
    assert len(lora.w1_lora_B) == n_adapters
    assert len(lora.w2_lora_A) == n_adapters
    assert len(lora.w2_lora_B) == n_adapters
    assert len(lora.w3_lora_A) == n_adapters
    assert len(lora.w3_lora_B) == n_adapters


@pytest.mark.gpu
def test_named_parameters_for_adapter():
    """Test FQN generation for vLLM compatibility."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 3

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters)

    # Get parameters for adapter 0
    params = lora.named_parameters_for_adapter(0)

    # Should return 48 parameters (8 experts × 3 projections × 2 matrices)
    assert len(params) == num_experts * 3 * 2

    # Check first expert's gate_proj FQNs
    assert params[0][0] == "experts.0.gate_proj.lora_A"
    assert params[0][1].shape == (rank, dim)

    assert params[1][0] == "experts.0.gate_proj.lora_B"
    assert params[1][1].shape == (hidden_dim, rank)

    # Check first expert's down_proj FQNs
    assert params[2][0] == "experts.0.down_proj.lora_A"
    assert params[2][1].shape == (rank, hidden_dim)

    assert params[3][0] == "experts.0.down_proj.lora_B"
    assert params[3][1].shape == (dim, rank)

    # Check first expert's up_proj FQNs
    assert params[4][0] == "experts.0.up_proj.lora_A"
    assert params[4][1].shape == (rank, dim)

    assert params[5][0] == "experts.0.up_proj.lora_B"
    assert params[5][1].shape == (hidden_dim, rank)

    # Check last expert's up_proj FQNs
    assert params[-2][0] == f"experts.{num_experts - 1}.up_proj.lora_A"
    assert params[-1][0] == f"experts.{num_experts - 1}.up_proj.lora_B"

    # Verify all parameters are 2D tensors (per-expert slices)
    for name, param in params:
        assert param.dim() == 2, f"Parameter {name} should be 2D, got {param.dim()}D"


@pytest.mark.gpu
def test_forward_phase1():
    """Test Phase 1 forward pass (single adapter mode)."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 3

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters, use_phase2=False).cuda()

    # Simulate routed input: 100 tokens distributed across experts
    x = torch.randn(100, dim, device="cuda")
    num_tokens_per_expert = torch.tensor([10, 15, 12, 8, 20, 13, 11, 11], device="cuda", dtype=torch.int32)

    # Forward pass
    out = lora(x, num_tokens_per_expert)

    # Check output shape
    assert out.shape == (100, dim)

    # Check output is not NaN or Inf
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.gpu
def test_backward_pass():
    """Test backward pass and gradient computation."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 3

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
    # Initialize base weights to avoid NaN gradients
    base.init_weights(init_std=0.02)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters, use_phase2=False).cuda()

    # Simulate routed input
    x = torch.randn(100, dim, device="cuda", requires_grad=True)
    num_tokens_per_expert = torch.tensor([10, 15, 12, 8, 20, 13, 11, 11], device="cuda", dtype=torch.int32)

    # Forward pass
    out = lora(x, num_tokens_per_expert)

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check gradients exist for LoRA parameters
    assert lora.w1_lora_A[0].grad is not None
    assert lora.w1_lora_B[0].grad is not None
    assert lora.w2_lora_A[0].grad is not None
    assert lora.w2_lora_B[0].grad is not None
    assert lora.w3_lora_A[0].grad is not None
    assert lora.w3_lora_B[0].grad is not None

    # Check gradients are not NaN
    assert not torch.isnan(lora.w1_lora_A[0].grad).any()
    assert not torch.isnan(lora.w1_lora_B[0].grad).any()

    # Check base layer parameters have no gradients (frozen)
    assert base.w1.grad is None
    assert base.w2.grad is None
    assert base.w3.grad is None


@pytest.mark.gpu
def test_reset_parameters():
    """Test parameter initialization."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 3

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters)

    # Check B matrices are zero-initialized
    assert torch.allclose(lora.w1_lora_B[0], torch.zeros_like(lora.w1_lora_B[0]))
    assert torch.allclose(lora.w2_lora_B[0], torch.zeros_like(lora.w2_lora_B[0]))
    assert torch.allclose(lora.w3_lora_B[0], torch.zeros_like(lora.w3_lora_B[0]))

    # Check A matrices are Kaiming-initialized (non-zero)
    assert not torch.allclose(lora.w1_lora_A[0], torch.zeros_like(lora.w1_lora_A[0]))
    assert not torch.allclose(lora.w2_lora_A[0], torch.zeros_like(lora.w2_lora_A[0]))
    assert not torch.allclose(lora.w3_lora_A[0], torch.zeros_like(lora.w3_lora_A[0]))

    # Test resetting a specific adapter
    lora.w1_lora_B[1] = torch.nn.Parameter(torch.ones_like(lora.w1_lora_B[1]))
    lora.reset_parameters(index=1)
    assert torch.allclose(lora.w1_lora_B[1], torch.zeros_like(lora.w1_lora_B[1]))


@pytest.mark.gpu
def test_phase2_not_implemented():
    """Test Phase 2 raises NotImplementedError."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 3

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters, use_phase2=True).cuda()

    x = torch.randn(100, dim, device="cuda")
    num_tokens_per_expert = torch.tensor([10, 15, 12, 8, 20, 13, 11, 11], device="cuda", dtype=torch.int32)

    # Should raise NotImplementedError for Phase 2
    with pytest.raises(NotImplementedError):
        lora(x, num_tokens_per_expert)


@pytest.mark.gpu
def test_different_num_experts():
    """Test with different number of experts."""
    for num_experts in [4, 8, 16]:
        dim = 128
        hidden_dim = 512
        rank = 16
        n_adapters = 2

        base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
        lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters).cuda()

        # Check parameter count
        params = lora.named_parameters_for_adapter(0)
        assert len(params) == num_experts * 3 * 2

        # Test forward pass
        total_tokens = num_experts * 10
        x = torch.randn(total_tokens, dim, device="cuda")
        num_tokens_per_expert = torch.full((num_experts,), 10, device="cuda", dtype=torch.int32)

        out = lora(x, num_tokens_per_expert)
        assert out.shape == (total_tokens, dim)


@pytest.mark.gpu
def test_zero_initialization_effect():
    """Test that zero-initialized lora_B means LoRA has no effect initially."""
    dim = 128
    hidden_dim = 512
    num_experts = 8
    rank = 16
    n_adapters = 1

    base = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False).cuda()
    # Initialize base weights
    base.init_weights(init_std=0.02)
    lora = MultiLoRAGroupedExperts(base, rank=rank, n_adapters=n_adapters).cuda()

    x = torch.randn(100, dim, device="cuda")
    num_tokens_per_expert = torch.tensor([10, 15, 12, 8, 20, 13, 11, 11], device="cuda", dtype=torch.int32)

    # Forward through base only
    base_out = base(x, num_tokens_per_expert)

    # Forward through LoRA
    lora_out = lora(x, num_tokens_per_expert)

    # Since lora_B is zero-initialized, output should be identical to base
    assert torch.allclose(lora_out, base_out, rtol=1e-3, atol=1e-5)
