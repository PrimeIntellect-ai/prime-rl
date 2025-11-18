import pytest
import torch
import torch.nn as nn

from prime_rl.trainer.models.layers.lora import MultiLoRALinear


# Marker for tests that require CUDA with compute capability >= 9.0
# torch._grouped_mm is only supported on modern GPUs (H100, etc.)
def _can_run_grouped_mm() -> bool:
    if not torch.cuda.is_available():
        return False
    # Check compute capability
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return capability[0] >= 9


requires_cuda = pytest.mark.skipif(
    not _can_run_grouped_mm(), reason="torch._grouped_mm requires CUDA with compute capability >= 9.0"
)


class TestMultiLoRALinear:
    """Test suite for MultiLoRALinear layer."""

    def test_initialization(self) -> None:
        """Test that MultiLoRALinear initializes correctly."""
        base = nn.Linear(10, 20)
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3, alpha=16.0)

        assert lora.r == 4
        assert lora.n_adapters == 3
        assert lora.alpha == 16.0
        assert lora.scaling == 4.0  # alpha / r = 16 / 4
        assert lora.in_features == 10
        assert lora.out_features == 20
        assert lora.lora_A.shape == (3, 10, 4)  # [n_adapters, in_features, r]
        assert lora.lora_B.shape == (3, 4, 20)  # [n_adapters, r, out_features]

    def test_initialization_with_invalid_params(self) -> None:
        """Test that initialization fails with invalid parameters."""
        base = nn.Linear(10, 20)

        with pytest.raises(ValueError, match="r and n_adapters must be > 0"):
            MultiLoRALinear(base_linear=base, r=0, n_adapters=3)

        with pytest.raises(ValueError, match="r and n_adapters must be > 0"):
            MultiLoRALinear(base_linear=base, r=4, n_adapters=0)

        with pytest.raises(ValueError, match="r and n_adapters must be > 0"):
            MultiLoRALinear(base_linear=base, r=-1, n_adapters=3)

    def test_reset_parameters_all(self) -> None:
        """Test resetting all adapter parameters."""
        base = nn.Linear(10, 20)
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        # Reset all adapters
        lora.reset_parameters()

        # lora_B should be all zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

        # lora_A should have non-zero values (randomly initialized)
        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))

    def test_reset_parameters_single_adapter(self) -> None:
        """Test resetting a single adapter's parameters."""
        base = nn.Linear(10, 20)
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        # Set all to known values first
        with torch.no_grad():
            lora.lora_A.fill_(1.0)
            lora.lora_B.fill_(1.0)

        # Reset only adapter 1
        lora.reset_parameters(index=1)

        # Adapter 0 and 2 should still be 1.0
        assert torch.allclose(lora.lora_A[0], torch.ones_like(lora.lora_A[0]))
        assert torch.allclose(lora.lora_A[2], torch.ones_like(lora.lora_A[2]))
        assert torch.allclose(lora.lora_B[0], torch.ones_like(lora.lora_B[0]))
        assert torch.allclose(lora.lora_B[2], torch.ones_like(lora.lora_B[2]))

        # Adapter 1 should be reset (lora_B[1] = 0, lora_A[1] != 1)
        assert torch.allclose(lora.lora_B[1], torch.zeros_like(lora.lora_B[1]))
        assert not torch.allclose(lora.lora_A[1], torch.ones_like(lora.lora_A[1]))

    def test_reset_parameters_init_base(self) -> None:
        """Test that reset_parameters can also reset base layer."""
        base = nn.Linear(10, 20)
        original_weight = base.weight.clone()
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        # Reset with init_base=True
        lora.reset_parameters(init_base=True)

        # Base weights should have changed
        assert not torch.allclose(lora.base.weight, original_weight)

    @requires_cuda
    def test_forward_shape_2d(self) -> None:
        """Test forward pass with 2D input [B, in_features]."""
        base = nn.Linear(10, 20).cuda()
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        batch_size = 8
        x = torch.randn(batch_size, 10).cuda()

        # Create offsets for grouped matmul
        # Example: first 3 samples use adapter 0, next 3 use adapter 1, last 2 use adapter 2
        offsets = torch.tensor([0, 3, 6, 8], dtype=torch.long).cuda()

        output = lora(x, offsets)

        assert output.shape == (batch_size, 20)

    @requires_cuda
    def test_forward_shape_3d(self) -> None:
        """Test forward pass with 3D input [B, T, in_features]."""
        base = nn.Linear(10, 20).cuda()
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        batch_size = 4
        seq_len = 5
        x = torch.randn(batch_size, seq_len, 10).cuda()

        # Create offsets for grouped matmul
        # Each batch element uses a different adapter
        offsets = torch.tensor([0, 5, 10, 15, 20], dtype=torch.long).cuda()

        output = lora(x, offsets)

        assert output.shape == (batch_size, seq_len, 20)

    @requires_cuda
    def test_forward_with_zero_lora_equals_base(self) -> None:
        """Test that with zero LoRA weights, output equals base layer output."""
        base = nn.Linear(10, 20).cuda()
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        # Set LoRA weights to zero
        with torch.no_grad():
            lora.lora_A.zero_()
            lora.lora_B.zero_()

        x = torch.randn(8, 10).cuda()
        offsets = torch.tensor([0, 3, 6, 8], dtype=torch.long).cuda()

        lora_output = lora(x, offsets)
        base_output = base(x)

        assert torch.allclose(lora_output, base_output, atol=1e-6)

    @requires_cuda
    def test_forward_different_adapters_produce_different_outputs(self) -> None:
        """Test that different adapters produce different outputs."""
        base = nn.Linear(10, 20).cuda()
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        # Initialize with non-zero LoRA weights
        with torch.no_grad():
            lora.lora_A[0].normal_(0, 0.1)
            lora.lora_B[0].normal_(0, 0.1)
            lora.lora_A[1].normal_(0, 0.1)
            lora.lora_B[1].normal_(0, 0.1)
            lora.lora_A[2].normal_(0, 0.1)
            lora.lora_B[2].normal_(0, 0.1)

        x = torch.randn(6, 10).cuda()

        # Use adapter 0 for first 2 samples, adapter 1 for next 2, adapter 2 for last 2
        offsets_0_1_2 = torch.tensor([0, 2, 4, 6], dtype=torch.long).cuda()
        output1 = lora(x, offsets_0_1_2)

        # Use adapter 0 for all samples
        offsets_all_0 = torch.tensor([0, 6], dtype=torch.long).cuda()
        output2 = lora(x, offsets_all_0)

        # Outputs should be different
        assert not torch.allclose(output1, output2, atol=1e-4)

    def test_device_consistency(self) -> None:
        """Test that LoRA parameters are on the same device as base layer."""
        base = nn.Linear(10, 20)
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        assert lora.lora_A.device == base.weight.device
        assert lora.lora_B.device == base.weight.device

    def test_dtype_consistency(self) -> None:
        """Test that LoRA parameters have the same dtype as base layer."""
        base = nn.Linear(10, 20, dtype=torch.float32)
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=3)

        assert lora.lora_A.dtype == torch.float32
        assert lora.lora_B.dtype == torch.float32

    def test_properties(self) -> None:
        """Test that in_features and out_features properties work correctly."""
        base = nn.Linear(15, 25)
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=2)

        assert lora.in_features == 15
        assert lora.out_features == 25
        assert lora.in_features == base.in_features
        assert lora.out_features == base.out_features

    @requires_cuda
    def test_gradient_flow(self) -> None:
        """Test that gradients flow through both base and LoRA parameters."""
        base = nn.Linear(10, 20).cuda()
        lora = MultiLoRALinear(base_linear=base, r=4, n_adapters=2)

        x = torch.randn(4, 10, requires_grad=True).cuda()
        offsets = torch.tensor([0, 2, 4], dtype=torch.long).cuda()

        output = lora(x, offsets)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert lora.base.weight.grad is not None
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert x.grad is not None

        # Check that gradients are non-zero
        assert not torch.allclose(lora.base.weight.grad, torch.zeros_like(lora.base.weight.grad))
        assert not torch.allclose(lora.lora_A.grad, torch.zeros_like(lora.lora_A.grad))

    @requires_cuda
    @pytest.mark.parametrize(
        "r,n_adapters,alpha",
        [
            (2, 2, 8.0),
            (8, 5, 32.0),
            (16, 10, 64.0),
        ],
    )
    def test_different_configurations(self, r: int, n_adapters: int, alpha: float) -> None:
        """Test MultiLoRALinear with different hyperparameter configurations."""
        base = nn.Linear(10, 20).cuda()
        lora = MultiLoRALinear(base_linear=base, r=r, n_adapters=n_adapters, alpha=alpha)

        assert lora.r == r
        assert lora.n_adapters == n_adapters
        assert lora.alpha == alpha
        assert lora.scaling == alpha / r
        assert lora.lora_A.shape == (n_adapters, 10, r)
        assert lora.lora_B.shape == (n_adapters, r, 20)

        # Test forward pass works
        x = torch.randn(n_adapters * 2, 10).cuda()
        offsets = torch.arange(0, n_adapters * 2 + 1, 2, dtype=torch.long).cuda()
        output = lora(x, offsets)
        assert output.shape == (n_adapters * 2, 20)
