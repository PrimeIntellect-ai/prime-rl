import math

import pytest
import torch
import torch.nn as nn

from prime_rl.trainer.models.layers.lora import MultiLoRALinear

pytestmark = [pytest.mark.gpu]


# Marker for tests that require CUDA with compute capability >= 9.0
# torch._grouped_mm is only supported on CUDA compute capability >= 9.0
def _can_run_grouped_mm() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 9


requires_grouped_mm = pytest.mark.skipif(
    not _can_run_grouped_mm(), reason="torch._grouped_mm requires CUDA with compute capability >= 9.0"
)


def test_initialization() -> None:
    """Test that MultiLoRALinear initializes correctly."""
    base = nn.Linear(10, 20)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=3, alpha=16.0)

    assert lora.rank == 4
    assert lora.n_adapters == 3
    assert lora.alpha == 16.0
    assert lora.scaling == 4.0  # alpha / r = 16 / 4
    assert lora.in_features == 10
    assert lora.out_features == 20
    assert lora.lora_A.shape == (3, 4, 10)  # [n_adapters, r, in_features]
    assert lora.lora_B.shape == (3, 20, 4)  # [n_adapters, out_features, r]


def test_initialization_with_invalid_params() -> None:
    """Test that initialization fails with invalid parameters."""
    base = nn.Linear(10, 20)

    with pytest.raises(ValueError, match="rank and n_adapters must be > 0"):
        MultiLoRALinear(base_linear=base, rank=0, n_adapters=3)

    with pytest.raises(ValueError, match="rank and n_adapters must be > 0"):
        MultiLoRALinear(base_linear=base, rank=4, n_adapters=0)

    with pytest.raises(ValueError, match="rank and n_adapters must be > 0"):
        MultiLoRALinear(base_linear=base, rank=-1, n_adapters=3)


def test_reset_parameters_all() -> None:
    """Test resetting all adapter parameters."""
    base = nn.Linear(10, 20)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=3)

    # Reset all adapters
    lora.reset_parameters()

    # lora_B should be all zeros
    assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    # lora_A should have non-zero values (randomly initialized)
    assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))


def test_reset_parameters_single_adapter() -> None:
    """Test resetting a single adapter's parameters."""
    base = nn.Linear(10, 20)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=3)

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


def test_forward_with_init_lora_equals_base() -> None:
    """Test that with init LoRA weights, output equals base layer output."""
    base = nn.Linear(10, 20).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=4)

    x = torch.randn(8, 10).cuda()
    offsets = torch.tensor([3, 3, 6, 8], dtype=torch.int32).cuda()

    lora_output = lora(x, offsets)
    base_output = base(x)

    assert lora_output.shape == (8, 20), f"lora_output.shape: {lora_output.shape}"
    assert torch.allclose(lora_output, base_output, atol=1e-6)


def test_forward_with_non_zero_lora() -> None:
    """Test that with non-zero LoRA weights, output is different from base layer output."""
    base = nn.Linear(10, 20).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=4)

    # Initialize with non-zero LoRA weights
    with torch.no_grad():
        lora.lora_B.normal_(0, 1e-1)

    x = torch.randn(6, 10).cuda()

    base_output = base(x)
    offsets_1 = torch.tensor([0, 2, 4, 6], dtype=torch.int32).cuda()
    lora_output_1 = lora(x, offsets_1)
    offsets_2 = torch.tensor([2, 2, 4, 6], dtype=torch.int32).cuda()
    lora_output_2 = lora(x, offsets_2)

    # Outputs should be different
    assert not torch.allclose(lora_output_1, base_output, atol=1e-6)
    assert not torch.allclose(lora_output_2, base_output, atol=1e-6)
    assert not torch.allclose(lora_output_1, lora_output_2, atol=1e-6)


def test_device_consistency() -> None:
    """Test that LoRA parameters are on the same device as base layer."""
    base = nn.Linear(10, 20)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=3)

    assert lora.lora_A.device == base.weight.device
    assert lora.lora_B.device == base.weight.device


def test_dtype_consistency() -> None:
    """Test that LoRA parameters have the same dtype as base layer."""
    base = nn.Linear(10, 20, dtype=torch.float32)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=3)

    assert lora.lora_A.dtype == torch.float32
    assert lora.lora_B.dtype == torch.float32


def test_properties() -> None:
    """Test that in_features and out_features properties work correctly."""
    base = nn.Linear(15, 25)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=2)

    assert lora.in_features == 15
    assert lora.out_features == 25
    assert lora.in_features == base.in_features
    assert lora.out_features == base.out_features


def test_gradient_flow_init() -> None:
    """Test that gradients flow through just LoRA parameters and base layer is frozen."""
    base = nn.Linear(10, 20).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=2)

    x = torch.randn(4, 10, requires_grad=True, device="cuda")
    offsets = torch.tensor([2, 4], dtype=torch.int32, device="cuda")

    output = lora(x, offsets)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert lora.base_linear.weight.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert x.grad is not None

    # Because B is initialized to zero, A will have zero grads
    assert torch.allclose(lora.lora_A.grad, torch.zeros_like(lora.lora_A.grad))
    assert not torch.allclose(lora.lora_B.grad, torch.zeros_like(lora.lora_B.grad))


def test_gradient_flow_non_zero_lora() -> None:
    """Test that gradients flow through both base and LoRA parameters with non-zero LoRA weights."""
    base = nn.Linear(10, 20).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=2)
    with torch.no_grad():
        nn.init.kaiming_uniform_(lora.lora_B, a=math.sqrt(5))

    x = torch.randn(4, 10, requires_grad=True, device="cuda")
    offsets = torch.tensor([2, 4], dtype=torch.int32, device="cuda")

    output = lora(x, offsets)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert lora.base_linear.weight.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert x.grad is not None

    # Check that gradients are non-zero
    assert not torch.allclose(lora.lora_A.grad, torch.zeros_like(lora.lora_A.grad))
    assert not torch.allclose(lora.lora_B.grad, torch.zeros_like(lora.lora_B.grad))


@pytest.mark.parametrize(
    "rank,n_adapters,alpha",
    [
        (2, 2, 8.0),
        (8, 5, 32.0),
        (16, 10, 64.0),
    ],
)
def test_different_configurations(rank: int, n_adapters: int, alpha: float) -> None:
    """Test MultiLoRALinear with different hyperparameter configurations."""
    base = nn.Linear(10, 20).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=rank, n_adapters=n_adapters, alpha=alpha)

    assert lora.rank == rank
    assert lora.n_adapters == n_adapters
    assert lora.alpha == alpha
    assert lora.scaling == alpha / rank
    assert lora.lora_A.shape == (n_adapters, rank, 10)
    assert lora.lora_B.shape == (n_adapters, 20, rank)

    # Test forward pass works
    x = torch.randn(n_adapters * 2, 10).cuda()
    offsets = torch.arange(2, n_adapters * 2 + 1, 2, dtype=torch.long).cuda()
    output = lora(x, offsets)
    assert output.shape == (n_adapters * 2, 20)
