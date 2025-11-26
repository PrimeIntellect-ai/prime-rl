import math

import pytest
import torch
import torch.nn as nn

from prime_rl.trainer.models.layers.lora import MultiLoRALinear, set_offsets

pytestmark = [pytest.mark.gpu]

torch.set_default_device(torch.device("cuda"))
torch.set_default_dtype(torch.bfloat16)


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


@requires_grouped_mm
def test_no_graph_breaks() -> None:
    torch._dynamo.reset()

    set_offsets(torch.tensor([5, 16], dtype=torch.int32), reset_reference=True)

    model = MultiLoRALinear(base_linear=nn.Linear(32, 16), rank=8, n_adapters=2, alpha=16.0)

    opt_model = torch.compile(model)
    _ = opt_model(torch.randn(1, 16, 32))

    explanation = torch._dynamo.explain(opt_model)(torch.randn(1, 16, 32))
    for reason in explanation.break_reasons:
        print(reason)
    assert explanation.graph_break_count == 0, f"There are / is {explanation.graph_break_count} graph break(s)"


def test_initialization() -> None:
    """Test that MultiLoRALinear initializes correctly."""
    base = nn.Linear(10, 20)
    N_ADAPTERS = 3
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS, alpha=16.0)

    assert lora.rank == 4
    assert lora.n_adapters == 3
    assert lora.alpha == 16.0
    assert lora.scaling == 4.0  # alpha / r = 16 / 4
    assert lora.in_features == 10
    assert lora.out_features == 20
    assert len(lora.lora_A) == N_ADAPTERS
    assert len(lora.lora_B) == N_ADAPTERS
    for i in range(N_ADAPTERS):
        assert lora.lora_A[i].shape == (4, 10)  # [r, in_features]
        assert lora.lora_B[i].shape == (20, 4)  # [out_features, r]


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
    N_ADAPTERS = 3
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS)

    # Reset all adapters
    lora.reset_parameters()

    for i in range(N_ADAPTERS):
        # lora_B should be all zeros
        assert torch.allclose(lora.lora_B[i], torch.zeros_like(lora.lora_B[i]))
        # lora_A should have non-zero values (randomly initialized)
        assert not torch.allclose(lora.lora_A[i], torch.zeros_like(lora.lora_A[i]))


def test_reset_parameters_single_adapter() -> None:
    """Test resetting a single adapter's parameters."""
    base = nn.Linear(10, 20)
    N_ADAPTERS = 3
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS)

    # Set all to known values first
    with torch.no_grad():
        for i in range(N_ADAPTERS):
            lora.lora_A[i].fill_(1.0)
            lora.lora_B[i].fill_(1.0)

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
    N_ADAPTERS = 4
    set_offsets(torch.zeros(N_ADAPTERS, dtype=torch.int32, device="cuda"), reset_reference=True)
    base = nn.Linear(8, 24).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=8, n_adapters=4)

    x = torch.randn(8, 8).cuda()
    offsets = torch.tensor([3, 3, 6, 8], dtype=torch.int32).cuda()
    set_offsets(offsets)

    lora_output = lora(x)
    base_output = base(x)

    assert lora_output.shape == (8, 24), f"lora_output.shape: {lora_output.shape}"
    assert torch.allclose(lora_output, base_output, atol=1e-6)


def test_forward_with_non_zero_lora() -> None:
    """Test that with non-zero LoRA weights, output is different from base layer output."""
    N_ADAPTERS = 4
    set_offsets(torch.zeros(N_ADAPTERS, dtype=torch.int32, device="cuda"), reset_reference=True)
    base = nn.Linear(8, 24).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=4)

    # Initialize with non-zero LoRA weights
    with torch.no_grad():
        for i in range(N_ADAPTERS):
            lora.lora_B[i].normal_(0, 1e-1)

    x = torch.randn(6, 8).cuda()

    base_output = base(x)
    offsets_1 = torch.tensor([0, 2, 4, 6], dtype=torch.int32).cuda()
    set_offsets(offsets_1)
    lora_output_1 = lora(x)
    offsets_2 = torch.tensor([2, 2, 4, 6], dtype=torch.int32).cuda()
    set_offsets(offsets_2)
    lora_output_2 = lora(x)

    # Outputs should be different
    assert not torch.allclose(lora_output_1, base_output, atol=1e-6)
    assert not torch.allclose(lora_output_2, base_output, atol=1e-6)
    assert not torch.allclose(lora_output_1, lora_output_2, atol=1e-6)


def test_device_consistency() -> None:
    """Test that LoRA parameters are on the same device as base layer."""
    N_ADAPTERS = 4
    base = nn.Linear(8, 24)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS)

    for i in range(N_ADAPTERS):
        assert lora.lora_A[i].device == base.weight.device
        assert lora.lora_B[i].device == base.weight.device


def test_dtype_consistency() -> None:
    """Test that LoRA parameters have the same dtype as base layer."""
    N_ADAPTERS = 4
    base = nn.Linear(8, 24, dtype=torch.float32)
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS)

    for i in range(N_ADAPTERS):
        assert lora.lora_A[i].dtype == torch.float32
        assert lora.lora_B[i].dtype == torch.float32


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
    N_ADAPTERS = 2
    set_offsets(torch.zeros(N_ADAPTERS, dtype=torch.int32, device="cuda"), reset_reference=True)
    base = nn.Linear(8, 24).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS)

    x = torch.randn(4, 8, requires_grad=True, device="cuda")
    offsets = torch.tensor([2, 4], dtype=torch.int32, device="cuda")

    set_offsets(offsets)
    output = lora(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert lora.base_linear.weight.grad is None
    for i in range(N_ADAPTERS):
        assert lora.lora_A[i].grad is not None
        assert lora.lora_B[i].grad is not None
    assert x.grad is not None

    # Because B is initialized to zero, A will have zero grads
    for i in range(N_ADAPTERS):
        assert torch.allclose(lora.lora_A[i].grad, torch.zeros_like(lora.lora_A[i].grad))
        assert not torch.allclose(lora.lora_B[i].grad, torch.zeros_like(lora.lora_B[i].grad))


def test_gradient_flow_non_zero_lora() -> None:
    """Test that gradients flow through both base and LoRA parameters with non-zero LoRA weights."""
    N_ADAPTERS = 2
    set_offsets(torch.zeros(N_ADAPTERS, dtype=torch.int32, device="cuda"), reset_reference=True)
    base = nn.Linear(8, 24).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=4, n_adapters=N_ADAPTERS)
    with torch.no_grad():
        for i in range(N_ADAPTERS):
            nn.init.kaiming_uniform_(lora.lora_B[i], a=math.sqrt(5))

    x = torch.randn(4, 8, requires_grad=True, device="cuda")
    offsets = torch.tensor([2, 4], dtype=torch.int32, device="cuda")

    set_offsets(offsets)
    output = lora(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert lora.base_linear.weight.grad is None
    for i in range(N_ADAPTERS):
        assert lora.lora_A[i].grad is not None
        assert lora.lora_B[i].grad is not None
    assert x.grad is not None

    # Check that gradients are non-zero
    for i in range(N_ADAPTERS):
        assert not torch.allclose(lora.lora_A[i].grad, torch.zeros_like(lora.lora_A[i].grad))
        assert not torch.allclose(lora.lora_B[i].grad, torch.zeros_like(lora.lora_B[i].grad))


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
    set_offsets(torch.zeros(n_adapters, dtype=torch.int32, device="cuda"), reset_reference=True)
    base = nn.Linear(8, 24).cuda()
    lora = MultiLoRALinear(base_linear=base, rank=rank, n_adapters=n_adapters, alpha=alpha)

    assert lora.rank == rank
    assert lora.n_adapters == n_adapters
    assert lora.alpha == alpha
    assert lora.scaling == alpha / rank
    assert len(lora.lora_A) == n_adapters
    assert len(lora.lora_B) == n_adapters
    for i in range(n_adapters):
        assert lora.lora_A[i].shape == (rank, 8)
        assert lora.lora_B[i].shape == (24, rank)

    # Test forward pass works
    x = torch.randn(n_adapters * 2, 8).cuda()
    offsets = torch.arange(2, n_adapters * 2 + 1, 2, dtype=torch.long).cuda()
    set_offsets(offsets)
    output = lora(x)
    assert output.shape == (n_adapters * 2, 24)
