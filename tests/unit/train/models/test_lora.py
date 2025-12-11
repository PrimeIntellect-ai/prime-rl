import pytest
import torch
from torch import nn

from prime_rl.trainer.models.layers.lora import _LORA_PREFIX, LoRALinear


@pytest.fixture
def base_layer() -> nn.Linear:
    """Create a simple base linear layer for testing."""
    torch.manual_seed(42)
    return nn.Linear(in_features=64, out_features=32, bias=True)


@pytest.fixture
def lora_layer(base_layer: nn.Linear) -> LoRALinear:
    """Create a LoRALinear layer wrapping the base layer."""
    return LoRALinear(base_layer, rank=8, alpha=16.0)


def test_state_dict_keys_without_prefix(lora_layer: LoRALinear) -> None:
    """state_dict() should have keys WITHOUT the 'base_layer.' prefix."""
    state_dict = lora_layer.state_dict()

    # Check that base layer params don't have the prefix
    assert "weight" in state_dict
    assert "bias" in state_dict

    # Check that LoRA params are present
    assert "lora_A" in state_dict
    assert "lora_B" in state_dict

    # Ensure the prefix is NOT in any key
    for key in state_dict.keys():
        assert not key.startswith(_LORA_PREFIX), f"Key '{key}' should not have prefix '{_LORA_PREFIX}'"


def test_base_layer_params_frozen(lora_layer: LoRALinear) -> None:
    """Base layer parameters should have requires_grad=False."""
    for name, param in lora_layer.named_parameters():
        if name not in ("lora_A", "lora_B"):
            assert not param.requires_grad, f"Base layer param '{name}' should be frozen"


def test_lora_params_trainable(lora_layer: LoRALinear) -> None:
    """LoRA parameters should have requires_grad=True."""
    for name, param in lora_layer.named_parameters():
        if name in ("lora_A", "lora_B"):
            assert param.requires_grad, f"LoRA param '{name}' should be trainable"


def test_load_state_dict_into_lora_layer(lora_layer: LoRALinear) -> None:
    """Loading a state_dict back into the same LoRALinear should work."""
    original_state_dict = lora_layer.state_dict()

    # Create a new LoRALinear with different weights
    torch.manual_seed(123)
    new_base = nn.Linear(in_features=64, out_features=32, bias=True)
    new_lora = LoRALinear(new_base, rank=8, alpha=16.0)

    # Verify weights are different before loading
    assert not torch.allclose(new_lora.state_dict()["weight"], original_state_dict["weight"])

    # Load the original state dict
    new_lora.load_state_dict(original_state_dict)

    # Verify weights match after loading
    for key in original_state_dict:
        assert torch.allclose(new_lora.state_dict()[key], original_state_dict[key]), (
            f"State dict key '{key}' did not match after loading"
        )


def test_load_lora_state_dict_into_linear(lora_layer: LoRALinear) -> None:
    """Loading a LoRA state_dict into a regular nn.Linear should work for base params."""
    lora_state_dict = lora_layer.state_dict()

    # Create a new nn.Linear with the same dimensions
    target_linear = nn.Linear(in_features=64, out_features=32, bias=True)

    # Load should work with strict=False (ignoring lora_A, lora_B)
    target_linear.load_state_dict(lora_state_dict, strict=False)

    # Verify base weights match
    assert torch.allclose(target_linear.weight, lora_state_dict["weight"])
    assert torch.allclose(target_linear.bias, lora_state_dict["bias"])


def test_load_linear_state_dict_into_lora() -> None:
    """Loading a regular nn.Linear state_dict into LoRALinear should work."""
    torch.manual_seed(42)
    source_linear = nn.Linear(in_features=64, out_features=32, bias=True)
    source_state_dict = source_linear.state_dict()

    # Create a LoRALinear with different initial weights
    torch.manual_seed(123)
    target_base = nn.Linear(in_features=64, out_features=32, bias=True)
    target_lora = LoRALinear(target_base, rank=8, alpha=16.0)

    # Load nn.Linear state dict into LoRALinear with strict=False
    target_lora.load_state_dict(source_state_dict, strict=False)

    # Verify base weights match
    loaded_state_dict = target_lora.state_dict()
    assert torch.allclose(loaded_state_dict["weight"], source_state_dict["weight"])
    assert torch.allclose(loaded_state_dict["bias"], source_state_dict["bias"])


def test_state_dict_in_sequential() -> None:
    """state_dict keys should be correct when LoRALinear is in a Sequential."""
    base_layer = nn.Linear(in_features=64, out_features=32)
    lora_layer = LoRALinear(base_layer, rank=8)

    model = nn.Sequential(lora_layer)
    state_dict = model.state_dict()

    # Keys should be prefixed with '0.' (Sequential index) but not 'base_layer.'
    expected_keys = {"0.weight", "0.bias", "0.lora_A", "0.lora_B"}
    assert set(state_dict.keys()) == expected_keys


def test_state_dict_in_module_dict() -> None:
    """state_dict keys should be correct when LoRALinear is in a ModuleDict."""
    base_layer = nn.Linear(in_features=64, out_features=32)
    lora_layer = LoRALinear(base_layer, rank=8)

    model = nn.ModuleDict({"proj": lora_layer})
    state_dict = model.state_dict()

    # Keys should be prefixed with 'proj.' but not 'base_layer.'
    expected_keys = {"proj.weight", "proj.bias", "proj.lora_A", "proj.lora_B"}
    assert set(state_dict.keys()) == expected_keys


def test_forward_uses_both_weights() -> None:
    """Forward pass should use both base and LoRA weights."""
    torch.manual_seed(42)
    base_layer = nn.Linear(in_features=64, out_features=32, bias=False)
    lora_layer = LoRALinear(base_layer, rank=8, alpha=8.0)

    # Set LoRA B to non-zero to ensure LoRA contribution
    lora_layer.lora_B.data.fill_(0.1)

    x = torch.randn(2, 64)

    # Get outputs
    base_output = base_layer(x)
    lora_output = lora_layer(x)

    # Outputs should differ due to LoRA contribution
    assert not torch.allclose(base_output, lora_output), "LoRA should contribute to output"


def test_forward_with_zero_lora_b_equals_base() -> None:
    """Forward pass with zero lora_B should equal base layer output."""
    torch.manual_seed(42)
    base_layer = nn.Linear(in_features=64, out_features=32, bias=True)

    # Clone base layer for comparison
    base_layer_clone = nn.Linear(in_features=64, out_features=32, bias=True)
    base_layer_clone.load_state_dict(base_layer.state_dict())

    lora_layer = LoRALinear(base_layer, rank=8, alpha=8.0)
    # lora_B is initialized to zero by default

    x = torch.randn(2, 64)

    base_output = base_layer_clone(x)
    lora_output = lora_layer(x)

    # Outputs should be the same when lora_B is zero
    assert torch.allclose(base_output, lora_output, atol=1e-6), "With zero lora_B, output should match base layer"


def test_getattr_forwards_to_base_layer() -> None:
    """__getattr__ should forward missing attributes to the base layer."""
    base_layer = nn.Linear(in_features=64, out_features=32, bias=True)
    lora_layer = LoRALinear(base_layer, rank=8)

    # in_features and out_features should be accessible via __getattr__
    assert lora_layer.in_features == 64
    assert lora_layer.out_features == 32

    # weight and bias should also be accessible
    assert lora_layer.weight is base_layer.weight
    assert lora_layer.bias is base_layer.bias


def test_getattr_raises_for_missing_attribute() -> None:
    """__getattr__ should raise AttributeError for truly missing attributes."""
    base_layer = nn.Linear(in_features=64, out_features=32)
    lora_layer = LoRALinear(base_layer, rank=8)

    with pytest.raises(AttributeError):
        _ = lora_layer.nonexistent_attribute


def test_sequential_base_layer_getitem() -> None:
    """__getitem__ should forward to base layer when base is Sequential."""
    layer0 = nn.Linear(64, 128)
    layer1 = nn.ReLU()
    layer2 = nn.Linear(128, 32)
    base_sequential = nn.Sequential(layer0, layer1, layer2)

    lora_layer = LoRALinear(base_sequential, rank=8, in_features=64, out_features=32)

    # __getitem__ should return the correct sublayer
    assert lora_layer[0] is layer0
    assert lora_layer[1] is layer1
    assert lora_layer[2] is layer2


def test_sequential_base_layer_state_dict() -> None:
    """state_dict should work correctly when base layer is Sequential."""
    base_sequential = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    lora_layer = LoRALinear(base_sequential, rank=8, in_features=64, out_features=32)

    state_dict = lora_layer.state_dict()

    # Should contain Sequential sublayer keys without base_layer prefix
    assert "0.weight" in state_dict
    assert "0.bias" in state_dict
    assert "2.weight" in state_dict
    assert "2.bias" in state_dict

    # Should contain LoRA params
    assert "lora_A" in state_dict
    assert "lora_B" in state_dict

    # Should NOT have base_layer prefix
    for key in state_dict.keys():
        assert not key.startswith(_LORA_PREFIX), f"Key '{key}' should not have prefix '{_LORA_PREFIX}'"


def test_sequential_base_layer_load_state_dict() -> None:
    """Loading state_dict should work correctly when base layer is Sequential."""
    base_sequential = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    lora_layer = LoRALinear(base_sequential, rank=8, in_features=64, out_features=32)
    original_state_dict = lora_layer.state_dict()

    # Create new LoRA with Sequential base and load state dict
    torch.manual_seed(999)
    new_base_sequential = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    new_lora = LoRALinear(new_base_sequential, rank=8, in_features=64, out_features=32)

    # Verify weights differ before loading
    assert not torch.allclose(new_lora.state_dict()["0.weight"], original_state_dict["0.weight"])

    new_lora.load_state_dict(original_state_dict)

    # Verify all weights match after loading
    for key in original_state_dict:
        assert torch.allclose(new_lora.state_dict()[key], original_state_dict[key]), (
            f"State dict key '{key}' did not match after loading"
        )


def test_sequential_base_layer_forward() -> None:
    """Forward pass should work correctly when base layer is Sequential."""
    torch.manual_seed(42)
    base_sequential = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )

    # Clone for comparison
    base_sequential_clone = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    base_sequential_clone.load_state_dict(base_sequential.state_dict())

    lora_layer = LoRALinear(base_sequential, rank=8, in_features=64, out_features=32)
    # lora_B is initialized to zero by default

    x = torch.randn(2, 64)

    base_output = base_sequential_clone(x)
    lora_output = lora_layer(x)

    # With zero lora_B, outputs should match
    assert torch.allclose(base_output, lora_output, atol=1e-6), "With zero lora_B, Sequential output should match base"
