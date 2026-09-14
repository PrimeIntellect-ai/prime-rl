import pytest
import torch
from torch import nn

from prime_rl.utils.mx_verification import perturb_weights, snapshot_weights, verify_weights


def _model():
    model = nn.Module()
    model.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4).T)
    model.tied = model.weight
    model.bias = nn.Parameter(torch.tensor([1.001, 2.003], dtype=torch.float32))
    return model


@pytest.mark.parametrize("damage", [None, "rounding", "address", "alias", "partial"])
@torch.no_grad()
def test_initial_verification_requires_exact_values_addresses_and_aliases(damage):
    model = _model()
    snapshot = snapshot_weights(model)
    perturb_weights(model, snapshot)
    assert torch.isnan(model.weight).all()
    for name, expected in snapshot.values.items():
        if damage != "partial" or name != "bias":
            getattr(model, name).copy_(expected)
    if damage == "rounding":
        model.bias.copy_(model.bias.bfloat16().float())
    elif damage == "address":
        model.bias = nn.Parameter(model.bias.clone())
    elif damage == "alias":
        model.tied = nn.Parameter(model.weight.clone())
    result = verify_weights(model, snapshot)
    assert result["passed"] is (damage is None)
    assert result["changed_tensors"] == 2
    if damage is None:
        assert result["verified_tensors"] == result["verified_fp32_tensors"] == 2


def test_snapshot_budget_fails_before_mutation():
    model = _model()
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match="CPU bytes"):
        snapshot_weights(model, max_bytes=1)
    assert torch.equal(model.weight, original)


@pytest.mark.parametrize("registered", [False, True])
@torch.no_grad()
def test_initial_verification_with_mla_parameters_and_fp8(registered):
    model = _model()
    model.kv_b_proj = nn.Identity()
    for name in ("W_UV", "W_UK_T"):
        tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).T
        setattr(model, name, nn.Parameter(tensor, requires_grad=False) if registered else tensor)
    model.fp8 = nn.Parameter(torch.ones(4, dtype=torch.float8_e4m3fn), requires_grad=False)
    snapshot = snapshot_weights(model)
    perturb_weights(model, snapshot)
    for name, value in snapshot.values.items():
        getattr(model, name).copy_(value)
    result = verify_weights(model, snapshot)
    assert result["passed"]
    assert result["changed_tensors"] == result["verified_tensors"] == 5
    model.fp8.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite initial weights: fp8"):
        snapshot_weights(model)
