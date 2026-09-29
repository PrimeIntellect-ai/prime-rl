import pytest
import torch
from torch import nn

from prime_rl.utils.mx_precision import build_trainer_context, wire_dtype_overrides


def test_models_without_policy_keep_default_transport_dtype():
    model = nn.Linear(2, 2)
    assert wire_dtype_overrides(model, model.state_dict()) == {}


def test_model_policy_receives_exact_state_dict_names():
    class Model(nn.Module):
        @staticmethod
        def keep_in_fp32_for_weight_transfer(name):
            return name.endswith("selection_bias") or name.endswith("scale")

    tensors = {
        "model._orig_mod.layers.0.mlp.router.selection_bias": torch.tensor([1.000123]),
        "model.norm.scale": torch.tensor(1.000123),
        "model.layers.0.mlp.experts.gate_proj": torch.ones(2, 3),
    }
    assert wire_dtype_overrides(Model(), tensors) == {
        "model._orig_mod.layers.0.mlp.router.selection_bias": torch.float32,
        "model.norm.scale": torch.float32,
    }


class _OldContext:
    """Context without transfer-dtype overrides."""

    def __init__(self):
        self.wire_dtype_overrides = {}


class _NewContext:
    def __init__(self, wire_dtype_overrides=None):
        self.wire_dtype_overrides = dict(wire_dtype_overrides or {})


class _NeedsFP32(nn.Module):
    @staticmethod
    def keep_in_fp32_for_weight_transfer(name):
        return name.endswith("scale")


def test_overrides_are_omitted_for_a_client_that_cannot_carry_them():
    """Models without overrides can use a context that does not support them."""
    model = nn.Linear(2, 2)
    context = build_trainer_context(_OldContext, model, model.state_dict())
    assert context.wire_dtype_overrides == {}


def test_overrides_reach_a_client_that_accepts_them():
    tensors = {"model.norm.scale": torch.tensor(1.0)}
    context = build_trainer_context(_NewContext, _NeedsFP32(), tensors)
    assert context.wire_dtype_overrides == {"model.norm.scale": torch.float32}


def test_required_overrides_fail_loudly_rather_than_transfer_at_bf16():
    """Required FP32 tensors must not be silently transferred as BF16."""
    tensors = {"model.norm.scale": torch.tensor(1.0)}
    with pytest.raises(RuntimeError, match="wire_dtype_overrides"):
        build_trainer_context(_OldContext, _NeedsFP32(), tensors)


def test_installed_fsdp_context_carries_required_overrides():
    """The stand-ins above cannot catch a signature change in the real client."""
    context_module = pytest.importorskip("modelexpress_rl.train.context")
    tensors = {"model.norm.scale": torch.tensor(1.0)}
    context = build_trainer_context(context_module.FSDPTrainerContext, _NeedsFP32(), tensors)
    assert dict(context.wire_dtype_overrides) == {"model.norm.scale": torch.float32}
