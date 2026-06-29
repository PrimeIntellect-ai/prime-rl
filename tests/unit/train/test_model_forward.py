from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prime_rl.multimodal.adapters.base import ForwardPolicy
from prime_rl.trainer.model import forward


class _CaptureModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        input_ids = kwargs["input_ids"]
        return {"logits": torch.zeros(*input_ids.shape, 4)}


def test_forward_passes_renderer_mm_token_type_ids_through():
    """``forward()`` forwards renderer-supplied ``mm_token_type_ids``
    verbatim — the trainer no longer auto-computes from the model
    config, since the renderer is the source of truth."""
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_vl"))
    input_ids = torch.tensor([[1, 10, 10, 2, 20]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    pixel_values = torch.ones(2, 3)
    image_grid_thw = torch.tensor([[1, 1, 2]])
    mm_token_type_ids = torch.tensor([[0, 1, 1, 0, 2]])

    forward(
        model,
        input_ids,
        position_ids,
        mm_kwargs={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        mm_forward_policy=ForwardPolicy(pass_position_ids_with_mm=False),
        mm_token_type_ids=mm_token_type_ids,
    )

    assert model.kwargs is not None
    assert "position_ids" not in model.kwargs
    torch.testing.assert_close(model.kwargs["pixel_values"], pixel_values)
    torch.testing.assert_close(model.kwargs["image_grid_thw"], image_grid_thw)
    torch.testing.assert_close(model.kwargs["mm_token_type_ids"], mm_token_type_ids)


def test_forward_omits_mm_token_type_ids_when_renderer_does_not_supply():
    """When the renderer doesn't ship ``mm_token_type_ids`` (text-only
    or a family without modality markers), ``forward()`` doesn't
    fabricate one."""
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_vl"))
    input_ids = torch.tensor([[1, 10, 10, 2]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    forward(
        model,
        input_ids,
        position_ids,
        mm_kwargs={"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])},
        mm_forward_policy=ForwardPolicy(pass_position_ids_with_mm=False),
    )

    assert model.kwargs is not None
    assert "position_ids" not in model.kwargs
    assert "mm_token_type_ids" not in model.kwargs


def test_forward_keeps_position_ids_for_non_mrope_vlm():
    """Families whose adapter asks for position_ids keep the trainer's values."""
    model = _CaptureModel(SimpleNamespace(model_type="gemma3"))
    input_ids = torch.tensor([[1, 10, 10, 2]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    forward(
        model,
        input_ids,
        position_ids,
        mm_kwargs={"pixel_values": torch.ones(2, 3)},
        mm_forward_policy=ForwardPolicy(pass_position_ids_with_mm=True),
    )

    assert model.kwargs is not None
    torch.testing.assert_close(model.kwargs["position_ids"], position_ids)


def test_forward_policy_can_require_mm_token_type_ids():
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_vl"))
    input_ids = torch.tensor([[1, 10, 10, 2]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with pytest.raises(ValueError, match="mm_token_type_ids"):
        forward(
            model,
            input_ids,
            position_ids,
            mm_kwargs={"pixel_values": torch.ones(2, 3)},
            mm_forward_policy=ForwardPolicy(requires_mm_token_type_ids=True),
        )
