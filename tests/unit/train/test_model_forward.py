from types import SimpleNamespace

import torch
import torch.nn as nn

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


def test_forward_adds_qwen3_vl_mm_token_type_ids():
    image_token_id = 10
    video_token_id = 20
    model = _CaptureModel(
        SimpleNamespace(model_type="qwen3_vl", image_token_id=image_token_id, video_token_id=video_token_id)
    )
    input_ids = torch.tensor([[1, image_token_id, image_token_id, 2, video_token_id]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    pixel_values = torch.ones(2, 3)
    image_grid_thw = torch.tensor([[1, 1, 2]])

    forward(model, input_ids, position_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    assert model.kwargs is not None
    assert "position_ids" not in model.kwargs
    torch.testing.assert_close(model.kwargs["pixel_values"], pixel_values)
    torch.testing.assert_close(model.kwargs["image_grid_thw"], image_grid_thw)
    torch.testing.assert_close(model.kwargs["mm_token_type_ids"], torch.tensor([[0, 1, 1, 0, 2]]))


def test_forward_skips_mm_token_type_ids_for_other_vlm_models():
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_5_moe", image_token_id=10))
    input_ids = torch.tensor([[1, 10, 10, 2]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    forward(model, input_ids, position_ids, pixel_values=torch.ones(2, 3), image_grid_thw=torch.tensor([[1, 1, 2]]))

    assert model.kwargs is not None
    assert "position_ids" not in model.kwargs
    assert "mm_token_type_ids" not in model.kwargs
