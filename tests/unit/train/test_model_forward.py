from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from prime_rl.trainer.model import forward
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL


class _CaptureModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        if "input_ids" in kwargs:
            shape = kwargs["input_ids"].shape
        else:
            shape = kwargs["inputs_embeds"].shape[:2]
        return {"logits": torch.zeros(*shape, 4)}


class _PrimeCaptureModel(PreTrainedModelPrimeRL):
    config_class = PretrainedConfig

    def __init__(self):
        super().__init__(PretrainedConfig())
        self.kwargs = None

    def forward(self, seq_lens: torch.Tensor | None = None, **kwargs):
        self.kwargs = {**kwargs, "seq_lens": seq_lens}
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
        mm_token_type_ids=mm_token_type_ids,
    )

    assert model.kwargs is not None
    # MRoPE families (image_grid_thw present) get position_ids stripped.
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
    )

    assert model.kwargs is not None
    assert "position_ids" not in model.kwargs
    assert "mm_token_type_ids" not in model.kwargs


def test_forward_keeps_position_ids_for_non_mrope_vlm():
    """Non-MRoPE VLM families (no ``image_grid_thw``) keep the trainer's
    pre-computed ``position_ids``."""
    model = _CaptureModel(SimpleNamespace(model_type="gemma3"))
    input_ids = torch.tensor([[1, 10, 10, 2]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    forward(
        model,
        input_ids,
        position_ids,
        mm_kwargs={"pixel_values": torch.ones(2, 3)},
    )

    assert model.kwargs is not None
    torch.testing.assert_close(model.kwargs["position_ids"], position_ids)


def test_forward_does_not_leak_seq_lens_to_generic_text_models():
    model = _CaptureModel(SimpleNamespace(model_type="qwen3"))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    seq_lens = torch.tensor([2, 2])

    forward(model, input_ids, position_ids, seq_lens=seq_lens)

    assert model.kwargs is not None
    assert "seq_lens" not in model.kwargs


def test_forward_passes_typed_seq_lens_to_custom_models():
    model = _PrimeCaptureModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    seq_lens = torch.tensor([2, 2])

    forward(model, input_ids, position_ids, seq_lens=seq_lens)

    assert model.kwargs is not None
    torch.testing.assert_close(model.kwargs["seq_lens"], seq_lens)


def test_forward_strips_position_ids_without_leaking_seq_lens_for_mrope_vlm():
    """Generic VLMs do not receive PrimeRL-only packed-boundary kwargs."""
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_5_moe"))
    input_ids = torch.tensor([[1, 10, 10, 2, 20, 20]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
    seq_lens = torch.tensor([3, 3])

    forward(
        model,
        input_ids,
        position_ids,
        mm_kwargs={"pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2]])},
        seq_lens=seq_lens,
    )

    assert model.kwargs is not None
    assert "position_ids" not in model.kwargs
    assert "seq_lens" not in model.kwargs
