from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import prime_rl.trainer.model as trainer_model
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
        seq_lens=torch.tensor([input_ids.shape[1]]),
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
        seq_lens=torch.tensor([input_ids.shape[1]]),
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
        seq_lens=torch.tensor([input_ids.shape[1]]),
        mm_kwargs={"pixel_values": torch.ones(2, 3)},
    )

    assert model.kwargs is not None
    torch.testing.assert_close(model.kwargs["position_ids"], position_ids)


@pytest.mark.parametrize("freeze_vision_encoder,expected", [(True, False), (False, True)])
def test_setup_fsdp_keeps_frozen_vision_encoder_unsharded_after_forward(
    monkeypatch: pytest.MonkeyPatch, freeze_vision_encoder: bool, expected: bool
):
    vision_encoder = object()
    language_model = SimpleNamespace(layers=[], embed_tokens=object(), norm=object())
    model = SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=False), lm_head=object())
    config = SimpleNamespace(
        reduce_dtype="bfloat16",
        fsdp_cpu_offload=False,
        fusions=SimpleNamespace(shard_fused_on_dim1=False),
        reshard_after_forward=True,
        vlm=SimpleNamespace(
            vision_encoder_attr="vision",
            language_model_attr="language",
            freeze_vision_encoder=freeze_vision_encoder,
        ),
        moe_router_dtype="bfloat16",
    )
    parallel_dims = SimpleNamespace(ep_enabled=False, get_mesh=lambda _: object())
    shard_calls = []

    monkeypatch.setattr(trainer_model, "get_vision_encoder", lambda *_args, **_kwargs: vision_encoder)
    monkeypatch.setattr(trainer_model, "get_language_model", lambda *_args, **_kwargs: language_model)
    monkeypatch.setattr(trainer_model, "fully_shard", lambda target, **kwargs: shard_calls.append((target, kwargs)))

    trainer_model.setup_fsdp(model, config, parallel_dims)

    vision_kwargs = next(kwargs for target, kwargs in shard_calls if target is vision_encoder)
    assert vision_kwargs["reshard_after_forward"] is expected
