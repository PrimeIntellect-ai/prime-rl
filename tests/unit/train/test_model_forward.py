from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prime_rl.configs.shared import VLMConfig
from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.model import configure_trainable_parameters, forward
from prime_rl.trainer.models.layers.lora import MultiLoRALinear


class _CaptureModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        input_ids = kwargs["input_ids"]
        return {"logits": torch.zeros(*input_ids.shape, 4)}


class _ToyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.gate_proj = nn.Linear(4, 4)
        self.language_model = nn.Module()
        self.language_model.gate_proj = nn.Linear(4, 4)


def test_frozen_vision_encoder_is_excluded_from_lora(monkeypatch):
    runs = SimpleNamespace(max_runs=1, register_module=lambda *args: None)
    monkeypatch.setattr("prime_rl.trainer.lora.get_multi_run_manager", lambda: runs)
    monkeypatch.setattr("prime_rl.trainer.models.layers.lora.base.LORA_NUM_TOKENS", torch.zeros(1, dtype=torch.int32))
    monkeypatch.setattr("prime_rl.trainer.models.layers.lora.base.SCALING_FACTORS", torch.ones(1, dtype=torch.bfloat16))
    model = _ToyVLM()
    config = ModelConfig(
        vlm=VLMConfig(vision_encoder_attr="visual", language_model_attr="language_model"),
        lora=LoRAConfig(target_modules=["gate_proj"]),
        moe_router_dtype="bfloat16",
    )

    configure_trainable_parameters(model, config, SimpleNamespace(ep_enabled=False))

    assert isinstance(model.visual.gate_proj, nn.Linear)
    assert not isinstance(model.visual.gate_proj, MultiLoRALinear)
    assert all(not parameter.requires_grad for parameter in model.visual.parameters())
    assert isinstance(model.language_model.gate_proj, MultiLoRALinear)
    lora_parameters = [
        parameter for name, parameter in model.language_model.gate_proj.named_parameters() if "lora_" in name
    ]
    assert lora_parameters
    assert all(parameter.requires_grad for parameter in lora_parameters)
    base_parameters = [
        parameter for name, parameter in model.language_model.gate_proj.named_parameters() if "lora_" not in name
    ]
    assert base_parameters
    assert all(not parameter.requires_grad for parameter in base_parameters)


def test_frozen_vision_encoder_rejects_lora_targets_only_in_vision(monkeypatch):
    runs = SimpleNamespace(max_runs=1, register_module=lambda *args: None)
    monkeypatch.setattr("prime_rl.trainer.lora.get_multi_run_manager", lambda: runs)
    model = _ToyVLM()
    config = ModelConfig(
        vlm=VLMConfig(vision_encoder_attr="visual", language_model_attr="language_model"),
        lora=LoRAConfig(target_modules=["visual"]),
        moe_router_dtype="bfloat16",
    )

    with pytest.raises(ValueError, match="No LoRA target modules found"):
        configure_trainable_parameters(model, config, SimpleNamespace(ep_enabled=False))


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


def test_forward_omits_prime_only_kwargs_for_hf_mrope_vlm():
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
