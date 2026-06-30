from types import SimpleNamespace

import torch
import torch.nn as nn

import prime_rl.trainer.model as trainer_model
from prime_rl.trainer.model import forward


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.visual = nn.Linear(4, 4)
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(8, 4)
        self.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.model.language_model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


class _FakeParallelDims:
    ep_enabled = False

    def get_mesh(self, name):
        assert name == "hsdp"
        return "hsdp-mesh"


def _vlm_cp_config():
    return SimpleNamespace(
        reduce_dtype="bfloat16",
        fsdp_cpu_offload=False,
        reshard_after_forward=True,
        cp=2,
        name="tiny-vlm",
        vlm=SimpleNamespace(
            vision_encoder_attr="model.visual",
            language_model_attr="model.language_model",
            freeze_vision_encoder=True,
        ),
    )


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


def test_setup_fsdp_vlm_context_parallel_ignores_frozen_vision_encoder(monkeypatch):
    model = _TinyVLM()
    for param in model.model.visual.parameters():
        param.requires_grad = False

    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(trainer_model, "fully_shard", fake_fully_shard)

    trainer_model.setup_fsdp(model, _vlm_cp_config(), _FakeParallelDims())

    sharded_modules = [module for module, _ in calls]
    assert model.model.language_model.embed_tokens in sharded_modules
    assert model.model.visual not in sharded_modules

    root_kwargs = calls[-1][1]
    ignored_params = root_kwargs["ignored_params"]
    assert set(model.model.visual.parameters()) <= ignored_params
    assert model.model.language_model.embed_tokens.weight not in ignored_params


def test_setup_fsdp_vlm_context_parallel_shards_trainable_vision_encoder(monkeypatch):
    model = _TinyVLM()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(trainer_model, "fully_shard", fake_fully_shard)

    trainer_model.setup_fsdp(model, _vlm_cp_config(), _FakeParallelDims())

    sharded_modules = [module for module, _ in calls]
    assert model.model.visual in sharded_modules
    assert model.model.language_model.embed_tokens in sharded_modules

    root_kwargs = calls[-1][1]
    assert root_kwargs["ignored_params"] is None


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


def test_forward_strips_position_ids_and_forwards_seq_lens_for_mrope_vlm():
    """Qwen-style MRoPE VLMs build 3D positions internally from seq_lens."""
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
    torch.testing.assert_close(model.kwargs["seq_lens"], seq_lens)


def test_forward_accepts_premerged_inputs_embeds_without_cp_metadata():
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_5_moe"))
    inputs_embeds = torch.randn(1, 4, 8)
    position_ids = torch.arange(12).view(3, 1, 4)
    seq_lens = torch.tensor([2, 2])

    forward(
        model,
        None,
        position_ids,
        inputs_embeds=inputs_embeds,
        seq_lens=seq_lens,
        seq_lens_are_global=True,
    )

    assert model.kwargs is not None
    assert "input_ids" not in model.kwargs
    torch.testing.assert_close(model.kwargs["inputs_embeds"], inputs_embeds)
    torch.testing.assert_close(model.kwargs["position_ids"], position_ids)
    torch.testing.assert_close(model.kwargs["seq_lens"], seq_lens)
    assert model.kwargs["seq_lens_are_global"] is True


def test_forward_passes_raw_vlm_inputs_with_context_parallel_metadata():
    model = _CaptureModel(SimpleNamespace(model_type="qwen3_5_moe"))
    input_ids = torch.tensor([[1, 10, 10, 2]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    seq_lens = torch.tensor([4])
    cp_group = object()

    forward(
        model,
        input_ids,
        position_ids,
        mm_kwargs={"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])},
        mm_token_type_ids=torch.tensor([[0, 1, 1, 0]]),
        seq_lens=seq_lens,
        cp_group=cp_group,
        cp_rank=1,
        cp_world_size=2,
        cp_style="ulysses",
    )

    assert model.kwargs is not None
    torch.testing.assert_close(model.kwargs["input_ids"], input_ids)
    assert "inputs_embeds" not in model.kwargs
    assert "position_ids" not in model.kwargs
    torch.testing.assert_close(model.kwargs["seq_lens"], seq_lens)
    assert model.kwargs["cp_group"] is cp_group
    assert model.kwargs["cp_rank"] == 1
    assert model.kwargs["cp_world_size"] == 2
    assert model.kwargs["cp_style"] == "ulysses"
