from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from prime_rl.trainer.model import forward as trainer_forward
from prime_rl.trainer.models import get_custom_vlm_cls
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h_omni import NemotronHOmniForCausalLM
from prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni import (
    _load_remote_vision_components,
    merge_image_embeddings,
)
from prime_rl.utils.vlm import VLM_REGISTRY


def test_nemotron_h_omni_is_registered_as_packed_vlm():
    config = PretrainedConfig()
    config.model_type = "nemotron_h_omni"

    assert get_custom_vlm_cls(config) is NemotronHOmniForCausalLM
    assert VLM_REGISTRY[config.model_type].vision_encoder_attr == "model.vision_model"
    assert VLM_REGISTRY[config.model_type].language_model_attr == "model.language_model"
    assert NemotronHOmniForCausalLM.is_hf_state_dict({"language_model.backbone.embeddings.weight": torch.empty(0)})
    assert NemotronHOmniForCausalLM.is_prime_state_dict({"model.language_model.embed_tokens.weight": torch.empty(0)})


def test_merge_image_embeddings_replaces_only_image_tokens():
    input_ids = torch.tensor([[1, 18, 2, 18]])
    inputs_embeds = torch.zeros(1, 4, 3)
    image_embeds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    merged = merge_image_embeddings(input_ids, inputs_embeds, image_embeds, image_token_id=18)

    torch.testing.assert_close(merged[0, 1], image_embeds[0])
    torch.testing.assert_close(merged[0, 3], image_embeds[1])
    torch.testing.assert_close(merged[0, [0, 2]], torch.zeros(2, 3))


@pytest.mark.parametrize("num_image_embeddings", [1, 3])
def test_merge_image_embeddings_rejects_token_count_mismatch(
    num_image_embeddings,
):
    with pytest.raises(ValueError, match="image token count"):
        merge_image_embeddings(
            torch.tensor([[18, 18]]),
            torch.zeros(1, 2, 3),
            torch.ones(num_image_embeddings, 3),
            image_token_id=18,
        )


def test_merge_image_embeddings_rejects_hidden_size_mismatch():
    with pytest.raises(ValueError, match="hidden size"):
        merge_image_embeddings(
            torch.tensor([[18]]),
            torch.zeros(1, 1, 3),
            torch.ones(1, 4),
            image_token_id=18,
        )


def test_nemotron_h_model_requires_exactly_one_embedding_input():
    from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHModel

    model = NemotronHModel(_language_config())
    seq_lens = torch.tensor([2])

    with pytest.raises(ValueError, match="[Ee]xactly one"):
        model(seq_lens=seq_lens)
    with pytest.raises(ValueError, match="[Ee]xactly one"):
        model(
            input_ids=torch.tensor([[1, 2]]),
            inputs_embeds=torch.zeros(1, 2, 4),
            seq_lens=seq_lens,
        )


class RadioLayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.full((4,), config.layerscale_value))


class _FakeVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.ones(1))
        self.layer_scale = RadioLayerScale(config)
        self.register_buffer(
            "summary_idxs",
            torch.tensor(config.summary_idxs),
        )
        self.preprocessor_external = False

    def make_preprocessor_external(self):
        self.preprocessor_external = True


class _FakeVisionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, pixel_values, vision_model):
        if isinstance(pixel_values, list):
            return torch.cat([self.forward(item, vision_model) for item in pixel_values], dim=0)
        if pixel_values.ndim == 4:
            return pixel_values.new_ones(pixel_values.shape[0], 1, 4) * vision_model.anchor
        return pixel_values.unsqueeze(0) * vision_model.anchor


def _language_config():
    return NemotronHConfig(
        hidden_size=4,
        vocab_size=32,
        layers_block_type=[],
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        moe_shared_expert_intermediate_size=4,
        moe_latent_size=4,
    )


def _composite_config():
    config = PretrainedConfig()
    config.model_type = "nemotron_h_omni"
    config.llm_config = _language_config()
    config.vision_config = SimpleNamespace(
        layerscale_value=1.0,
        summary_idxs=[0],
        num_channels=3,
    )
    config.img_context_token_id = 18
    config.force_image_size = 2
    config.patch_size = 2
    config.downsample_ratio = 0.5
    config.tie_word_embeddings = False
    return config


def _remote_composite_config(model_path: str, revision: str | None):
    config = _composite_config()
    config._name_or_path = model_path
    config._commit_hash = revision
    config.auto_map = {"AutoModelForImageTextToText": "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3"}
    return config


def test_remote_vision_loader_requires_explicit_trust(monkeypatch):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    calls = []
    monkeypatch.setattr(modeling, "_load_remote_model_module", lambda *args: calls.append(args))

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        _load_remote_vision_components(
            _remote_composite_config("nvidia/model", "a" * 40),
            trust_remote_code=False,
        )

    assert calls == []


def test_remote_vision_loader_rejects_unpinned_remote_source(monkeypatch):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    calls = []
    monkeypatch.setattr(modeling, "_load_remote_model_module", lambda *args: calls.append(args))

    with pytest.raises(ValueError, match="immutable commit"):
        _load_remote_vision_components(
            _remote_composite_config("nvidia/model", None),
            trust_remote_code=True,
        )

    assert calls == []


def test_remote_vision_loader_rejects_unversioned_local_directory(monkeypatch, tmp_path):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    model_path = tmp_path / "model"
    model_path.mkdir()
    calls = []
    monkeypatch.setattr(modeling, "_load_remote_model_module", lambda *args: calls.append(args))

    with pytest.raises(ValueError, match="40-hex snapshot"):
        _load_remote_vision_components(
            _remote_composite_config(str(model_path), None),
            trust_remote_code=True,
        )

    assert calls == []


def test_remote_vision_loader_accepts_trusted_local_snapshot(monkeypatch, tmp_path):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    model_path = tmp_path / "snapshots" / ("a" * 40)
    model_path.mkdir(parents=True)
    module = SimpleNamespace(
        RadioModel=_FakeVisionModel,
        NemotronH_Omni_Reasoning_V3VisionProjector=_FakeVisionProjector,
    )
    calls = []
    monkeypatch.setattr(
        modeling,
        "_load_remote_model_module",
        lambda *args: calls.append(args) or module,
    )

    components = _load_remote_vision_components(
        _remote_composite_config(str(model_path), None),
        trust_remote_code=True,
    )

    assert components == (_FakeVisionModel, _FakeVisionProjector)
    assert calls == [
        (
            "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3",
            str(model_path),
            None,
        )
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "nemotron_h", "model_type"),
        (
            "auto_map",
            {"AutoModelForImageTextToText": "modeling_other.UnexpectedModel"},
            "AutoModelForImageTextToText",
        ),
    ],
)
def test_remote_vision_loader_rejects_unexpected_checkpoint_contract(monkeypatch, field, value, message):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    config = _remote_composite_config("nvidia/model", "a" * 40)
    setattr(config, field, value)
    calls = []
    monkeypatch.setattr(modeling, "_load_remote_model_module", lambda *args: calls.append(args))

    with pytest.raises(ValueError, match=message):
        _load_remote_vision_components(config, trust_remote_code=True)

    assert calls == []


def test_custom_loading_paths_propagate_explicit_trust(monkeypatch):
    from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHPreTrainedModel

    config_calls = []
    pretrained_calls = []

    def fake_from_config(cls, config, **kwargs):
        config_calls.append((config, kwargs))
        return "from-config"

    def fake_from_pretrained(cls, *args, **kwargs):
        pretrained_calls.append((args, kwargs))
        return "from-pretrained"

    monkeypatch.setattr(NemotronHOmniForCausalLM, "_from_config", classmethod(fake_from_config))
    monkeypatch.setattr(NemotronHPreTrainedModel, "from_pretrained", classmethod(fake_from_pretrained))

    config = _remote_composite_config("nvidia/model", "a" * 40)
    assert NemotronHOmniForCausalLM.from_config(config, trust_remote_code=True) == "from-config"
    assert (
        NemotronHOmniForCausalLM.from_pretrained("nvidia/model", config=config, trust_remote_code=True)
        == "from-pretrained"
    )
    assert config_calls == [(config, {"_prime_trust_remote_code": True})]
    assert pretrained_calls == [
        (
            ("nvidia/model",),
            {
                "config": config,
                "trust_remote_code": True,
                "_prime_trust_remote_code": True,
            },
        )
    ]


@pytest.fixture
def composite_model(monkeypatch):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    monkeypatch.setattr(
        modeling,
        "_load_remote_vision_components",
        lambda config, *, trust_remote_code: (_FakeVisionModel, _FakeVisionProjector),
    )
    return NemotronHOmniForCausalLM(_composite_config(), _prime_trust_remote_code=True)


def test_composite_text_only_forward(composite_model):
    input_ids = torch.tensor([[1, 2, 3]])
    seq_lens = torch.tensor([3])

    output = composite_model(input_ids=input_ids, seq_lens=seq_lens)

    output["logits"].sum().backward()

    assert composite_model.supports_packed_multimodal_training
    assert composite_model.model.vision_model.preprocessor_external
    assert composite_model.model.vision_model.anchor.grad is not None
    torch.testing.assert_close(
        composite_model.model.vision_model.anchor.grad,
        torch.zeros(1),
    )
    assert output["logits"].shape == (1, 3, 32)


def test_composite_minimal_image_forward_without_weights(composite_model):
    input_ids = torch.tensor([[1, 18, 3]])
    inject_prime_lm_head(composite_model)
    output = composite_model(
        input_ids=input_ids,
        pixel_values=torch.full((1, 3, 2, 2), 0.5),
        mm_token_type_ids=torch.tensor([[0, 1, 0]]),
        seq_lens=torch.tensor([3]),
    )

    assert output["logits"].shape == (1, input_ids.shape[1], 32)


def test_trainer_forward_accepts_renderer_nemotron_image_fields(composite_model):
    input_ids = torch.tensor([[1, 18, 3]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    inject_prime_lm_head(composite_model)

    output = trainer_forward(
        composite_model,
        input_ids,
        position_ids,
        seq_lens=torch.tensor([3]),
        mm_kwargs={
            "pixel_values": torch.zeros(1, 3, 2, 2),
            "imgs_sizes": torch.tensor([[2, 2]]),
            "num_tokens": torch.tensor([1]),
            "num_patches": torch.tensor([1]),
        },
        mm_token_type_ids=torch.tensor([[0, 1, 0]]),
    )

    assert output["logits"].shape == (1, input_ids.shape[1], 32)


def test_composite_rejects_renderer_image_markers_that_disagree_with_tokens(composite_model):
    inject_prime_lm_head(composite_model)

    with pytest.raises(ValueError, match="image markers"):
        composite_model(
            input_ids=torch.tensor([[1, 18, 3]]),
            pixel_values=torch.zeros(1, 3, 2, 2),
            imgs_sizes=torch.tensor([[2, 2]]),
            num_tokens=torch.tensor([1]),
            num_patches=torch.tensor([1]),
            mm_token_type_ids=torch.tensor([[0, 0, 0]]),
            seq_lens=torch.tensor([3]),
        )


def test_composite_reconstructs_dynamic_resolution_images_from_flat_wire(composite_model):
    inject_prime_lm_head(composite_model)
    pixels = torch.cat(
        (
            torch.zeros(1, 3, 2, 3).reshape(-1),
            torch.ones(1, 3, 3, 2).reshape(-1),
        )
    )

    output = composite_model(
        input_ids=torch.tensor([[18, 2, 18]]),
        pixel_values=pixels,
        imgs_sizes=torch.tensor([[2, 3], [3, 2]]),
        num_tokens=torch.tensor([1, 1]),
        num_patches=torch.tensor([1, 1]),
        mm_token_type_ids=torch.tensor([[1, 0, 1]]),
        seq_lens=torch.tensor([3]),
    )

    assert output["logits"].shape == (1, 3, 32)


@pytest.mark.parametrize(
    ("pixel_values", "imgs_sizes", "num_tokens", "message"),
    [
        (torch.arange(11, dtype=torch.float32), torch.tensor([[2, 2]]), torch.tensor([1]), "flat pixel_values"),
        (torch.arange(12, dtype=torch.float32), torch.tensor([[0, 2]]), torch.tensor([1]), "positive"),
        (torch.arange(12, dtype=torch.int64), torch.tensor([[2, 2]]), torch.tensor([1]), "floating-point"),
        (torch.arange(12, dtype=torch.float32), torch.tensor([[2, 2]]), torch.tensor([0]), "positive"),
    ],
)
def test_composite_rejects_malformed_flat_image_wire(
    composite_model,
    pixel_values,
    imgs_sizes,
    num_tokens,
    message,
):
    with pytest.raises(ValueError, match=message):
        composite_model(
            input_ids=torch.tensor([[1, 18, 3]]),
            pixel_values=pixel_values,
            imgs_sizes=imgs_sizes,
            num_tokens=num_tokens,
            num_patches=torch.tensor([1]),
            mm_token_type_ids=torch.tensor([[0, 1, 0]]),
            seq_lens=torch.tensor([3]),
        )


def test_composite_merges_images_before_context_parallel_sharding(composite_model, monkeypatch):
    import prime_rl.trainer.models.nemotron_h_omni.modeling_nemotron_h_omni as modeling

    events = []
    original_merge = modeling.merge_image_embeddings

    def record_merge(*args, **kwargs):
        events.append("merge")
        return original_merge(*args, **kwargs)

    def record_shard(tensor, **kwargs):
        events.append("shard")
        return tensor

    monkeypatch.setattr(modeling, "merge_image_embeddings", record_merge)
    monkeypatch.setattr(modeling, "shard_for_cp", record_shard)
    monkeypatch.setattr(modeling, "shard_position_ids_for_cp", lambda position_ids, **kwargs: position_ids)
    monkeypatch.setattr(modeling, "setup_cp_attention_params", lambda *args, **kwargs: None)
    language_model = composite_model.model.language_model
    language_model.context_parallel_group = object()
    language_model.context_parallel_rank = 0
    language_model.context_parallel_world_size = 1

    output = composite_model.model(
        input_ids=torch.tensor([[1, 18, 3]]),
        position_ids=torch.arange(3).unsqueeze(0),
        pixel_values=torch.zeros(1, 3, 2, 2),
        imgs_sizes=torch.tensor([[2, 2]]),
        num_tokens=torch.tensor([1]),
        num_patches=torch.tensor([1]),
        mm_token_type_ids=torch.tensor([[0, 1, 0]]),
        routed_experts=torch.zeros(1, 3, 0, 0, dtype=torch.long),
        seq_lens=torch.tensor([3]),
    )

    assert output.last_hidden_state.shape == (1, 3, 4)
    assert events == ["merge", "shard", "shard"]


def test_composite_handles_checkpoint_missing_radio_state_after_to_empty(
    composite_model,
):
    assert isinstance(composite_model.model.vision_model.layer_scale, nn.Identity)
    composite_model.to_empty(device="cpu")
    composite_model.init_buffers_post_meta()

    torch.testing.assert_close(
        composite_model.model.vision_model.summary_idxs,
        torch.tensor([0]),
    )
