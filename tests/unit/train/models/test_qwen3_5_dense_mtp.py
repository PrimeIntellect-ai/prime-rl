import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5_dense_mtp import patch_qwen3_5_dense_mtp
from prime_rl.trainer.mtp import roll_tensor
from prime_rl.trainer.rl.broadcast.nccl import preprocess_layer_checkpoint
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def _tiny_config(mtp_enabled: bool = False) -> Qwen3_5Config:
    patch_qwen3_5_dense_mtp()
    text_config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,
        use_cache=False,
    )
    text_config._attn_implementation = "sdpa"
    text_config.mtp_num_hidden_layers = 1
    text_config.prime_mtp_enabled = mtp_enabled
    text_config.prime_mtp_loss_scale = 0.2
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=4,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
    )
    config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    config.use_cache = False
    config._attn_implementation = "sdpa"
    return config


def _tiny_model(mtp_enabled: bool = False) -> Qwen3_5ForConditionalGeneration:
    with torch.device("cuda"), default_dtype(torch.float32):
        model = Qwen3_5ForConditionalGeneration(_tiny_config(mtp_enabled))
    inject_prime_lm_head(model, chunk_size=None)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    return model


def test_qwen3_5_dense_mtp_disabled_by_default():
    model = _tiny_model(mtp_enabled=False)
    input_ids = torch.randint(1, model.config.text_config.vocab_size, (1, 8), device="cuda")
    labels = roll_tensor(input_ids)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

    out = model(input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask, use_cache=False)

    assert "mtp_loss" not in out
    assert not any(key.startswith("mtp.") for key in model.state_dict())


def test_qwen3_5_dense_mtp_registers_hf_checkpoint_keys():
    model = _tiny_model(mtp_enabled=True)
    mtp_keys = {key for key in model.state_dict() if key.startswith("mtp.")}

    assert "mtp.pre_fc_norm_embedding.weight" in mtp_keys
    assert "mtp.pre_fc_norm_hidden.weight" in mtp_keys
    assert "mtp.fc.weight" in mtp_keys
    assert "mtp.layers.0.self_attn.q_proj.weight" in mtp_keys
    assert "mtp.layers.0.self_attn.k_norm.weight" in mtp_keys
    assert "mtp.norm.weight" in mtp_keys
    assert not any(key.startswith("mtp_layers.") for key in model.state_dict())


def test_qwen3_5_dense_mtp_nccl_preprocess_keeps_hf_keys():
    model = _tiny_model(mtp_enabled=True)
    non_layer_state_dict = {
        key: value for key, value in model.state_dict().items() if not key.startswith("model.language_model.layers.")
    }

    converted = preprocess_layer_checkpoint(model, non_layer_state_dict, layer_idx=-1)

    assert "mtp.fc.weight" in converted
    assert "mtp.layers.0.self_attn.q_proj.weight" in converted
    assert not any(key.startswith("mtp_layers.") for key in converted)


def test_qwen3_5_dense_mtp_loss_only_updates_mtp_params():
    model = _tiny_model(mtp_enabled=True)
    input_ids = torch.randint(1, model.config.text_config.vocab_size, (1, 8), device="cuda")
    labels = roll_tensor(input_ids)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

    out = model(input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask, use_cache=False)
    assert out["mtp_loss"].isfinite()
    out["mtp_loss"].backward()

    assert model.model.language_model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert all(param.grad is None for param in model.model.language_model.layers.parameters())
    assert any(param.grad is not None for param in model.mtp.parameters())
