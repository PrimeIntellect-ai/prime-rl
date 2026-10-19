import pytest

from prime_rl.utils.hf_export_schema import (
    expected_hf_schema,
    hf_schema_metadata,
    is_hf_identity_layout,
    matches_hf_schema,
)


@pytest.fixture
def gpt2_config():
    return {
        "model_type": "gpt2",
        "vocab_size": 32,
        "n_positions": 16,
        "n_embd": 8,
        "n_layer": 1,
        "n_head": 2,
        "tie_word_embeddings": True,
    }


def test_native_gpt2_identity_schema(gpt2_config):
    schema = expected_hf_schema(gpt2_config)
    assert schema["transformer.wte.weight"] == (32, 8)
    assert schema["transformer.h.0.attn.c_attn.weight"] == (8, 24)
    assert schema["lm_head.weight"] == (32, 8)
    assert is_hf_identity_layout(schema, gpt2_config)


def test_native_schema_rejects_missing_extra_and_wrong_shapes(gpt2_config):
    schema, aliases = hf_schema_metadata(gpt2_config)
    missing = dict(schema)
    missing.pop("transformer.h.0.attn.c_attn.weight")
    assert not matches_hf_schema(missing, schema, aliases)
    assert not matches_hf_schema({**schema, "unknown.weight": (8, 8)}, schema, aliases)
    assert not matches_hf_schema({**schema, "transformer.wte.weight": (8, 32)}, schema, aliases)


def test_tied_output_alias_can_be_absent(gpt2_config):
    schema, aliases = hf_schema_metadata(gpt2_config)
    assert aliases["lm_head.weight"] == "transformer.wte.weight"
    checkpoint = dict(schema)
    checkpoint.pop("lm_head.weight")
    assert matches_hf_schema(checkpoint, schema, aliases)
    checkpoint.pop("transformer.wte.weight")
    assert not matches_hf_schema(checkpoint, schema, aliases)
    untied, untied_aliases = hf_schema_metadata({**gpt2_config, "tie_word_embeddings": False})
    assert "lm_head.weight" not in untied_aliases
    assert not matches_hf_schema({k: v for k, v in untied.items() if k != "lm_head.weight"}, untied, untied_aliases)


def test_unknown_remote_architecture_is_rejected():
    with pytest.raises(ValueError, match="No native Transformers config"):
        expected_hf_schema({"model_type": "private_remote_model", "auto_map": {"AutoConfig": "custom.Config"}})


def test_explicit_trust_loads_local_custom_code_without_weights(tmp_path, gpt2_config):
    (tmp_path / "configuration_custom.py").write_text(
        "from transformers import GPT2Config\nclass CustomConfig(GPT2Config):\n    model_type = 'local_custom_gpt'\n"
    )
    (tmp_path / "modeling_custom.py").write_text(
        "from transformers import GPT2LMHeadModel\n"
        "from .configuration_custom import CustomConfig\n"
        "class CustomModel(GPT2LMHeadModel):\n"
        "    config_class = CustomConfig\n"
    )
    config = {
        **gpt2_config,
        "model_type": "local_custom_gpt",
        "auto_map": {
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "modeling_custom.CustomModel",
        },
    }
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        expected_hf_schema(config, {"name": str(tmp_path), "trust_remote_code": False})
    schema, aliases = hf_schema_metadata(config, {"name": str(tmp_path), "trust_remote_code": True})
    assert schema["transformer.wte.weight"] == (32, 8)
    assert aliases["lm_head.weight"] == "transformer.wte.weight"
    assert not list(tmp_path.glob("*.safetensors"))


def test_native_constructor_errors_are_not_schema_unavailable(gpt2_config):
    from prime_rl.utils.hf_export_schema import ModelSchemaUnavailable

    with pytest.raises(ValueError, match="divisible") as caught:
        expected_hf_schema({**gpt2_config, "n_embd": 7})
    assert not isinstance(caught.value, ModelSchemaUnavailable)


def test_schema_details_records_expected_state_dtypes(gpt2_config):
    import torch

    from prime_rl.utils.hf_export_schema import hf_schema_details

    schema, aliases, dtypes = hf_schema_details(gpt2_config)
    assert dtypes.keys() == schema.keys()
    assert dtypes["transformer.wte.weight"].is_floating_point
    assert dtypes["lm_head.weight"] == dtypes["transformer.wte.weight"]
    assert all(isinstance(dtype, torch.dtype) for dtype in dtypes.values())
    assert aliases["lm_head.weight"] == "transformer.wte.weight"


def test_native_schema_ignores_trainer_config_registration():
    from transformers import AutoConfig
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    from prime_rl.trainer.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as PrimeQwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
    ).to_dict()
    assert isinstance(AutoConfig.for_model("qwen3_moe"), PrimeQwen3MoeConfig)
    schema = expected_hf_schema(config)
    assert schema["model.layers.0.mlp.experts.gate_up_proj"] == (2, 8, 8)
    assert is_hf_identity_layout(schema, config)


def test_native_gpt_sw3_alias_schema(gpt2_config):
    schema = expected_hf_schema({**gpt2_config, "model_type": "gpt-sw3"})
    assert schema["transformer.wte.weight"] == (32, 8)
    assert schema["transformer.h.0.attn.c_attn.weight"] == (8, 24)
    assert schema["lm_head.weight"] == (32, 8)
    assert is_hf_identity_layout(schema, {**gpt2_config, "model_type": "gpt-sw3"})
