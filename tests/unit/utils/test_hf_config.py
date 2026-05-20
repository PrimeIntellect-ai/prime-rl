from transformers.configuration_utils import PreTrainedConfig

from prime_rl.utils import hf_config
from prime_rl.utils.hf_config import normalize_rope_numeric_types, patch_transformers_config_rope_numeric_types


def test_normalize_yarn_rope_numeric_types_casts_float_fields():
    config = {
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 8,
            "attention_factor": 1,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 8192,
        }
    }

    assert normalize_rope_numeric_types(config)

    rope_scaling = config["rope_scaling"]
    assert rope_scaling["factor"] == 8.0
    assert rope_scaling["attention_factor"] == 1.0
    assert rope_scaling["beta_fast"] == 32.0
    assert rope_scaling["beta_slow"] == 1.0
    assert isinstance(rope_scaling["factor"], float)
    assert isinstance(rope_scaling["attention_factor"], float)
    assert isinstance(rope_scaling["beta_fast"], float)
    assert isinstance(rope_scaling["beta_slow"], float)
    assert isinstance(rope_scaling["original_max_position_embeddings"], int)


def test_normalize_yarn_rope_numeric_types_handles_nested_layer_types():
    config = {
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "factor": 8,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 8192,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 500000,
            },
        }
    }

    assert normalize_rope_numeric_types(config)

    full_attention = config["rope_parameters"]["full_attention"]
    sliding_attention = config["rope_parameters"]["sliding_attention"]
    assert full_attention["factor"] == 8.0
    assert full_attention["beta_fast"] == 32.0
    assert full_attention["beta_slow"] == 1.0
    assert isinstance(full_attention["factor"], float)
    assert isinstance(full_attention["beta_fast"], float)
    assert isinstance(full_attention["beta_slow"], float)
    assert isinstance(sliding_attention["rope_theta"], int)


def test_patch_transformers_config_rope_numeric_types_normalizes_at_ingestion(monkeypatch):
    def fake_get_config_dict(cls, *args, **kwargs):
        return (
            {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 8,
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "original_max_position_embeddings": 8192,
                }
            },
            {},
        )

    monkeypatch.setattr(hf_config, "_PATCHED_TRANSFORMERS_CONFIG", False)
    monkeypatch.setattr(PreTrainedConfig, "get_config_dict", classmethod(fake_get_config_dict))

    patch_transformers_config_rope_numeric_types()

    config_dict, _ = PreTrainedConfig.get_config_dict("unused")
    rope_scaling = config_dict["rope_scaling"]
    assert isinstance(rope_scaling["factor"], float)
    assert isinstance(rope_scaling["beta_fast"], float)
    assert isinstance(rope_scaling["beta_slow"], float)
