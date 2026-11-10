import json
from types import SimpleNamespace

import pytest
from transformers import Qwen3Config

import prime_rl.trainer.ckpt as ckpt
from prime_rl.trainer.ckpt import check_model_config, write_model_config

FIELDS = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}


def test_a_checkpoint_with_another_architecture_does_not_load(tmp_path, monkeypatch):
    config = Qwen3Config(**FIELDS)
    # A checkpoint without config.json loads after a warning.
    warnings = []
    monkeypatch.setattr(ckpt, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    check_model_config(config, tmp_path)
    assert len(warnings) == 1 and "no config.json" in warnings[0]
    write_model_config(config, tmp_path)
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match="num_hidden_layers 2 -> 3"):
        check_model_config(Qwen3Config(**{**FIELDS, "num_hidden_layers": 3}), tmp_path)

    # A field the checkpoint's config.json lacks has the class default there.
    saved = json.loads((tmp_path / "config.json").read_text())
    del saved["attention_bias"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    check_model_config(config, tmp_path)
    with pytest.raises(ValueError, match=r"attention_bias \(missing\) -> True"):
        check_model_config(Qwen3Config(**FIELDS, attention_bias=True), tmp_path)

    saved["layer_types"][1] = "sliding_attention"
    (tmp_path / "config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=r"has another architecture: layer_types \(differs from index 1\)$"):
        check_model_config(config, tmp_path)
