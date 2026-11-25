"""Qwen3_5ForSlotDecision: the ``modeling_overrides`` key of a stock Qwen3.5 config.json selects it, as in vLLM main, and
it refuses the overrides vLLM refuses to serve. On the tiny config of clef_flash_release.py."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.ckpt import setup_model_files
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.qwen3_5 import ClefFlashForJointSchema, Qwen3_5ForCausalLM, Qwen3_5ForSlotDecision
from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5_slot import get_n_slots
from tests.unit.train.models.qwen3_5.clef_flash_release import JOINT_HEAD_CONFIG, tiny_config, tokenizer

SERVED = {"attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"}, "readout": "slot_lm"}
# The modeling_overrides of baselabs/mammoth-1
OVERRIDES = {**SERVED, "head_dim": 256, "n_slots": 256}


def write_config(directory: Path, **keys) -> ModelConfig:
    """A checkpoint of only a tiny stock Qwen3.5 config.json with ``keys`` set, which is all that building reads."""
    directory.mkdir(parents=True, exist_ok=True)
    tiny_config().save_pretrained(directory)
    path = directory / "config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | keys))
    return ModelConfig(name=str(directory), attn="flash_attention_2")


def test_modeling_overrides_select_the_class(tmp_path):
    config = write_config(tmp_path / "slot", modeling_overrides=OVERRIDES)
    model = get_model(config)
    assert type(model) is Qwen3_5ForSlotDecision
    assert model.config.modeling_overrides == OVERRIDES and get_n_slots(model.config) == 256
    # No rename and no tensors of its own: the checkpoint stays a stock Qwen3.5 one
    assert model.config.architectures == ["Qwen3_5ForConditionalGeneration"]
    stock = get_model(write_config(tmp_path / "stock"))
    assert type(stock) is Qwen3_5ForCausalLM and model.state_dict().keys() == stock.state_dict().keys()

    files = setup_model_files(config, tokenizer(), None)
    assert files.selection == (Qwen3_5ForSlotDecision, OVERRIDES) and files.config.modeling_overrides == OVERRIDES


def test_only_a_stock_config_with_the_key_selects_the_class(tmp_path):
    # A null key selects nothing, as in vLLM
    assert type(get_model(write_config(tmp_path / "null", modeling_overrides=None))) is Qwen3_5ForCausalLM
    other = write_config(tmp_path / "other", modeling_overrides=OVERRIDES, architectures=["Qwen3_5ForCausalLM"])
    assert type(get_model(other)) is Qwen3_5ForCausalLM


def test_a_joint_head_config_file_selects_clef_flash_first(tmp_path):
    """As in vLLM, which looks for joint_head_config.json before the key."""
    config = write_config(tmp_path, modeling_overrides=OVERRIDES)
    (tmp_path / "joint_head_config.json").write_text(json.dumps(JOINT_HEAD_CONFIG))
    model = get_model(config)
    assert type(model) is ClefFlashForJointSchema and model.config.joint_head_config == JOINT_HEAD_CONFIG


def test_n_slots_default_to_256():
    assert get_n_slots(SimpleNamespace(modeling_overrides=SERVED)) == 256
    assert get_n_slots(SimpleNamespace(modeling_overrides={**SERVED, "head_dim": 8, "n_slots": 1})) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        "slot_lm",
        {**OVERRIDES, "attention": {"full_attention": "causal"}},
        {**OVERRIDES, "attention": {"full_attention": "block_bidirectional", "linear_attention": "bidirectional"}},
        {**OVERRIDES, "readout": "head"},
        {key: value for key, value in OVERRIDES.items() if key != "readout"},
        {**OVERRIDES, "pooling": "last"},
        {**OVERRIDES, "n_slots": 0},
        {**OVERRIDES, "n_slots": 257},
        {**OVERRIDES, "n_slots": True},
        {**OVERRIDES, "n_slots": 8.0},
    ],
)
def test_overrides_vllm_refuses_to_serve_are_refused(tmp_path, overrides):
    config = write_config(tmp_path, modeling_overrides=overrides)
    with pytest.raises(ValueError, match=re.escape(f"Unsupported modeling_overrides {overrides!r}: vLLM serves")):
        get_model(config)
