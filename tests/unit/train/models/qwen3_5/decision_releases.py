"""Tiny d3 and Clef-Flash releases, laid out as the real ones are, for the decision model tests."""

import json
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_5 import configuration_qwen3_5 as hf_configuration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_modeling

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from prime_rl.trainer.models.qwen3_5.clef_joint_schema import JointSchemaHead
from prime_rl.trainer.models.qwen3_5.modeling_clef_flash import SCALARS

HIDDEN = 64
CODES = ["A", "B", "C", "D", "E"]
# d3-nano's decision_config.json, with 5 of its 255 codes and another temperature.
DECISION_CONFIG = {
    "format_version": 1,
    "format_id": "d3-code-readout-v1",
    "prompt": "d3",
    "codes": CODES,
    "token_ids": [32, 33, 34, 35, 36],
    "temperature": 1.5,
    "attention_mode": "noncausal_full_attention",
    "pooling": "last",
    "max_length": None,
    "readout_dtype": "float32",
}
JOINT_HEAD_CONFIG = {
    "hidden_size": HIDDEN,
    "width": 32,
    "routing_layers": 1,
    "layers": 1,
    "heads": 4,
    "feedforward": 48,
}


def vllm_route(directory: Path) -> str:
    """The class vLLM 0.33 serves a Qwen3.5 checkpoint with (transformers_utils/model_arch_config_convertor.py:629-650)."""
    architectures = json.loads((directory / "config.json").read_text())["architectures"]
    if architectures == ["Qwen3_5ForConditionalGeneration"] and (directory / "joint_head_config.json").exists():
        return "ClefFlashForJointSchema"
    if architectures == ["Qwen3_5Model"] and (directory / "decision_config.json").exists():
        return "D3ForSequenceClassification"
    return architectures[0]


def tiny_config(architecture: str, tied: bool) -> Qwen3_5Config:
    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=tied,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=HIDDEN
    )
    return Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
        tie_word_embeddings=tied,
        architectures=[architecture],
    )


def release_backbone(config: Qwen3_5Config) -> dict[str, torch.Tensor]:
    """A release's backbone, under the names of transformers' class for its architecture."""
    torch.manual_seed(0)
    model = getattr(hf_modeling, config.architectures[0])(hf_configuration.Qwen3_5Config(**config.to_dict()))
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def write_tokenizer(directory: Path) -> None:
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "a": 1}, unk_token="<eos>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>").save_pretrained(directory)


def write_d3_release(
    directory: Path, decision_config: dict = DECISION_CONFIG, tied: bool = True
) -> dict[str, torch.Tensor]:
    """d3-nano's layout: config.json with the bare backbone's architecture and a tied embedding, decision_config.json,
    the backbone in one model.safetensors without an index or MTP weights, and readout.safetensors {"weight": F32}.
    Untied, as d3 and d3-flash are, the backbone has no LM head either."""
    config = tiny_config("Qwen3_5Model", tied=tied)
    config.save_pretrained(directory)
    (directory / "decision_config.json").write_text(json.dumps(decision_config))
    weights = release_backbone(config)
    save_file(weights, directory / "model.safetensors")
    readout = {"weight": torch.randn(len(decision_config["codes"]), HIDDEN)}
    save_file(readout, directory / "readout.safetensors")
    write_tokenizer(directory)
    return weights | readout


def clef_head_weights() -> dict[str, torch.Tensor]:
    """A head as joint_head.safetensors stores it: vLLM's names without a prefix, 0-dimensional scalars."""
    torch.manual_seed(1)
    head = {name: tensor.detach().clone() for name, tensor in JointSchemaHead(**JOINT_HEAD_CONFIG).state_dict().items()}
    for scalar in SCALARS:
        head[scalar] = torch.randn(())
    return head


def write_clef_release(directory: Path) -> dict[str, torch.Tensor]:
    """Clef-Flash's layout: the Qwen3.5 checkpoint with an untied LM head, its MTP weights and an index, plus
    joint_head_config.json and joint_head.safetensors."""
    config = tiny_config("Qwen3_5ForConditionalGeneration", tied=False)
    config.save_pretrained(directory)
    (directory / "joint_head_config.json").write_text(json.dumps(JOINT_HEAD_CONFIG))
    backbone = release_backbone(config) | {"mtp.fc.weight": torch.randn(HIDDEN, 2 * HIDDEN)}
    save_file(backbone, directory / "model-00001-of-00001.safetensors")
    index = {"metadata": {}, "weight_map": dict.fromkeys(backbone, "model-00001-of-00001.safetensors")}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    head = clef_head_weights()
    save_file(head, directory / "joint_head.safetensors")
    write_tokenizer(directory)
    return backbone | head


WRITE_RELEASE = {"d3": write_d3_release, "d3-untied": partial(write_d3_release, tied=False), "clef": write_clef_release}


def model_config(directory: Path, implementation: str | None = None) -> ModelConfig:
    implementation = None if implementation is None else {"import_path": implementation}
    return ModelConfig(name=str(directory), attn="flash_attention_2", implementation=implementation)
