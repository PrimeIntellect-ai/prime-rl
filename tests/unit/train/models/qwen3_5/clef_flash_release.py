"""A tiny Clef-Flash release, laid out as Cloudflare/clef-flash is, for the Clef-Flash tests."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_5 import configuration_qwen3_5 as hf_configuration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_modeling

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from prime_rl.trainer.models.qwen3_5.clef_joint_schema import JointSchemaHead
from prime_rl.trainer.models.qwen3_5.modeling_clef_flash import SCALARS, ClefFlashForJointSchema
from tests.unit.train.models.extra_config_stub import load_like_the_trainer

HIDDEN = 64
VOCAB = 288
JOINT_HEAD_CONFIG = {
    "hidden_size": HIDDEN,
    "width": 32,
    "routing_layers": 2,
    "layers": 2,
    "heads": 4,
    "feedforward": 48,
}


def tokenizer(merges: tuple[tuple[str, str], ...] = ()) -> PreTrainedTokenizerFast:
    """One token per byte, plus one per merge of two tokens and Qwen's chat tokens."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate([*alphabet, *(left + right for left, right in merges)])}
    model = Tokenizer(models.BPE(vocab=vocab, merges=list(merges)))
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    model.decoder = decoders.ByteLevel()
    tokens = PreTrainedTokenizerFast(
        tokenizer_object=model,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokens.add_tokens(["<think>", "</think>"])
    return tokens


def tiny_config() -> Qwen3_5Config:
    text_config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=0,
        tie_word_embeddings=False,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=HIDDEN
    )
    return Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=280,
        video_token_id=281,
        vision_start_token_id=282,
        vision_end_token_id=283,
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )


def head_weights() -> dict[str, torch.Tensor]:
    """A head as joint_head.safetensors stores it: vLLM's names without a prefix, 0-dimensional scalars, bf16."""
    torch.manual_seed(1)
    head = {name: tensor.detach().clone() for name, tensor in JointSchemaHead(**JOINT_HEAD_CONFIG).state_dict().items()}
    for scalar in SCALARS:
        head[scalar] = torch.randn(())
    return {name: tensor.to(torch.bfloat16) for name, tensor in head.items()}


def write_backbone(directory: Path) -> dict[str, torch.Tensor]:
    """A stock Qwen3.5 config.json with an untied LM head, and the backbone (language model, vision encoder and LM
    head, no MTP weights) in shards with an index. Returns the weights as stored."""
    directory.mkdir(parents=True, exist_ok=True)
    config = tiny_config()
    config.save_pretrained(directory)
    torch.manual_seed(0)
    model = hf_modeling.Qwen3_5ForConditionalGeneration(hf_configuration.Qwen3_5Config(**config.to_dict()))
    backbone = {name: tensor.detach().to(torch.bfloat16).clone() for name, tensor in model.state_dict().items()}
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for name, tensor in backbone.items():
        shards[f"model-0000{1 if name.startswith('model.visual.') else 2}-of-00002.safetensors"][name] = tensor
    for file, tensors in shards.items():
        save_file(tensors, directory / file, metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {name: file for file, tensors in shards.items() for name in tensors}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    return backbone


def write_release(directory: Path) -> dict[str, torch.Tensor]:
    """Cloudflare/clef-flash's layout: the backbone of a stock Qwen3.5, joint_head_config.json and
    joint_head.safetensors. Returns the weights as stored."""
    backbone = write_backbone(directory)
    (directory / "joint_head_config.json").write_text(json.dumps(JOINT_HEAD_CONFIG))
    head = head_weights()
    save_file(head, directory / "joint_head.safetensors", metadata={"format": "pt"})
    tokenizer().save_pretrained(directory)
    return backbone | head


def clef_config(tied: bool = False, **head):
    """The config the class is built from: the stock config with joint_head_config.json's content, and ``head``."""
    config = tiny_config()
    config._attn_implementation = "flash_attention_2"
    config.tie_word_embeddings = tied
    config.joint_head_config = {**JOINT_HEAD_CONFIG, **head}
    return config


def meta_model(directory: Path) -> ClefFlashForJointSchema:
    """The release in ``directory`` built on the meta device, as the trainer builds it."""
    return get_model(ModelConfig(name=str(directory), attn="flash_attention_2"))


def load_release(directory: Path, monkeypatch) -> ClefFlashForJointSchema:
    """The release in ``directory`` loaded on the CPU by the trainer's loader, which converts it on the way."""
    model = meta_model(directory)
    load_like_the_trainer(model, directory, monkeypatch)
    return model
