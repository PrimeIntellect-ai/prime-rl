"""Tiny classes on prime-rl's Qwen3 that an extra config file next to a stock Qwen3 config.json selects."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import PreTrainedTokenizerFast, Qwen3Config

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer import models
from prime_rl.trainer.model import load_dcp_from_hf
from prime_rl.trainer.models import check_selected_by_extra_config
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM

EXTRA_CONFIG_FILE = "stub_head_config.json"
EXTRA_CONFIG_ATTR = "stub_head_config"
EXTRA_CONFIG = {"num_outputs": 3}


def tiny_config(tied: bool = False) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        tie_word_embeddings=tied,
        architectures=["Qwen3ForCausalLM"],
    )


class HeadStub(Qwen3ForCausalLM):
    """Qwen3 with a linear head of ``num_outputs`` outputs, which the export writes to a file of its own."""

    extra_config_file = EXTRA_CONFIG_FILE
    extra_config_attr = EXTRA_CONFIG_ATTR
    stock_model_type = "qwen3"
    stock_architecture = "Qwen3ForCausalLM"
    export_files = {"head.": "head.safetensors"}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.head = nn.Linear(config.hidden_size, getattr(config, EXTRA_CONFIG_ATTR)["num_outputs"], bias=False)


class ConvertingStub(HeadStub):
    """The stub for a release that ships no LM head: loading converts it."""

    @classmethod
    def is_hf_state_dict(cls, state_dict):
        return "lm_head.weight" not in state_dict

    @classmethod
    def is_prime_state_dict(cls, state_dict):
        return "lm_head.weight" in state_dict

    def convert_to_prime(self, state_dict):
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
        return state_dict


def register(monkeypatch, *classes: type[HeadStub]) -> None:
    """Make ``classes`` the classes selected by an extra config file."""
    check_selected_by_extra_config(classes)
    monkeypatch.setattr(models, "_SELECTED_BY_EXTRA_CONFIG", classes)


def stub_config(tied: bool = False) -> Qwen3Config:
    """The config get_model builds the stub from: the stock config with the extra config file's content on it."""
    config = tiny_config(tied)
    setattr(config, EXTRA_CONFIG_ATTR, dict(EXTRA_CONFIG))
    return config


def write_tokenizer(directory: Path) -> None:
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "a": 1}, unk_token="<eos>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>").save_pretrained(directory)


def write_stub(
    directory: Path, extra_config: dict | list | None = EXTRA_CONFIG, tied: bool = False, weights: bool = False
) -> dict[str, torch.Tensor] | None:
    """A release of the stub: the stock config.json, the extra config file (unless ``extra_config`` is None) and a
    tokenizer. With ``weights``, also a model.safetensors of random weights of the stub, which it returns."""
    directory.mkdir(parents=True, exist_ok=True)
    tiny_config(tied).save_pretrained(directory)
    if extra_config is not None:
        (directory / EXTRA_CONFIG_FILE).write_text(json.dumps(extra_config))
    write_tokenizer(directory)
    if not weights:
        return None
    torch.manual_seed(0)
    model = HeadStub(stub_config(tied))
    state = {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()}
    if tied:
        del state["lm_head.weight"]
    save_file(state, directory / "model.safetensors", metadata={"format": "pt"})
    return state


def write_release_layout(directory: Path) -> dict[str, torch.Tensor]:
    """The stub as ``ConvertingStub``'s release ships it: no LM head, and the head in its export file, without the
    prefix. Returns the stub's weights."""
    state = write_stub(directory, weights=True)
    weights = {name: tensor for name, tensor in state.items() if name not in ("lm_head.weight", "head.weight")}
    save_file(weights, directory / "model.safetensors", metadata={"format": "pt"})
    save_file({"weight": state["head.weight"]}, directory / "head.safetensors", metadata={"format": "pt"})
    return state


def load_like_the_trainer(model: nn.Module, name: Path, monkeypatch) -> None:
    """Load checkpoint ``name`` into the meta ``model`` with the trainer's loader, on CPU in one process."""
    monkeypatch.setattr(model, "to_empty", lambda device: nn.Module.to_empty(model, device="cpu"))
    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)
    load_dcp_from_hf(model, ModelConfig(name=str(name)))
