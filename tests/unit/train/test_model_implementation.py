from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from transformers import Qwen3Config

from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig, TrainerConfig
from prime_rl.trainer.model import get_model, load_dcp_from_hf
from prime_rl.trainer.models.llama.modeling_llama import LlamaForCausalLM
from prime_rl.trainer.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from prime_rl.utils.weights import save_state_dict


class LlamaWithHead(LlamaForCausalLM):
    """An implementation for another architecture than the checkpoint's."""


class Qwen3WithHead(Qwen3ForCausalLM):
    """prime-rl's Qwen3 with an added head, as a model.implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.score_head = nn.Linear(config.hidden_size, 1)


@pytest.fixture
def checkpoint(tmp_path):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.save_pretrained(tmp_path)
    return tmp_path


def model_config(checkpoint, implementation=f"{__name__}.Qwen3WithHead") -> ModelConfig:
    return ModelConfig(name=str(checkpoint), attn="flash_attention_2", implementation=implementation)


@pytest.mark.gpu
def test_get_model_builds_the_implementation(checkpoint):
    model = get_model(model_config(checkpoint), device=torch.device("meta"))
    base = get_model(model_config(checkpoint, implementation=None), device=torch.device("meta"))
    assert type(model) is Qwen3WithHead and type(base) is not Qwen3WithHead
    assert set(model.state_dict()) == set(base.state_dict()) | {"score_head.weight", "score_head.bias"}

    with pytest.raises(TypeError, match="must name a PreTrainedModelPrimeRL subclass"):
        get_model(model_config(checkpoint, implementation="torch.nn.Linear"), device=torch.device("meta"))
    with pytest.raises(TypeError, match="must extend prime-rl's class for 'qwen3'"):
        get_model(model_config(checkpoint, implementation=f"{__name__}.LlamaWithHead"), device=torch.device("meta"))


@pytest.mark.gpu
def test_added_modules_load_from_the_checkpoint(checkpoint, monkeypatch):
    torch.manual_seed(0)
    expected = {
        name: tensor.clone()
        for name, tensor in Qwen3WithHead(Qwen3Config.from_pretrained(checkpoint)).state_dict().items()
    }
    save_state_dict(dict(expected), checkpoint)  # the base weights plus the head's

    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)
    model = get_model(model_config(checkpoint), device=torch.device("meta"))
    load_dcp_from_hf(model, model_config(checkpoint), parallel_dims=MagicMock())

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), expected[name].to(tensor.dtype))


def test_implementation_is_refused_where_it_cannot_be_served():
    model = {"implementation": f"{__name__}.Qwen3WithHead"}
    assert SFTConfig(model=model).model.implementation == model["implementation"]
    assert TrainerConfig(model=model, data={"fake": {}}).model.implementation  # nothing is broadcast
    source = {"env": {"taskset": {"id": "reverse-text"}, "agent": {"harness": {"id": "null"}}}}
    with pytest.raises(ValidationError, match="cannot be served"):
        SFTConfig(model=model, eval={"source": [source]})  # online evals broadcast the weights
    with pytest.raises(ValidationError, match="cannot be served"):
        TrainerConfig(model=model)  # RL broadcasts the weights
    with pytest.raises(ValidationError, match="not supported with model.vlm"):
        ModelConfig(**model, vlm={"vision_encoder_attr": "model.visual", "language_model_attr": "model.language_model"})
