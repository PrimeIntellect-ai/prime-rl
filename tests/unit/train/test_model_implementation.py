from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model, load_dcp_from_hf
from prime_rl.trainer.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from prime_rl.utils.weights import save_state_dict

pytestmark = [pytest.mark.gpu]


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


def test_get_model_builds_the_implementation(checkpoint):
    model = get_model(model_config(checkpoint), device=torch.device("meta"))
    base = get_model(model_config(checkpoint, implementation=None), device=torch.device("meta"))
    assert type(model) is Qwen3WithHead and type(base) is not Qwen3WithHead
    assert set(model.state_dict()) == set(base.state_dict()) | {"score_head.weight", "score_head.bias"}

    with pytest.raises(TypeError, match="must name a PreTrainedModelPrimeRL subclass"):
        get_model(model_config(checkpoint, implementation="torch.nn.Linear"), device=torch.device("meta"))


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
