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


class Qwen3WithInitializedHead(Qwen3WithHead):
    """The head starts from two rows of the loaded LM head."""

    def init_missing_weights(self, names):
        rows = self.lm_head.weight
        return {"score_head.weight": rows[1:2] - rows[0:1], "score_head.bias": torch.zeros(1)}


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
    implementation = None if implementation is None else {"import_path": implementation}
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
    load_dcp_from_hf(model, model_config(checkpoint))

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), expected[name].to(tensor.dtype))


@pytest.mark.gpu
def test_added_modules_the_checkpoint_lacks_are_initialized(checkpoint, monkeypatch):
    torch.manual_seed(0)
    base = {
        name: tensor.clone()
        for name, tensor in Qwen3ForCausalLM(Qwen3Config.from_pretrained(checkpoint)).state_dict().items()
    }
    save_state_dict(dict(base), checkpoint)  # the base weights only

    monkeypatch.setattr("torch.distributed.barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.wait_for_master", lambda *args, **kwargs: None)
    monkeypatch.setattr("prime_rl.trainer.model.get_dcp_group", lambda: None)

    def load(implementation):
        config = model_config(checkpoint, implementation=implementation)
        model = get_model(config, device=torch.device("meta"))
        load_dcp_from_hf(model, config)
        return model

    model = load(f"{__name__}.Qwen3WithInitializedHead")
    rows = base["lm_head.weight"].to(model.lm_head.weight.dtype)  # the head is computed from the loaded rows
    expected = base | {"score_head.weight": rows[1:2] - rows[0:1], "score_head.bias": torch.zeros(1)}
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), expected[name].to(tensor.dtype))

    lacks = r"The checkpoint lacks \['score_head.bias', 'score_head.weight'\] and"
    with pytest.raises(RuntimeError, match=rf"{lacks} \S+\.Qwen3WithHead.init_missing_weights returned \[\]"):
        load(f"{__name__}.Qwen3WithHead")  # the class does not define the method
    monkeypatch.setattr(Qwen3WithInitializedHead, "init_missing_weights", lambda self, names: {"head.weight": None})
    with pytest.raises(RuntimeError, match=rf"{lacks} \S+ returned \['head.weight'\]"):
        load(f"{__name__}.Qwen3WithInitializedHead")
    monkeypatch.setattr(
        Qwen3WithInitializedHead, "init_missing_weights", lambda self, names: dict.fromkeys(names, torch.zeros(1))
    )
    with pytest.raises(ValueError, match=r"returned score_head.weight with shape \(1,\), the parameter has \(1, 32\)"):
        load(f"{__name__}.Qwen3WithInitializedHead")


def test_a_mistyped_implementation_names_the_setting(checkpoint):
    with pytest.raises(ImportError, match="model.implementation 'no_such_module.Model' cannot be imported"):
        get_model(model_config(checkpoint, implementation="no_such_module.Model"), device=torch.device("meta"))
    with pytest.raises(ImportError, match="model.implementation 'Model' cannot be imported"):
        get_model(model_config(checkpoint, implementation="Model"), device=torch.device("meta"))  # no module part


def test_implementation_is_refused_where_it_cannot_be_served():
    model = {"implementation": {"import_path": f"{__name__}.Qwen3WithHead"}}
    assert SFTConfig(model=model).model.implementation.import_path == f"{__name__}.Qwen3WithHead"
    assert TrainerConfig(model=model, data={"fake": {}}).model.implementation  # nothing is broadcast
    source = {"env": {"taskset": {"id": "reverse-text"}, "agent": {"harness": {"id": "null"}}}}
    with pytest.raises(ValidationError, match="cannot be served"):
        SFTConfig(model=model, eval={"source": [source]})  # online evals broadcast the weights
    with pytest.raises(ValidationError, match="cannot be served"):
        TrainerConfig(model=model)  # RL broadcasts the weights
    with pytest.raises(ValidationError, match="not supported with model.vlm"):
        ModelConfig(**model, vlm={})
    with pytest.raises(ValidationError, match="takes no kwargs"):
        ModelConfig(implementation={"import_path": f"{__name__}.Qwen3WithHead", "kwargs": {"num_options": 4}})
