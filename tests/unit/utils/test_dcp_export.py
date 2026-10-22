import importlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torch.distributed.checkpoint import FileSystemReader
from transformers import PretrainedConfig

from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prime_rl.utils.utils import default_dtype
from prime_rl.utils.weights import convert_state_dict_to_hf, resolve_wire_dtype


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("convert_dcp_to_bf16")


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_layer_export_matches_trainer_conversion(tmp_path, exporter, tied, fp8):
    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        tie_word_embeddings=tied,
        moe_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
    )
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM._from_config(config)
    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.init_weights(config.initializer_range)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    dcp.save({"app": {"model": state, "optimizer": {"unused": torch.ones(1)}}}, checkpoint_id=tmp_path, no_dist=True)
    checkpoint = exporter.Checkpoint(tmp_path, FileSystemReader(tmp_path).read_metadata(), model, None, None)
    expected = convert_state_dict_to_hf(
        model,
        {
            name: tensor.to(resolve_wire_dtype(model.keep_in_fp32_for_weight_transfer, name, torch.bfloat16))
            for name, tensor in state.items()
            if not (tied and name == "lm_head.weight")
        },
    )
    quantize = importlib.import_module("convert_bf16_to_fp8").quantize_state_dict

    def transform(shard):
        return quantize(shard, 2, "cpu")[0] if fp8 else shard

    expected = transform(expected)
    for workers in (1, 2):
        output = tmp_path / f"export-{workers}"
        exporter.write_weights(checkpoint, output, workers, transform)
        index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
        actual = {name: tensor for file in set(index.values()) for name, tensor in load_file(output / file).items()}
        assert actual.keys() == expected.keys() == index.keys()
        for name, tensor in expected.items():
            assert actual[name].dtype == tensor.dtype
            assert actual[name].shape == tensor.shape
            assert torch.equal(actual[name].view(torch.uint8), tensor.view(torch.uint8))


def test_export_config_drops_source_quantization(exporter):
    config = PretrainedConfig()
    config.text_config = PretrainedConfig()
    config.text_config.quantization_config = {"quant_method": "fp8"}
    exporter._export_config(config)
    assert config.dtype == config.text_config.dtype == "bfloat16"
    assert not hasattr(config.text_config, "quantization_config")


def test_export_rejects_multiple_processes(tmp_path, exporter, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one process"):
        exporter.open_checkpoint(tmp_path)
