import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


@pytest.fixture
def fp8_tool(tiny_run):
    return importlib.import_module("convert_dcp_to_fp8")


def _weights(path: Path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    return {
        name: tensor
        for filename in set(index["weight_map"].values())
        for name, tensor in load_file(path / filename).items()
    }


def test_fp8_scale_matches_cuda_scalar_division_rounding(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    quantize = importlib.import_module("convert_bf16_to_fp8").quantize_to_fp8_blockwise
    weight = torch.tensor([[13.0 * 448.0]])
    _, scales = quantize(weight, block_size=1)
    expected = weight * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    assert torch.equal(scales, expected)
    assert not torch.equal(scales, weight / 448.0)


def test_fp8_export_matches_quantizing_bf16_export(tmp_path, tiny_run, fp8_tool):
    bf16_tool, step, _, before = tiny_run
    output = fp8_tool.convert(step, block_size=2)
    assert output == step / "weights-FP8"
    bf16 = bf16_tool.convert(step)
    reference = importlib.import_module("convert_bf16_to_fp8").convert(bf16, tmp_path / "reference", block_size=2)
    actual_weights, expected_weights = _weights(output), _weights(reference)
    assert actual_weights.keys() == expected_weights.keys()
    assert any(tensor.dtype == torch.float8_e4m3fn for tensor in actual_weights.values())
    for name, tensor in expected_weights.items():
        actual = actual_weights[name]
        assert actual.dtype == tensor.dtype
        assert actual.shape == tensor.shape
        assert torch.equal(actual.view(torch.uint8), tensor.view(torch.uint8))
    actual_assets = {path.name: path.read_bytes() for path in output.iterdir() if path.suffix != ".safetensors"}
    expected_assets = {path.name: path.read_bytes() for path in reference.iterdir() if path.suffix != ".safetensors"}
    assert actual_assets == expected_assets
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "fp8"
    assert config["quantization_config"]["weight_block_size"] == [2, 2]
    assert config["quantization_config"]["modules_to_not_convert"] == ["lm_head", "model.embed_tokens"]
    assert all((tmp_path / relative).read_bytes() == content for relative, content in before.items())


def test_fp8_export_rolls_back_on_quantization_failure(tiny_run, fp8_tool, monkeypatch):
    _, step, _, _ = tiny_run

    def fail_after_assets(tensors, block_size, device):
        assert device == "cpu"
        raise ValueError("quantization failed")

    monkeypatch.setattr(fp8_tool, "quantize_state_dict", fail_after_assets)
    with pytest.raises(ValueError, match="quantization failed"):
        fp8_tool.convert(step)
    assert not (step / "weights-FP8").exists()
    assert not list(step.glob(".weights-FP8-*"))
