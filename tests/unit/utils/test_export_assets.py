import json

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig


def test_convert_publishes_complete_assets_or_no_output(tmp_path, tiny_run):
    cli, step, state, before = tiny_run
    output = step / "weights"
    assert cli.convert(step) == output
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(state)
    actual = {}
    for filename in set(index["weight_map"].values()):
        actual.update(load_file(output / filename))
    assert actual.keys() == state.keys()
    for name, tensor in state.items():
        assert actual[name].dtype == torch.bfloat16
        torch.testing.assert_close(actual[name], tensor.bfloat16(), rtol=0, atol=0)
    exported_config = json.loads((output / "config.json").read_text())
    assert exported_config["dtype"] == "bfloat16"
    assert "torch_dtype" not in exported_config
    assert "quantization_config" not in exported_config
    assert GenerationConfig.from_pretrained(output).use_cache is True
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1
    assert tokenizer.chat_template == "{{ messages[0]['content'] }}"
    assert not list(step.glob(".weights-*"))
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(output)
    }
    assert after == before
