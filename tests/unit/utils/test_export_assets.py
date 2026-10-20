import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils.dcp_export import iter_cpu_hf_groups
from prime_rl.utils.export_assets import prepare_export_config
from prime_rl.utils.export_conversion import conversion_chain, resolve_config
from prime_rl.utils.utils import default_dtype


@pytest.mark.parametrize("fail_writer", [False, True], ids=["publish", "writer-failure"])
def test_convert_publishes_complete_assets_or_no_output(tmp_path, tiny_run, monkeypatch, fail_writer):
    cli, step, state, before = tiny_run
    output = step / "weights"
    if fail_writer:

        def fail_after_assets(groups, staging):
            assert (staging / "config.json").exists()
            assert (staging / "tokenizer.json").exists()
            (staging / "partial.safetensors").write_bytes(b"incomplete")
            raise OSError("injected writer failure")

        monkeypatch.setattr(cli, "write_hf_shards", fail_after_assets)
        with pytest.raises(OSError, match="injected writer failure"):
            cli.convert(step)
        assert not output.exists()
    else:
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


def test_debug_shortened_exports_rebuild_first_n_layouts(tmp_path):
    from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    text = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
    )
    prepared = prepare_export_config(text.to_dict(), {"debug": {"num_layers": 2}})
    assert prepared["num_hidden_layers"] == 2
    assert len(prepared["layer_types"]) == 2

    # Shortened configs must rebuild through their own strict validation.
    deepseek = prepare_export_config(
        {"model_type": "deepseek_v4", "num_hidden_layers": 4, "num_hash_layers": 3},
        {"debug": {"num_layers": 2}},
    )
    assert deepseek["num_hidden_layers"] == 2
    assert deepseek["num_hash_layers"] == 2
    assert len(deepseek["layer_types"]) == 2
    resolve_config(deepseek)
    laguna = prepare_export_config(
        {
            "model_type": "laguna",
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_attention_heads_per_layer": [2, 2, 4, 2],
        },
        {"debug": {"num_layers": 2}},
    )
    assert laguna["num_hidden_layers"] == 2
    assert laguna["num_attention_heads_per_layer"] == [2, 2]
    resolve_config(laguna)

    shortened = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
    )
    torch.manual_seed(0)
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = Qwen3_5ForCausalLM._from_config(shortened)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}}, checkpoint_id=checkpoint, no_dist=True)
    expected = apply_prime_to_hf(dict(state), conversion_chain(prepared))
    output = {name: tensor for group in iter_cpu_hf_groups(checkpoint, prepared) for name, tensor in group.items()}
    assert output.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(output[name], tensor, rtol=0, atol=0)
