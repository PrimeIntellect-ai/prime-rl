"""Compare legacy conversion goldens and load the exported checkpoints in native HF."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "dcp_export"
FAMILY_MODEL_TYPES = {"llama_dense": "llama", "qwen3_moe": "qwen3_moe"}
HF_LOADERS = {"llama_dense": (LlamaConfig, LlamaForCausalLM), "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM)}


def _load_golden(family):
    return json.loads((GOLDEN_DIR / f"{family}_legacy_export_golden.json").read_text())


def _restore_tensors(section):
    tensors = {}
    for name, entry in section.items():
        raw = bytes.fromhex(entry["hex"])
        tensors[name] = (
            torch.frombuffer(bytearray(raw), dtype=getattr(torch, entry["dtype"])).reshape(entry["shape"]).clone()
        )
    return tensors


def _raw_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _export_prime_state(tmp_path, state, config):
    checkpoint = tmp_path / "checkpoint"
    dcp.save(
        {"app": {"model": {name: tensor.clone() for name, tensor in state.items()}}},
        checkpoint_id=checkpoint,
        no_dist=True,
    )
    output = tmp_path / "hf"
    write_hf_shards(iter_cpu_hf_groups(checkpoint, config), output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    exported = {}
    for filename in sorted(set(index["weight_map"].values())):
        exported.update(load_file(output / filename))
    return output, index, exported


@pytest.mark.parametrize("family", sorted(FAMILY_MODEL_TYPES))
def test_cpu_export_matches_legacy_golden_and_loads_cleanly_in_hf(tmp_path, family):
    golden = _load_golden(family)
    source = _restore_tensors(golden["prime_source"])
    expected = _restore_tensors(golden["legacy_hf"])
    config = {"model_type": FAMILY_MODEL_TYPES[family], **golden["config_kwargs"]}
    output, index, exported = _export_prime_state(tmp_path, source, config)

    assert set(exported) == set(expected) == set(index["weight_map"])
    assert index["metadata"]["total_size"] == sum(len(_raw_bytes(tensor)) for tensor in expected.values())
    for name, want in expected.items():
        got = exported[name]
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert _raw_bytes(got) == _raw_bytes(want)

    config_class, model_class = HF_LOADERS[family]
    config_class(**golden["config_kwargs"]).save_pretrained(output)
    model, info = model_class.from_pretrained(output, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    with torch.no_grad():
        logits = model(torch.tensor([[0, 1, 2, 3]])).logits
    assert logits.shape == (1, 4, golden["config_kwargs"]["vocab_size"])
    assert torch.isfinite(logits).all()


def test_native_tied_alias_export_drops_alias_and_loads(tmp_path):
    kwargs = dict(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=16,
        tie_word_embeddings=True,
    )
    torch.manual_seed(20260101)
    state = {
        name: tensor.detach().clone()
        for name, tensor in LlamaForCausalLM._from_config(LlamaConfig(**kwargs)).state_dict().items()
    }
    output, _, exported = _export_prime_state(tmp_path, state, {"model_type": "llama", **kwargs})

    assert "lm_head.weight" not in exported
    assert set(exported) == set(state) - {"lm_head.weight"}
    LlamaConfig(**kwargs).save_pretrained(output)
    restored, info = LlamaForCausalLM.from_pretrained(output, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    with torch.no_grad():
        logits = restored(torch.tensor([[0, 1, 2, 3]])).logits
    assert torch.isfinite(logits).all()
