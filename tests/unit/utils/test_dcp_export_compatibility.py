"""Compare CPU exports with compact hashes of main's legacy conversion outputs."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards

GOLDENS = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures/dcp_export/legacy_export_goldens.json").read_text()
)
MODEL_TYPES = {"llama_dense": "llama", "qwen3_moe": "qwen3_moe"}
HF_LOADERS = {"llama_dense": (LlamaConfig, LlamaForCausalLM), "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM)}


def _production_init(model):
    """Production init for prime-rl custom modules HF post_init does not reach (grouped experts are empty otherwise)."""
    from prime_rl.trainer.models.layers.mlp import FeedForward
    from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter

    std = getattr(model.config, "initializer_range", 0.02)
    for module in model.modules():
        if isinstance(module, MoE):
            module.init_weights(std, buffer_device=torch.device("cpu"))
        elif isinstance(module, (GroupedExperts, TokenChoiceTopKRouter, FeedForward)):
            module.init_weights(std)


def _build_prime_source(family):
    """Deterministically rebuild the real tiny PrimeRL model exactly as the golden generator did."""
    if family == "llama_dense":
        from prime_rl.trainer.models.llama import LlamaForCausalLM as Model
    else:
        from prime_rl.trainer.models.qwen3_moe import Qwen3MoeForCausalLM as Model
    golden = GOLDENS[family]
    torch.manual_seed(golden["seed"])
    model = Model._from_config(Model.config_class(**golden["config_kwargs"]))
    model.to("cpu")
    _production_init(model)
    return model, golden


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("family", sorted(MODEL_TYPES))
def test_cpu_export_matches_legacy_golden_and_loads_cleanly_in_hf(tmp_path, family):
    model, golden = _build_prime_source(family)
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": dict(model.state_dict())}}, checkpoint_id=checkpoint, no_dist=True)
    output = tmp_path / "hf"
    write_hf_shards(
        iter_cpu_hf_groups(checkpoint, {"model_type": MODEL_TYPES[family], **golden["config_kwargs"]}), output
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    exported = {
        name: tensor
        for filename in set(index["weight_map"].values())
        for name, tensor in load_file(output / filename).items()
    }
    expected = golden["hf"]
    assert set(exported) == set(expected)
    for name, (sha, shape, dtype) in expected.items():
        got = exported[name]
        assert list(got.shape) == shape
        assert str(got.dtype).replace("torch.", "") == dtype
        assert _digest(got) == sha

    config_class, model_class = HF_LOADERS[family]
    config_class(**golden["config_kwargs"]).save_pretrained(output)
    restored, info = model_class.from_pretrained(output, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    with torch.no_grad():
        logits = restored(torch.tensor([[0, 1, 2, 3]])).logits
    assert logits.shape == (1, 4, golden["config_kwargs"]["vocab_size"])
    assert torch.isfinite(logits).all()
