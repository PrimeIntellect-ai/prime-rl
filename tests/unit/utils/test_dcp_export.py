import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, TensorProperties, TensorStorageMetadata
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from prime_rl.utils import dcp_export
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards


def _save_model(tmp_path: Path, model: dict[str, torch.Tensor]) -> Path:
    checkpoint = tmp_path / "checkpoint"
    dcp.save(
        {
            "app": {
                "model": model,
                "optimizer": {
                    "state": {"model.layers.0.mlp.experts.gate_proj": {"exp_avg": torch.full((2, 2, 3), 99.0)}},
                    "param_groups": [{"lr": 0.001}],
                },
                "step": 7,
            }
        },
        checkpoint_id=checkpoint,
        no_dist=True,
    )
    return checkpoint


def _glm_config(model_type="glm4_moe", num_hidden_layers=2):
    return {
        "model_type": model_type,
        "num_hidden_layers": num_hidden_layers,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 0,
    }


def _glm_state(model_type="glm4_moe", num_hidden_layers=2):
    state = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "model.norm.weight": torch.ones(3),
        "lm_head.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
    }
    attention = ("q_proj", "k_proj", "v_proj", "o_proj")
    if model_type == "glm_moe_dsa":
        attention = (
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o_proj",
        )
    for layer in range(num_hidden_layers):
        base = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            state[f"{base}.{norm}.weight"] = torch.ones(3)
        for projection in attention:
            state[f"{base}.self_attn.{projection}.weight"] = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        prefix = f"{base}.mlp"
        state[f"{prefix}.router.gate.weight"] = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        state[f"{prefix}.router.selection_bias"] = torch.tensor([0.1234567, -0.7654321], dtype=torch.float32)
        state[f"{prefix}.tokens_per_expert"] = torch.tensor([2, 3], dtype=torch.int64)
        for offset, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            shape = (2, 3, 2) if projection == "down_proj" else (2, 2, 3)
            state[f"{prefix}.experts.{projection}"] = torch.arange(12, dtype=torch.float32).reshape(shape) + offset * 20
            state[f"{prefix}.shared_expert.{projection}.weight"] = torch.arange(6, dtype=torch.float32).reshape(
                (1, *shape[1:])
            )
    return state


@pytest.fixture
def glm_state():
    return _glm_state()


@pytest.mark.parametrize("model_type", ["glm4_moe", "glm_moe_dsa"])
def test_cpu_export_reads_model_groups_and_converts_glm(tmp_path, monkeypatch, glm_state, model_type):
    glm_state = _glm_state(model_type)
    checkpoint = _save_model(tmp_path, glm_state)
    config = _glm_config(model_type)
    reads = []
    original_load = dcp_export.dcp_load

    def record_load(state_dict, **kwargs):
        reads.append(set(state_dict))
        assert all(tensor.device.type == "cpu" for tensor in state_dict.values())
        return original_load(state_dict, **kwargs)

    monkeypatch.setattr(dcp_export, "dcp_load", record_load)
    groups = list(iter_cpu_hf_groups(checkpoint, config))
    output = {name: tensor for group in groups for name, tensor in group.items()}

    assert len(reads) == len(groups) == 5
    assert set.union(*reads) == {f"app.model.{name}" for name in glm_state}
    assert sum(map(len, reads)) == len(glm_state)
    layer_zero = {f"app.model.{name}" for name in glm_state if name.startswith("model.layers.0.")}
    assert layer_zero in reads
    layer_one = {f"app.model.{name}" for name in glm_state if name.startswith("model.layers.1.")}
    assert layer_one in reads

    expected = {name: tensor for name, tensor in glm_state.items() if ".mlp." not in name}
    for layer in range(2):
        prefix = f"model.layers.{layer}.mlp"
        expected[f"{prefix}.gate.weight"] = glm_state[f"{prefix}.router.gate.weight"]
        expected[f"{prefix}.gate.e_score_correction_bias"] = glm_state[f"{prefix}.router.selection_bias"]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for expert in range(2):
                expected[f"{prefix}.experts.{expert}.{projection}.weight"] = glm_state[
                    f"{prefix}.experts.{projection}"
                ][expert]
            expected[f"{prefix}.shared_experts.{projection}.weight"] = glm_state[
                f"{prefix}.shared_expert.{projection}.weight"
            ][0]
    assert output.keys() == expected.keys()
    for name, tensor in expected.items():
        dtype = torch.float32 if name.endswith("e_score_correction_bias") else torch.bfloat16
        assert output[name].device.type == "cpu"
        assert output[name].dtype == dtype
        torch.testing.assert_close(output[name], tensor.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_cpu_export_dense_and_tied_embeddings(tmp_path, model_type):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    model = {
        "model.embed_tokens.weight": weight,
        "lm_head.weight": weight.clone(),
        "model.norm.weight": torch.ones(3),
        "model.layers.0.input_layernorm.weight": torch.ones(3),
    }
    model["model.layers.0.post_attention_layernorm.weight"] = torch.ones(3)
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        model[f"model.layers.0.self_attn.{projection}.weight"] = torch.ones(3, 3)
    if model_type == "qwen3":
        for norm in ("q_norm", "k_norm"):
            model[f"model.layers.0.self_attn.{norm}.weight"] = torch.ones(3)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        model[f"model.layers.0.mlp.{projection}.weight"] = torch.ones(3, 3)
    checkpoint = _save_model(tmp_path, model)
    groups = list(
        iter_cpu_hf_groups(
            checkpoint,
            {"model_type": model_type, "num_hidden_layers": 1, "tie_word_embeddings": True},
        )
    )
    output = {name: tensor for group in groups for name, tensor in group.items()}
    assert set(output) == set(model) - {"lm_head.weight"}
    torch.testing.assert_close(output["model.embed_tokens.weight"], weight.bfloat16(), rtol=0, atol=0)


def test_cpu_export_rejects_unrecognized_hf_model(tmp_path):
    checkpoint = _save_model(tmp_path, {"weight": torch.ones(1)})
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, {"model_type": "unsupported"}))


@pytest.mark.parametrize("key", ["lora_A.weight", "lora_B.weight", "base_layer.weight"])
def test_cpu_export_rejects_lora(tmp_path, key):
    checkpoint = _save_model(tmp_path, {f"model.layers.0.self_attn.q_proj.{key}": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="LoRA"):
        list(iter_cpu_hf_groups(checkpoint, {"model_type": "glm4_moe", "num_hidden_layers": 1}))


@pytest.mark.parametrize("missing", ["gate_proj", "up_proj", "down_proj"])
def test_cpu_export_rejects_missing_expert_dependency(tmp_path, missing, glm_state):
    model = dict(glm_state)
    del model[f"model.layers.0.mlp.experts.{missing}"]
    checkpoint = _save_model(tmp_path, model)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, _glm_config()))


def test_write_hf_shards_preserves_bytes_views_and_oversized_tensors(tmp_path):
    groups = [
        {"a": torch.arange(4, dtype=torch.bfloat16), "b": torch.tensor([0.1234567, -0.7654321])},
        {"oversized": torch.arange(10, dtype=torch.float32)},
        {"view": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)[:2, :2].T, "flags": torch.tensor([True, False])},
    ]
    output_dir = tmp_path / "hf"
    expected = {name: tensor.clone() for group in groups for name, tensor in group.items()}
    write_hf_shards(iter(groups), output_dir, max_shard_size=16)
    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in expected.values())
    assert set(index["weight_map"]) == set(expected)
    shard_names = set(index["weight_map"].values())
    assert len(shard_names) == 3
    assert {path.name for path in output_dir.glob("*.safetensors")} == shard_names
    for filename in shard_names:
        tensors = load_file(output_dir / filename)
        assert set(tensors) == {name for name, shard in index["weight_map"].items() if shard == filename}
        with safe_open(output_dir / filename, framework="pt") as handle:
            assert handle.metadata() == {"format": "pt"}
        for name, tensor in tensors.items():
            assert tensor.dtype == expected[name].dtype
            assert torch.equal(tensor.contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8))
        shard_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if shard_bytes > 16:
            assert set(tensors) == {"oversized"}
    assert not list(output_dir.glob("*.tmp"))


def test_write_hf_shards_rejects_duplicate_keys(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        write_hf_shards([{"weight": torch.ones(1)}, {"weight": torch.zeros(1)}], tmp_path / "hf")
    assert not (tmp_path / "hf" / "model.safetensors.index.json").exists()


def _save_gloo_checkpoint(rank: int, checkpoint: str, rendezvous: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30))
    try:
        mesh = init_device_mesh("cpu", (2,))
        model = _glm_state("glm_moe_dsa", num_hidden_layers=1)
        for offset, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            shape = (2, 3, 2) if projection == "down_proj" else (2, 2, 3)
            full = torch.arange(12, dtype=torch.float32).reshape(shape) + offset * 20
            model[f"model.layers.0.mlp.experts.{projection}"] = DTensor.from_local(
                full[rank : rank + 1].clone(), mesh, [Shard(0)], run_check=True
            )
        model["model.layers.0.mlp.router.selection_bias"] = torch.tensor([0.1234567, -0.7654321])
        dcp.save(
            {"app": {"model": model, "optimizer": {"exp_avg": torch.full((2, 2, 3), 99.0)}}},
            checkpoint_id=checkpoint,
        )
    finally:
        dist.destroy_process_group()


@pytest.fixture
def gloo_checkpoint(tmp_path):
    checkpoint = tmp_path / "distributed-checkpoint"
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_save_gloo_checkpoint, args=(str(checkpoint), rendezvous), nprocs=2, join=True)
    return checkpoint


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_cpu_export_reconstructs_multirank_expert_chunks(gloo_checkpoint):
    assert not dist.is_initialized()
    groups = list(iter_cpu_hf_groups(gloo_checkpoint, _glm_config("glm_moe_dsa", num_hidden_layers=1)))
    assert len(groups) == 4
    output = {name: tensor for group in groups for name, tensor in group.items()}
    prefix = "model.layers.0.mlp"
    expected_keys = {
        f"{prefix}.gate.e_score_correction_bias",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for offset, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
        shape = (2, 3, 2) if projection == "down_proj" else (2, 2, 3)
        full = (torch.arange(12, dtype=torch.float32).reshape(shape) + offset * 20).bfloat16()
        for expert in range(2):
            name = f"{prefix}.experts.{expert}.{projection}.weight"
            expected_keys.add(name)
            torch.testing.assert_close(output[name], full[expert], rtol=0, atol=0)
    assert expected_keys <= set(output)
    assert all(not name.startswith("app.") for name in output)
    torch.testing.assert_close(
        output[f"{prefix}.gate.e_score_correction_bias"],
        torch.tensor([0.1234567, -0.7654321]),
        rtol=0,
        atol=0,
    )
    assert not dist.is_initialized()


@pytest.mark.parametrize(
    ("chunks", "error"),
    [
        ([((0, 0), (2, 3)), ((2, 0), (2, 3))], None),
        ([((0, 0), (2, 3)), ((3, 0), (1, 3))], "do not cover"),
        ([((0, 0), (2, 3)), ((1, 0), (2, 3))], "Overlapping"),
        ([((0, 0), (2, 3)), ((3, 0), (2, 3))], "Out-of-bounds"),
    ],
    ids=["valid-sharded", "gap", "overlap-with-equal-volume", "out-of-bounds"],
)
def test_validate_checkpoint_chunks(chunks, error):
    metadata = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size((4, 3)),
        chunks=[ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes)) for offsets, sizes in chunks],
    )
    if error is None:
        dcp_export._validate_chunks("model.weight", metadata)
    else:
        with pytest.raises(ValueError, match=error):
            dcp_export._validate_chunks("model.weight", metadata)


@pytest.mark.parametrize(
    "missing", ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "model.layers.1."]
)
def test_cpu_export_rejects_missing_model_structure(tmp_path, glm_state, missing):
    model = {name: tensor for name, tensor in glm_state.items() if name != missing and not name.startswith(missing)}
    checkpoint = _save_model(tmp_path, model)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, _glm_config()))
