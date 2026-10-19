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

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.utils import dcp_export
from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_conversion import conversion_chain
from prime_rl.utils.utils import default_dtype


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


def _glm_model(model_type="glm4_moe", num_hidden_layers=2):
    torch.manual_seed(0)
    if model_type == "glm4_moe":
        from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM

        config = Glm4MoeConfig(
            hidden_size=8,
            intermediate_size=16,
            max_position_embeddings=16,
            moe_intermediate_size=4,
            norm_topk_prob=True,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
            num_hidden_layers=num_hidden_layers,
            rope_theta=10000.0,
            first_k_dense_replace=0,
            partial_rotary_factor=0.5,
            vocab_size=32,
            tie_word_embeddings=False,
        )
        model_cls = Glm4MoeForCausalLM
    else:
        from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

        config = GlmMoeDsaConfig(
            hidden_size=8,
            intermediate_size=16,
            max_position_embeddings=16,
            moe_intermediate_size=4,
            norm_topk_prob=True,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
            num_hidden_layers=num_hidden_layers,
            first_k_dense_replace=0,
            kv_lora_rank=8,
            q_lora_rank=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            qk_nope_head_dim=8,
            vocab_size=32,
            pad_token_id=0,
            dsa_backend="tilelang",
            tie_word_embeddings=False,
        )
        model_cls = GlmMoeDsaForCausalLM
    with torch.device("cpu"), default_dtype(torch.bfloat16):
        model = model_cls._from_config(config)
    from prime_rl.trainer.models.layers.moe import GroupedExperts

    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.init_weights(config.initializer_range)
    return model, config


def _glm_config(model_type="glm4_moe", num_hidden_layers=2):
    _, config = _glm_model(model_type, num_hidden_layers)
    return config.to_dict()


def _glm_state(model_type="glm4_moe", num_hidden_layers=2):
    model, _ = _glm_model(model_type, num_hidden_layers)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _legacy_glm_hf_state(state, config):
    model, _ = _glm_model(config["model_type"], config["num_hidden_layers"])
    gathered = {
        name: tensor.to(torch.float32 if model.keep_in_fp32_for_weight_transfer(name) else torch.bfloat16)
        if tensor.is_floating_point()
        else tensor
        for name, tensor in state.items()
    }
    return apply_prime_to_hf(gathered, conversion_chain(config))


@pytest.fixture
def glm_state():
    return _glm_state()


@pytest.mark.parametrize("model_type", ["glm4_moe", "glm_moe_dsa"])
def test_cpu_export_reads_model_groups_and_converts_glm(tmp_path, monkeypatch, model_type):
    state = _glm_state(model_type)
    checkpoint = _save_model(tmp_path, state)
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

    units = {f"model.layers.{name.split('.')[2]}" if name.startswith("model.layers.") else name for name in state}
    assert len(reads) == len(groups) == len(units)
    assert set.union(*reads) == {f"app.model.{name}" for name in state}
    assert sum(map(len, reads)) == len(state)
    for layer in range(config["num_hidden_layers"]):
        layer_names = {f"app.model.{name}" for name in state if name.startswith(f"model.layers.{layer}.")}
        assert sum(1 for read in reads if read & layer_names) == 1

    expected = _legacy_glm_hf_state(state, config)
    assert output.keys() == expected.keys()
    for name, tensor in expected.items():
        assert output[name].device.type == "cpu"
        assert output[name].dtype == tensor.dtype
        torch.testing.assert_close(output[name], tensor, rtol=0, atol=0)


def test_cpu_export_rejects_unrecognized_hf_model(tmp_path):
    checkpoint = _save_model(tmp_path, {"weight": torch.ones(1)})
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, {"model_type": "unsupported"}))


@pytest.mark.parametrize("key", ["lora_A.weight"])
def test_cpu_export_rejects_lora(tmp_path, key):
    checkpoint = _save_model(tmp_path, {f"model.layers.0.self_attn.q_proj.{key}": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="LoRA"):
        list(iter_cpu_hf_groups(checkpoint, {"model_type": "glm4_moe", "num_hidden_layers": 1}))


@pytest.mark.parametrize("missing", ["down_proj"])
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
        state = _glm_state("glm_moe_dsa", num_hidden_layers=1)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.0.mlp.experts.{projection}"
            state[name] = DTensor.from_local(state[name][rank : rank + 1].clone(), mesh, [Shard(0)], run_check=True)
        dcp.save(
            {"app": {"model": state, "optimizer": {"exp_avg": torch.ones(1)}}},
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
    reference = _glm_state("glm_moe_dsa", num_hidden_layers=1)
    config = _glm_config("glm_moe_dsa", num_hidden_layers=1)
    groups = list(iter_cpu_hf_groups(gloo_checkpoint, config))
    assert len(groups) == 4
    output = {name: tensor for group in groups for name, tensor in group.items()}
    assert all(not name.startswith("app.") for name in output)
    expected = _legacy_glm_hf_state(reference, config)
    assert output.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(output[name], tensor, rtol=0, atol=0)
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


@pytest.mark.parametrize("missing", ["model.norm.weight", "model.layers.1."])
def test_cpu_export_rejects_missing_model_structure(tmp_path, glm_state, missing):
    model = {name: tensor for name, tensor in glm_state.items() if name != missing and not name.startswith(missing)}
    checkpoint = _save_model(tmp_path, model)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, _glm_config()))


@pytest.mark.parametrize(
    ("model_type", "config_fields", "sensitive"),
    [
        ("afmoe", {}, ["model.layers.0.mlp.expert_bias"]),
        ("glm4_moe", {}, ["model.layers.0.mlp.gate.e_score_correction_bias"]),
        ("glm_moe_dsa", {}, ["model.layers.0.mlp.gate.e_score_correction_bias"]),
        (
            "nemotron_h",
            {"layer_types": ["linear_attention", "moe"], "num_hidden_layers": 2},
            [
                "backbone.layers.0.mixer.A_log",
                "backbone.layers.0.mixer.D",
                "backbone.layers.1.mixer.gate.e_score_correction_bias",
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hf_identity_uses_canonical_fp32_policy(tmp_path, monkeypatch, model_type, config_fields, sensitive, dtype):
    exact = torch.tensor([0.1234567, -0.7654321], dtype=dtype)
    state = {name: exact.clone() for name in sensitive}
    state["lm_head.weight"] = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    schema = {name: tuple(tensor.shape) for name, tensor in state.items()}
    monkeypatch.setattr(
        dcp_export,
        "hf_schema_details",
        lambda *args: (schema, {}, {name: tensor.dtype for name, tensor in state.items()}),
    )
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    config = {"model_type": model_type, "num_hidden_layers": 1, **config_fields}
    output = {name: tensor for group in iter_cpu_hf_groups(checkpoint, config) for name, tensor in group.items()}
    assert output.keys() == state.keys()
    for name in sensitive:
        expected_dtype = dtype if name.endswith(("expert_bias", "e_score_correction_bias")) else torch.float32
        assert output[name].dtype == expected_dtype
        torch.testing.assert_close(output[name], exact.to(expected_dtype), rtol=0, atol=0)
    assert output["lm_head.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("integer_dtype", [torch.int8, torch.uint8])
def test_hf_identity_rejects_integer_payload_for_floating_state(tmp_path, monkeypatch, integer_dtype):
    state = {"lm_head.weight": torch.arange(6, dtype=integer_dtype).reshape(2, 3)}
    schema = {"lm_head.weight": (2, 3)}
    monkeypatch.setattr(dcp_export, "hf_schema_details", lambda *args: (schema, {}, {"lm_head.weight": torch.float32}))
    checkpoint = tmp_path / "checkpoint"
    dcp.save({"app": {"model": state}}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError):
        list(iter_cpu_hf_groups(checkpoint, {"model_type": "gpt2"}))
