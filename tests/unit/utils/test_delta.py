import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.utils.delta import (
    DELTA_INDEX_SUFFIX,
    DELTA_METADATA_FORMAT_KEY,
    DELTA_METADATA_FORMAT_VALUE,
    DELTA_METADATA_LAYOUT_KEY,
    DELTA_METADATA_SHAPES_KEY,
    DELTA_VALUE_SUFFIX,
    ModelDeltaManager,
    count_sparse_delta_values,
    decode_sparse_indices,
    decode_varint_delta_indices,
    encode_varint_delta_indices,
    is_streaming_delta_file,
    iter_streaming_delta_records,
    verify_sparse_delta_file,
    verify_sparse_delta_state_dicts,
)


def test_varint_delta_indices_round_trip() -> None:
    indices = torch.tensor([0, 1, 2, 127, 128, 129, 16_384, 100_000], dtype=torch.int64)

    encoded = encode_varint_delta_indices(indices)
    decoded = decode_varint_delta_indices(encoded, expected_count=indices.numel())

    assert encoded.dtype == torch.uint8
    assert torch.equal(decoded, indices)


def test_varint_delta_indices_fast_path_round_trip_dense_range() -> None:
    indices = torch.arange(4096, dtype=torch.int64)

    encoded = encode_varint_delta_indices(indices)
    decoded = decode_varint_delta_indices(encoded, expected_count=indices.numel())

    assert encoded.numel() == indices.numel()
    assert torch.equal(decoded, indices)


def test_varint_delta_indices_reject_unsorted_indices() -> None:
    with pytest.raises(ValueError, match="sorted"):
        encode_varint_delta_indices(torch.tensor([2, 1], dtype=torch.int64))


@pytest.mark.parametrize("streaming", [False, True])
def test_sparse_delta_from_state_dicts_round_trips_bias(tmp_path, streaming) -> None:
    base = {
        "linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "linear.bias": torch.tensor([1.0, 1.0]),
    }
    target = {
        "linear.weight": torch.tensor([[1.0, 2.5], [3.0, 6.0]]),
        "linear.bias": torch.tensor([4.0, 5.0]),
    }
    delta_path = tmp_path / "delta.safetensors"

    manager = ModelDeltaManager()
    extract = (
        manager.extract_sparse_delta_streaming_from_state_dicts
        if streaming
        else manager.extract_sparse_delta_from_state_dicts
    )
    extract(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    if streaming:
        assert {record.name for record in iter_streaming_delta_records(delta_path)} == {"linear.weight", "linear.bias"}
        return
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        assert f"linear.weight{DELTA_INDEX_SUFFIX}" in keys
        assert f"linear.weight{DELTA_VALUE_SUFFIX}" in keys
        assert f"linear.bias{DELTA_INDEX_SUFFIX}" in keys


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("target_value", [-0.001, 2**-40])
def test_sparse_delta_preserves_target_bits_across_sign_changes(tmp_path, dtype, streaming, target_value) -> None:
    base = {"weight": torch.tensor([1.0, 0.0], dtype=dtype)}
    target = {"weight": torch.tensor([target_value, 0.0], dtype=dtype)}
    delta_path = tmp_path / "delta"
    manager = ModelDeltaManager()
    extract = (
        manager.extract_sparse_delta_streaming_from_state_dicts
        if streaming
        else manager.extract_sparse_delta_from_state_dicts
    )
    extract(base, target, delta_path)
    assert verify_sparse_delta_state_dicts(base, target, delta_path, atol=0, rtol=0).ok


def test_sparse_delta_from_identical_state_dicts_creates_verifiable_empty_delta(tmp_path) -> None:
    base = {"linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    target = {"linear.weight": base["linear.weight"].clone()}
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        assert list(delta.keys()) == []
        assert delta.metadata()[DELTA_METADATA_FORMAT_KEY] == DELTA_METADATA_FORMAT_VALUE


def test_sparse_delta_preserves_logical_names_and_shapes(tmp_path) -> None:
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2)),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.mlp.up_proj.weight": torch.zeros((1, 2)),
    }
    target = {name: tensor.clone() for name, tensor in base.items()}
    target["model.layers.0.self_attn.q_proj.weight"][1, 1] = 10.0
    target["model.layers.0.self_attn.k_proj.weight"][0, 1] = 20.0
    target["model.layers.0.self_attn.v_proj.weight"][0, 0] = -1.0
    target["model.layers.0.mlp.gate_proj.weight"][0, 0] = 3.0
    target["model.layers.0.mlp.up_proj.weight"][0, 1] = 4.0
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        q_name = "model.layers.0.self_attn.q_proj.weight"
        k_name = "model.layers.0.self_attn.k_proj.weight"
        v_name = "model.layers.0.self_attn.v_proj.weight"
        assert f"{q_name}{DELTA_INDEX_SUFFIX}" in keys
        assert f"{k_name}{DELTA_INDEX_SUFFIX}" in keys
        assert f"{v_name}{DELTA_INDEX_SUFFIX}" in keys
        assert f"model.layers.0.self_attn.qkv_proj.weight{DELTA_INDEX_SUFFIX}" not in keys

        q_values = delta.get_tensor(f"{q_name}{DELTA_VALUE_SUFFIX}")
        q_indices = decode_sparse_indices(delta.get_tensor(f"{q_name}{DELTA_INDEX_SUFFIX}"), q_values.numel())
        assert q_indices.tolist() == [3]
        assert q_values.tolist() == [10.0]
        metadata = delta.metadata()
        assert metadata[DELTA_METADATA_LAYOUT_KEY] == "huggingface"
        shapes = json.loads(metadata[DELTA_METADATA_SHAPES_KEY])
        assert shapes[q_name] == [2, 2]
        assert shapes[k_name] == [1, 2]
        assert shapes[v_name] == [1, 2]


@pytest.mark.parametrize(
    ("index_encoding", "index_dtype"),
    [("optimized", torch.uint8), ("naive", torch.int32)],
)
def test_streaming_sparse_delta_round_trips_logical_tensors(
    tmp_path,
    index_encoding: str,
    index_dtype: torch.dtype,
) -> None:
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2)),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros((1, 2)),
        "model.norm.weight": torch.ones(2),
    }
    target = {name: tensor.clone() for name, tensor in base.items()}
    target["model.layers.0.self_attn.q_proj.weight"][1, 1] = 10.0
    target["model.layers.0.self_attn.k_proj.weight"][0, 1] = 20.0
    target["model.layers.0.self_attn.v_proj.weight"][0, 0] = -1.0
    delta_path = tmp_path / "delta.stream"

    stats = ModelDeltaManager().extract_sparse_delta_streaming_from_state_dicts(
        base,
        target,
        delta_path,
        group_size=4,
        index_encoding=index_encoding,
        save_stats=True,
    )

    records = list(iter_streaming_delta_records(delta_path))
    assert is_streaming_delta_file(delta_path)
    assert stats is not None and stats.changed_params == 3
    assert count_sparse_delta_values(delta_path) == 3
    assert verify_sparse_delta_state_dicts(base, target, delta_path).ok
    assert [record.name for record in records] == [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    assert all(record.format_version == 2 for record in records)
    assert all(record.encoded_indices.dtype == index_dtype for record in records)
    assert [record.shape for record in records] == [(1, 2), (2, 2), (1, 2)]
    assert [record.values.tolist() for record in records] == [[20.0], [10.0], [-1.0]]
    assert [decode_sparse_indices(record.encoded_indices, 1).tolist() for record in records] == [[1], [3], [0]]


def test_tied_lm_head_is_skipped(tmp_path) -> None:
    embedding = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    base = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding.clone(),
        "model.norm.weight": torch.ones(2),
    }
    target_embedding = embedding + 1.0
    target = {
        "model.embed_tokens.weight": target_embedding,
        "lm_head.weight": target_embedding.clone(),
        "model.norm.weight": torch.ones(2),
    }
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        assert f"model.embed_tokens.weight{DELTA_INDEX_SUFFIX}" in keys
        assert f"lm_head.weight{DELTA_INDEX_SUFFIX}" not in keys


def test_lm_head_is_not_skipped_when_target_is_untied(tmp_path) -> None:
    embedding = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    base = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding.clone(),
    }
    target = {
        "model.embed_tokens.weight": embedding + 1.0,
        "lm_head.weight": embedding + 2.0,
    }
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        assert f"model.embed_tokens.weight{DELTA_INDEX_SUFFIX}" in keys
        assert f"lm_head.weight{DELTA_INDEX_SUFFIX}" in keys


def test_extract_delta_fused_sparse_file_round_trip_and_stats(tmp_path) -> None:
    base = {"model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    target = {"model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 5.0], [3.0, 8.0]])}
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    delta_path = tmp_path / "delta.safetensors"
    save_file(base, base_path)
    save_file(target, target_path)

    summary = ModelDeltaManager().extract_delta_fused_sparse(
        base_model_path=base_path,
        finetuned_model_path=target_path,
        delta_output_path=delta_path,
        save_stats=True,
        index_encoding="naive",
    )

    assert summary["changed_params"] == 2
    assert verify_sparse_delta_file(base_path, target_path, delta_path).ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        idx = delta.get_tensor(f"model.layers.0.mlp.down_proj.weight{DELTA_INDEX_SUFFIX}")
        assert idx.dtype == torch.int32

    stats_path = tmp_path / "delta_stats.json"
    assert json.loads(stats_path.read_text(encoding="utf-8"))["summary"]["changed_params"] == 2
