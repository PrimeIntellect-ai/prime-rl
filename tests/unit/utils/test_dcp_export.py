import importlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorStorageMetadata
from transformers import PretrainedConfig

from prime_rl.trainer.models.layers.ngram import LAYOUT_BUFFERS
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextForCausalLM, Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("convert_dcp_to_bf16")


def save_checkpoint(path: Path, state: dict[str, torch.Tensor], row_chunks: int) -> int:
    """Save a DCP checkpoint whose n-gram tables are stored as several row chunks, like a multi-rank run."""
    tables = [name for name in state if name.startswith("model.ngram.tables.") and name.endswith(".weight")]
    saved = dict(state)
    bounds = {}
    for name in tables:
        table = saved.pop(name)
        # Uneven chunks, so HF shards straddle chunk boundaries.
        rows = table.shape[0]
        bounds[name] = [0, *sorted({rows * index // (2 * row_chunks) for index in range(1, row_chunks)}), rows]
        for index, (start, stop) in enumerate(zip(bounds[name], bounds[name][1:])):
            saved[f"{name}@{index}"] = table[start:stop].clone()
    dcp.save({"app": {"model": saved, "optimizer": {"unused": torch.ones(1)}}}, checkpoint_id=path, no_dist=True)
    metadata = FileSystemReader(path).read_metadata()
    for name in tables:
        key = f"app.model.{name}"
        chunks = []
        for index, start in enumerate(bounds[name][:-1]):
            piece = f"{key}@{index}"
            piece_metadata = metadata.state_dict_metadata.pop(piece)
            chunks.append(ChunkStorageMetadata(torch.Size([start, 0]), piece_metadata.size))
            stored = metadata.storage_data.pop(next(item for item in metadata.storage_data if item.fqn == piece))
            metadata.storage_data[MetadataIndex(key, chunks[-1].offsets, index)] = stored
        metadata.state_dict_metadata[key] = TensorStorageMetadata(piece_metadata.properties, state[name].shape, chunks)
    with (path / ".metadata").open("wb") as file:
        pickle.dump(metadata, file)
    return sum(len(table_bounds) - 1 for table_bounds in bounds.values())


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ngram_parts", [None, 3, 128])
def test_layer_export_matches_original_hf_weights(tmp_path, exporter, monkeypatch, tied, fp8, ngram_parts):
    config_kwargs = dict(
        hidden_size=16,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=32,
        tie_word_embeddings=tied,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
    )
    if ngram_parts is None:
        config = Qwen3MoeConfig(**config_kwargs)
        model = Qwen3MoeForCausalLM._from_config(config)
    else:
        config = Qwen3_8FlashNextTextConfig(
            **config_kwargs,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            hc_count=2,
            hc_lowrank=4,
            shared_expert_intermediate_size=16,
            ple_layer_ids=[1, 2],
            ple_embed_dim=16,
            heads_per_ngram=2,
            ngram_vocab_size_base=17,
            make_ngram_vocab_size_divisible_by=8,
            split_ngram_parts=ngram_parts,
        )
        model = Qwen3_8FlashNextForCausalLM(config)

    # Build the source HF layout independently of the export conversion.
    original = {
        name: torch.randn_like(tensor, dtype=torch.float32) if tensor.is_floating_point() else tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    if tied:
        original["lm_head.weight"] = original["model.embed_tokens.weight"].clone()
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp"
        original[f"{prefix}.gate.weight"] = original.pop(f"{prefix}.router.gate.weight")
        if ngram_parts is None:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for expert, tensor in enumerate(original.pop(f"{prefix}.experts.{proj}").unbind()):
                    original[f"{prefix}.experts.{expert}.{proj}.weight"] = tensor.clone()
        else:
            original[f"{prefix}.experts.gate_up_proj"] = torch.cat(
                [original.pop(f"{prefix}.experts.{proj}") for proj in ("gate_proj", "up_proj")], dim=1
            )
            original[f"{prefix}.shared_expert_gate.weight"] = original.pop(f"{prefix}.shared_expert.output_gate.weight")
            runtime_prefix = f"model.ngram.tables.{layer}"
            hf_prefix = f"model.layers.{layer}.ple.ple_embedding"
            table = original.pop(f"{runtime_prefix}.weight")
            rows = (table.shape[0] + ngram_parts - 1) // ngram_parts
            for index in range(ngram_parts):
                original[f"{hf_prefix}.ngram_embedding.shard_{index}.weight"] = table[
                    index * rows : (index + 1) * rows
                ].clone()
            for buffer in LAYOUT_BUFFERS:
                original[f"{hf_prefix}.{buffer}"] = original.pop(f"{runtime_prefix}.{buffer}")

    state = model.convert_to_prime({name: tensor.clone() for name, tensor in original.items()})
    assert all(tensor.dtype == torch.float32 for tensor in state.values() if tensor.is_floating_point())
    stored_table_chunks = save_checkpoint(tmp_path, state, row_chunks=3)
    table_reads = []
    load_rows = exporter.load_rows
    monkeypatch.setattr(exporter, "load_rows", lambda *args: table_reads.append(args[1:]) or load_rows(*args))
    checkpoint = exporter.Checkpoint(tmp_path, FileSystemReader(tmp_path).read_metadata(), model.to("meta"), None, None)
    expected = {
        name: tensor.bfloat16() if tensor.is_floating_point() else tensor.clone()
        for name, tensor in original.items()
        if not (tied and name == "lm_head.weight")
    }
    quantize = importlib.import_module("convert_bf16_to_fp8").quantize_state_dict

    def transform(shard):
        return quantize(shard, 2, "cpu")[0] if fp8 else shard

    expected = transform(expected)
    for workers in (1, 2):
        output = tmp_path / f"export-{workers}"
        table_reads.clear()
        written = exporter.write_weights(checkpoint, output, workers, transform)
        # Each stored table chunk is read once, however many HF shards it feeds.
        assert len(table_reads) == len(set(table_reads)) == stored_table_chunks
        index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
        actual = {name: tensor for file in set(index.values()) for name, tensor in load_file(output / file).items()}
        assert written == index
        assert actual.keys() == expected.keys() == index.keys()
        for name, tensor in expected.items():
            assert actual[name].dtype == tensor.dtype
            assert actual[name].shape == tensor.shape
            assert torch.equal(actual[name].view(torch.uint8), tensor.view(torch.uint8))
            if ".ngram_embedding.shard_" in name:
                assert list(load_file(output / index[name])) == [name]


def test_export_config_drops_source_quantization(exporter):
    config = PretrainedConfig()
    config.text_config = PretrainedConfig()
    config.text_config.quantization_config = {"quant_method": "fp8"}
    exporter._export_config(config)
    assert config.dtype == config.text_config.dtype == "bfloat16"
    assert not hasattr(config.text_config, "quantization_config")


def test_only_the_default_output_dir_is_replaced_without_overwrite(tmp_path, exporter):
    step = tmp_path / "step_10"
    checkpoint = exporter.Checkpoint(step / "trainer", None, None, None, None)
    checkpoint.dcp_dir.mkdir(parents=True)
    old_export = step / "weights"
    other_step = tmp_path / "step_20"
    empty = tmp_path / "empty"
    file = tmp_path / "file"
    for directory in (old_export, other_step / "trainer", empty):
        directory.mkdir(parents=True)
    (old_export / "model.safetensors").write_bytes(b"old")
    file.write_bytes(b"data")

    assert exporter.resolve_output_dir(checkpoint, None, "weights") == old_export
    assert exporter.resolve_output_dir(checkpoint, old_export, "weights") == old_export
    assert exporter.resolve_output_dir(checkpoint, empty, "weights") == empty
    assert exporter.resolve_output_dir(checkpoint, tmp_path / "new", "weights") == tmp_path / "new"
    for path in (other_step, file):
        with pytest.raises(ValueError, match="pass --overwrite"):
            exporter.resolve_output_dir(checkpoint, path, "weights")
        assert exporter.resolve_output_dir(checkpoint, path, "weights", overwrite=True) == path
    assert (other_step / "trainer").is_dir() and file.read_bytes() == b"data"


@pytest.mark.parametrize("existing", [False, True])
def test_publish_export_replaces_complete_directory_with_expected_permissions(tmp_path, exporter, existing):
    output = tmp_path / "weights"
    if existing:
        (output / "assets").mkdir(parents=True)
        (output / "assets" / "old.json").write_text("old")
        (output / "model-old.safetensors").write_bytes(b"old")
        output.chmod(0o705)
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        staging = Path(directory)
        (staging / "assets").mkdir()
        (staging / "assets" / "new.json").write_text("new")
        (staging / "model.safetensors").write_bytes(b"new")
        umask = os.umask(0o027)
        try:
            exporter.publish_export(staging, output)
        finally:
            os.umask(umask)
    assert {path.relative_to(output) for path in output.rglob("*")} == {
        Path("assets"),
        Path("assets/new.json"),
        Path("model.safetensors"),
    }
    assert (output / "model.safetensors").read_bytes() == b"new"
    assert output.stat().st_mode & 0o777 == (0o705 if existing else 0o750)
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("fail_restore", [False, True])
def test_failed_publish_preserves_previous_export(tmp_path, exporter, monkeypatch, fail_restore):
    output = tmp_path / "weights"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"old")
    rename = Path.rename
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        staging = Path(directory)
        (staging / "model.safetensors").write_bytes(b"new")

        def fail_publish(path, target):
            if path == staging or (fail_restore and path.name == "export"):
                raise OSError("rename failed")
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_publish)
        with pytest.raises(OSError, match="rename failed"):
            exporter.publish_export(staging, output)
    if fail_restore:
        backup = next(tmp_path.glob(".weights-backup-*/export"))
        assert (backup / "model.safetensors").read_bytes() == b"old"
    else:
        assert (output / "model.safetensors").read_bytes() == b"old"
        assert list(tmp_path.iterdir()) == [output]
