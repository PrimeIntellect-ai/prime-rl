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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorStorageMetadata
from transformers import PretrainedConfig, PreTrainedTokenizerFast

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.models.layers.ngram import LAYOUT_BUFFERS
from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextForCausalLM, Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prime_rl.utils.weights import load_state_dict_keys
from tests.unit.train.models.extra_config_stub import (
    EXTRA_CONFIG,
    EXTRA_CONFIG_ATTR,
    EXTRA_CONFIG_FILE,
    HeadStub,
    load_like_the_trainer,
    register,
    write_stub,
)


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
    if Path("/proc").is_mount():
        with pytest.raises(ValueError, match="mount point"):
            exporter.resolve_output_dir(checkpoint, Path("/proc"), "weights", overwrite=True)


@pytest.mark.parametrize("existing", [False, True])
def test_publish_export_replaces_complete_directory_with_expected_permissions(tmp_path, exporter, existing):
    output = tmp_path / "weights"
    if existing:
        (output / "assets").mkdir(parents=True)
        (output / "assets" / "old.json").write_text("old")
        (output / "model-old.safetensors").write_bytes(b"old")
        output.chmod(0o2705)
    tmp_path.chmod(0o2755)
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
    assert output.stat().st_mode & 0o7777 == (0o2705 if existing else 0o2750)
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_restore", [False, True])
def test_failed_publish_preserves_previous_export(tmp_path, exporter, monkeypatch, fail_restore, error):
    output = tmp_path / "weights"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"old")
    rename = Path.rename
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        staging = Path(directory)
        (staging / "model.safetensors").write_bytes(b"new")

        def fail_publish(path, target):
            if path == staging or (fail_restore and path.name == "export"):
                raise error("rename failed")
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_publish)
        with pytest.raises(error, match="rename failed"):
            exporter.publish_export(staging, output)
    if fail_restore:
        backup = next(tmp_path.glob(".weights-backup-*/export"))
        assert (backup / "model.safetensors").read_bytes() == b"old"
    else:
        assert (output / "model.safetensors").read_bytes() == b"old"
        assert list(tmp_path.iterdir()) == [output]


def test_interrupt_after_moving_the_previous_export_restores_it(tmp_path, exporter, monkeypatch):
    output = tmp_path / "weights"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"old")
    staging = tmp_path / "staging"
    staging.mkdir()
    rename = Path.rename

    def interrupt_after_move(path, target):
        rename(path, target)
        if path == output:
            raise KeyboardInterrupt

    monkeypatch.setattr(Path, "rename", interrupt_after_move)
    with pytest.raises(KeyboardInterrupt):
        exporter.publish_export(staging, output)
    assert (output / "model.safetensors").read_bytes() == b"old"
    assert sorted(tmp_path.iterdir()) == [staging, output]


def test_publish_succeeds_when_the_previous_export_cannot_be_removed(tmp_path, exporter, monkeypatch):
    output = tmp_path / "weights"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"old")
    monkeypatch.setattr(exporter.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("busy")))
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"new")
    exporter.publish_export(staging, output)
    assert (output / "model.safetensors").read_bytes() == b"new"
    assert (next(tmp_path.glob(".weights-backup-*/export")) / "model.safetensors").read_bytes() == b"old"


def test_model_assets_of_a_model_that_cannot_generate(tmp_path, exporter):
    """NemotronH does not inherit GenerationMixin, so it has no generation config: the export writes the rest."""
    model_dir = tmp_path / "model"
    NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        expand=2,
        mamba_num_heads=2,
        mamba_head_dim=16,
        ssm_state_size=16,
        n_groups=1,
        hybrid_override_pattern="M*",
    ).save_pretrained(model_dir)
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "a": 1}, unk_token="<eos>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>").save_pretrained(model_dir)
    model_config = ModelConfig(name=str(model_dir), attn="flash_attention_2")
    model = get_model(model_config, device=torch.device("meta"), dtype=torch.float32)
    assert not hasattr(model, "generation_config")
    output = tmp_path / "export"
    exporter.save_model_assets(model, model_config, TokenizerConfig(), output)
    assert (output / "config.json").is_file() and (output / "tokenizer.json").is_file()
    assert not (output / "generation_config.json").exists()


def test_model_assets_keep_a_config_key_the_config_class_does_not_define(tmp_path, exporter):
    """The inference server may read such a key: vLLM serves a Qwen3.5 checkpoint with ``modeling_overrides`` as
    another class."""
    model_dir, output = tmp_path / "model", tmp_path / "export"
    write_stub(model_dir, extra_config=None)
    overrides = {"attention": {"full_attention": "block_bidirectional"}, "readout": "slot_lm"}
    saved = json.loads((model_dir / "config.json").read_text())
    (model_dir / "config.json").write_text(json.dumps({**saved, "modeling_overrides": overrides}))
    model_config = ModelConfig(name=str(model_dir), attn="flash_attention_2")
    model = get_model(model_config, device=torch.device("meta"), dtype=torch.float32)
    exporter.save_model_assets(model, model_config, TokenizerConfig(), output)
    assert json.loads((output / "config.json").read_text())["modeling_overrides"] == overrides


def test_a_run_from_a_checkpoint_exports_without_its_source_step_dir(tmp_path, exporter, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    source = (tmp_path / "deleted-run" / "checkpoints" / "step_5").as_posix()
    run_dir = tmp_path / "run"
    config_dir = run_dir / "configs" / "latest" / "resolved"
    config_dir.mkdir(parents=True)
    run_config = {"model": {"name": source, "init": {"type": "pretrained"}}, "tokenizer": {"name": source}}
    (config_dir / "trainer.json").write_text(json.dumps(run_config))
    step_dir = run_dir / "checkpoints" / "step_1"
    Qwen3MoeConfig().save_pretrained(step_dir)

    model_config, tokenizer_config = exporter.resolve_run_configs(step_dir)
    assert model_config.name == tokenizer_config.name == str(step_dir)


def test_convert_exports_a_run_checkpoint_end_to_end(tmp_path, exporter, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    model_dir = tmp_path / "model"
    Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=32,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
    ).save_pretrained(model_dir)
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "a": 1}, unk_token="<eos>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>").save_pretrained(model_dir)
    run_dir = tmp_path / "run"
    step_dir = run_dir / "checkpoints" / "step_1"
    config_dir = run_dir / "configs" / "latest" / "resolved"
    config_dir.mkdir(parents=True)
    run_config = {"model": {"name": str(model_dir)}, "tokenizer": {}}
    (config_dir / "trainer.json").write_text(json.dumps(run_config))
    # The run config keeps attn=auto, which the exporter must resolve without a GPU.
    model_config = ModelConfig(**run_config["model"], attn="flash_attention_2")
    model = get_model(model_config, device=torch.device("meta"), dtype=torch.float32)
    state = {
        name: torch.randn(tensor.shape) if tensor.is_floating_point() else torch.zeros(tensor.shape, dtype=tensor.dtype)
        for name, tensor in model.state_dict().items()
    }
    dcp.save({"app": {"model": state}}, checkpoint_id=step_dir / "trainer", no_dist=True)

    def read_export(directory):
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        tensors = {name: tensor for file in set(index.values()) for name, tensor in load_file(directory / file).items()}
        assert tensors.keys() == index.keys()
        return tensors, json.loads((directory / "config.json").read_text())

    for _ in range(2):  # the second run replaces the first export
        output = exporter.convert(step_dir, workers=2)
        assert output == step_dir / "weights" and (output / "tokenizer.json").is_file()
        assert [path.name for path in step_dir.iterdir() if path.name.startswith(".")] == []
        tensors, config = read_export(output)
        assert config["dtype"] == "bfloat16" and "quantization_config" not in config
        assert sum(tensor.numel() for tensor in tensors.values()) == sum(tensor.numel() for tensor in state.values())
        embedding = state["model.embed_tokens.weight"].bfloat16()
        assert torch.equal(tensors["model.embed_tokens.weight"].view(torch.uint8), embedding.view(torch.uint8))

    fp8 = importlib.import_module("convert_dcp_to_fp8")
    output = fp8.convert(step_dir, block_size=2, workers=2)
    assert output == step_dir / "weights-FP8"
    quantized, config = read_export(output)
    quantization = config["quantization_config"]
    assert quantization["quant_method"] == "fp8" and quantization["weight_block_size"] == [2, 2]
    scaled = {name.removesuffix(".weight_scale_inv") for name in quantized if name.endswith("_scale_inv")}
    assert scaled and scaled.isdisjoint(quantization["modules_to_not_convert"])
    assert all(quantized[f"{name}.weight"].dtype == torch.float8_e4m3fn for name in scaled)
    assert all(quantized[f"{name}.weight"].dtype == torch.bfloat16 for name in quantization["modules_to_not_convert"])
    assert quantized.keys() - {f"{name}.weight_scale_inv" for name in scaled} == tensors.keys()


def save_stub_run(tmp_path) -> tuple[Path, dict[str, torch.Tensor]]:
    """A run of the registered stub from ``tmp_path / "model"``: its step directory and its weights."""
    model_dir = tmp_path / "model"
    write_stub(model_dir)
    config_dir = tmp_path / "run" / "configs" / "latest" / "resolved"
    config_dir.mkdir(parents=True)
    (config_dir / "sft.json").write_text(json.dumps({"model": {"name": str(model_dir)}, "tokenizer": {}}))
    model = get_model(ModelConfig(name=str(model_dir), attn="flash_attention_2"), device=torch.device("meta"))
    assert isinstance(model, HeadStub)
    torch.manual_seed(0)
    state = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    step_dir = tmp_path / "run" / "checkpoints" / "step_1"
    dcp.save({"app": {"model": state}}, checkpoint_id=step_dir / "trainer", no_dist=True)
    return step_dir, state


def test_the_export_writes_export_files_and_the_extra_config_file(tmp_path, exporter, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    register(monkeypatch, HeadStub)
    step_dir, state = save_stub_run(tmp_path)
    output = exporter.convert(step_dir)

    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(state) - {"head.weight"} and "head.safetensors" not in index.values()
    head = load_file(output / "head.safetensors")
    assert list(head) == ["weight"] and torch.equal(head["weight"], state["head.weight"].bfloat16())
    assert json.loads((output / EXTRA_CONFIG_FILE).read_text()) == EXTRA_CONFIG
    config = json.loads((output / "config.json").read_text())
    assert EXTRA_CONFIG_ATTR not in config and config["architectures"] == ["Qwen3ForCausalLM"]

    # The export loads back as the stub, with every weight under the model's names
    model = get_model(ModelConfig(name=str(output), attn="flash_attention_2"), device=torch.device("meta"))
    assert type(model) is HeadStub
    assert set(load_state_dict_keys(output, HeadStub.export_files)) == set(model.state_dict())
    load_like_the_trainer(model, output, monkeypatch)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[name].bfloat16()), name

    # The content is taken off a copy of the model's config
    checkpoint = exporter.open_checkpoint(step_dir)
    exporter.save_model_assets(checkpoint.model, checkpoint.model_config, checkpoint.tokenizer_config, tmp_path / "x")
    assert getattr(checkpoint.model.config, EXTRA_CONFIG_ATTR) == EXTRA_CONFIG
    assert (tmp_path / "x" / "config.json").read_text() == (output / "config.json").read_text()


def test_both_fp8_tools_refuse_a_model_with_export_files(tmp_path, exporter, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    register(monkeypatch, HeadStub)
    step_dir, _ = save_stub_run(tmp_path)
    with pytest.raises(ValueError, match="HeadStub has export files, which FP8 export does not support"):
        importlib.import_module("convert_dcp_to_fp8").convert(step_dir, block_size=2)
    output = exporter.convert(step_dir)
    with pytest.raises(ValueError, match="holds HeadStub, whose export files FP8 conversion does not support"):
        importlib.import_module("convert_bf16_to_fp8").convert(output, tmp_path / "fp8")
    assert not (tmp_path / "fp8").exists()


def test_the_export_refuses_a_checkpoint_the_model_lacks(tmp_path, exporter, monkeypatch):
    monkeypatch.delenv("PRL_ATTEMPT_CONFIG_DIR", raising=False)
    register(monkeypatch, HeadStub)
    step_dir, _ = save_stub_run(tmp_path)
    # Without the extra config file, model.name builds the stock model, which has no head
    (tmp_path / "model" / EXTRA_CONFIG_FILE).unlink()
    with pytest.raises(ValueError, match=r"Qwen3ForCausalLM, built from .*, lacks 1 tensors of the checkpoint \(head"):
        exporter.open_checkpoint(step_dir)
