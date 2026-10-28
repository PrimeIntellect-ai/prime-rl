import asyncio
import copy
import errno
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import prime_rl.orchestrator.hf_export as hf_export
import prime_rl.utils.weights as weight_utils
from prime_rl.orchestrator.hf_export import export_best_and_latest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(path: Path) -> dict[str, str]:
    return {str(file.relative_to(path)): _sha256(file) for file in sorted(path.rglob("*")) if file.is_file()}


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = ["[UNK]", "[EOS]", "[PAD]", "[BOS]", "HF", "export", "smoke", "hello", "world"]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        bos_token="[BOS]",
    )


def _tiny_model(source_dir: Path) -> LlamaForCausalLM:
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=9,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=3,
        eos_token_id=1,
        pad_token_id=2,
    )
    model = LlamaForCausalLM(config)
    model.config.save_pretrained(source_dir)
    model.generation_config.save_pretrained(source_dir)
    _tiny_tokenizer().save_pretrained(source_dir)
    return model


def _write_dcp(run_dir: Path, step: int, model: torch.nn.Module, *, optimizer_numel: int = 0) -> None:
    dcp_dir = run_dir / "checkpoints" / f"step_{step}" / "trainer"
    dcp_dir.parent.mkdir(parents=True, exist_ok=True)
    app_state = {"model": model.state_dict()}
    if optimizer_numel:
        app_state["optimizers"] = {
            "state": {0: {"exp_avg": torch.ones(optimizer_numel), "exp_avg_sq": torch.ones(optimizer_numel)}},
            "param_groups": [],
        }
    dcp_save({"app": app_state}, checkpoint_id=dcp_dir)


@pytest.fixture
def fake_run(tmp_path: Path) -> tuple[Path, LlamaForCausalLM]:
    run_dir = tmp_path / "run"
    source_dir = tmp_path / "tiny-model"
    source_dir.mkdir()
    model = _tiny_model(source_dir)
    config_dir = run_dir / "configs" / "resolved"
    config_dir.mkdir(parents=True)
    (config_dir / "trainer.json").write_text(
        json.dumps({"model": {"name": str(source_dir)}, "tokenizer": {"name": str(source_dir)}})
    )
    return run_dir, model


def test_exports_sharded_best_then_latest_and_reuses_immutable_steps(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    latest_model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in latest_model.parameters():
            parameter.add_(0.01)
    _write_dcp(run_dir, 40, latest_model)
    source_hashes = _tree_hashes(run_dir / "checkpoints")

    # Exercise multi-shard index validation without a large fake model.
    split_into_shards = weight_utils.split_torch_state_dict_into_shards

    def tiny_shards(state_dict, *, filename_pattern, max_shard_size="5GB"):
        return split_into_shards(state_dict, filename_pattern=filename_pattern, max_shard_size=256)

    monkeypatch.setattr(weight_utils, "split_torch_state_dict_into_shards", tiny_shards)

    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40) is True

    exports = run_dir / "exports"
    assert os.readlink(exports / "best") == "step_20"
    assert os.readlink(exports / "latest") == "step_40"
    complete = json.loads((exports / "COMPLETE.json").read_text())
    assert complete["status"] == "complete"
    assert complete["best"]["step"] == 20
    assert complete["latest"]["step"] == 40

    for step in (20, 40):
        step_dir = exports / f"step_{step}"
        manifest = json.loads((step_dir / "EXPORT.json").read_text())
        index = json.loads((step_dir / "model.safetensors.index.json").read_text())
        assert len(manifest["shards"]) > 1
        assert set(index["weight_map"]) == set(manifest["tensors"])
        assert set(index["weight_map"].values()) == {shard["path"] for shard in manifest["shards"]}
        for shard in manifest["shards"]:
            assert _sha256(step_dir / shard["path"]) == shard["sha256"]
        assert all(tensor["dtype"] == "torch.bfloat16" for tensor in manifest["tensors"].values())

    assert _tree_hashes(run_dir / "checkpoints") == source_hashes
    manifest_hashes = {step: _sha256(exports / f"step_{step}" / "EXPORT.json") for step in (20, 40)}

    def unexpected_reconversion(**kwargs):
        raise AssertionError("A valid immutable by-step export was converted again")

    monkeypatch.setattr(hf_export, "_convert_checkpoint", unexpected_reconversion)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40) is True
    assert {step: _sha256(exports / f"step_{step}" / "EXPORT.json") for step in (20, 40)} == manifest_hashes


def test_latest_only_preserves_tokenizer_trust_and_removes_stale_best(
    fake_run: tuple[Path, LlamaForCausalLM],
):
    run_dir, model = fake_run
    config_path = run_dir / "configs" / "resolved" / "trainer.json"
    run_config = json.loads(config_path.read_text())
    run_config["model"]["trust_remote_code"] = False
    run_config["tokenizer"]["trust_remote_code"] = True
    config_path.write_text(json.dumps(run_config))
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    _write_dcp(run_dir, 40, model)

    assert export_best_and_latest(output_dir=run_dir, best_step=None, latest_step=40) is True

    exports = run_dir / "exports"
    assert not (exports / "best").exists()
    assert not (exports / "best").is_symlink()
    assert os.readlink(exports / "latest") == "step_40"
    complete = json.loads((exports / "COMPLETE.json").read_text())
    assert complete["best"] is None
    assert complete["latest"]["step"] == 40
    manifest = json.loads((exports / "step_40" / "EXPORT.json").read_text())
    assert manifest["trust_remote_code"] is False
    assert manifest["tokenizer_trust_remote_code"] is True


def test_equal_best_and_latest_convert_once(fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    convert = hf_export._convert_checkpoint
    calls = []

    def counted_convert(**kwargs):
        calls.append(kwargs["step"])
        return convert(**kwargs)

    monkeypatch.setattr(hf_export, "_convert_checkpoint", counted_convert)

    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    assert calls == [20]
    exports = run_dir / "exports"
    assert os.readlink(exports / "best") == "step_20"
    assert os.readlink(exports / "latest") == "step_20"
    assert len(json.loads((exports / "COMPLETE.json").read_text())["exports"]) == 1


def test_invalid_latest_fails_closed_and_dcp_remains_retryable(fake_run: tuple[Path, LlamaForCausalLM]):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    invalid_model = copy.deepcopy(model)
    with torch.no_grad():
        invalid_model.lm_head.weight.view(-1)[0] = torch.nan
    _write_dcp(run_dir, 40, invalid_model)
    source_hashes = _tree_hashes(run_dir / "checkpoints")

    with pytest.raises(RuntimeError, match="Non-finite"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40)

    exports = run_dir / "exports"
    assert not (exports / "COMPLETE.json").exists()
    assert not (exports / "best").exists()
    assert not (exports / "latest").exists()
    assert (exports / "step_20" / "EXPORT.json").is_file()
    assert not (exports / "step_40").exists()
    assert not list(exports.glob("*.tmp"))
    assert not list(exports.glob(".*.tmp"))
    assert _tree_hashes(run_dir / "checkpoints") == source_hashes

    _write_dcp(run_dir, 40, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40) is True
    assert os.readlink(exports / "best") == "step_20"
    assert os.readlink(exports / "latest") == "step_40"


def test_early_rerun_failure_preserves_previous_complete(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    previous_complete = (exports / "COMPLETE.json").read_bytes()
    _write_dcp(run_dir, 40, model)
    conversion_attempts = []

    def transient_conversion_failure(**kwargs):
        conversion_attempts.append(kwargs["step"])
        raise RuntimeError("transient conversion failure")

    monkeypatch.setattr(hf_export, "_convert_checkpoint", transient_conversion_failure)

    with pytest.raises(RuntimeError, match="transient conversion failure"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40)

    assert conversion_attempts == [40]
    assert (exports / "COMPLETE.json").read_bytes() == previous_complete
    assert os.readlink(exports / "best") == "step_20"
    assert os.readlink(exports / "latest") == "step_20"


def test_preflight_failure_preserves_complete_and_counts_only_missing_unique_exports(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    previous_complete = (exports / "COMPLETE.json").read_bytes()
    _write_dcp(run_dir, 40, model, optimizer_numel=1_000_000)
    _write_dcp(run_dir, 60, model)
    source_sizes = {
        step: sum(
            path.stat().st_size for path in (run_dir / "checkpoints" / f"step_{step}").rglob("*") if path.is_file()
        )
        for step in (20, 40, 60)
    }
    model_sizes = {step: hf_export._model_checkpoint_bytes(run_dir / "checkpoints", step) for step in (20, 40, 60)}
    assert model_sizes[40] < source_sizes[40] // 100
    preflight_calls = []

    def fail_preflight(path, *, required_bytes):
        preflight_calls.append((path, required_bytes))
        raise OSError("insufficient export staging space")

    def unexpected_conversion(**kwargs):
        raise AssertionError("conversion ran after a failed storage preflight")

    monkeypatch.setattr(hf_export, "preflight_checkpoint_storage", fail_preflight)
    monkeypatch.setattr(hf_export, "_convert_checkpoint", unexpected_conversion)

    cases = [
        (20, 40, model_sizes[40]),  # existing immutable step 20 needs no staging
        (40, 60, model_sizes[40] + model_sizes[60]),
        (40, 40, model_sizes[40]),  # equal BEST/LATEST is one unique export
    ]
    for best_step, latest_step, expected_bytes in cases:
        with pytest.raises(OSError, match="insufficient export staging space"):
            export_best_and_latest(output_dir=run_dir, best_step=best_step, latest_step=latest_step)
        assert (exports / "COMPLETE.json").read_bytes() == previous_complete
        assert os.readlink(exports / "best") == "step_20"
        assert os.readlink(exports / "latest") == "step_20"

    assert preflight_calls == [(exports, expected_bytes) for _, _, expected_bytes in cases]


def test_pointer_phase_failure_removes_previous_complete(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    atomic_symlink = hf_export._atomic_symlink
    pointer_attempts = []

    def fail_second_pointer(target, pointer, parent):
        pointer_attempts.append(pointer.name)
        if pointer.name == "latest":
            raise OSError("pointer write failed")
        atomic_symlink(target, pointer, parent)

    monkeypatch.setattr(hf_export, "_atomic_symlink", fail_second_pointer)

    with pytest.raises(OSError, match="pointer write failed"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)

    assert pointer_attempts == ["best", "latest"]
    assert not (exports / "COMPLETE.json").exists()


def test_corrupted_published_export_invalidates_complete(fake_run: tuple[Path, LlamaForCausalLM]):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    shard = next((exports / "step_20").glob("*.safetensors"))
    shard.write_bytes(shard.read_bytes() + b"corruption")

    with pytest.raises(RuntimeError, match="Asset hash mismatch"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)

    assert not (exports / "COMPLETE.json").exists()


@pytest.mark.parametrize("relative_path", ["unexpected.txt", "nested/EXPORT.json"])
def test_unlisted_published_asset_invalidates_complete(fake_run: tuple[Path, LlamaForCausalLM], relative_path: str):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    asset = exports / "step_20" / relative_path
    asset.parent.mkdir(exist_ok=True)
    asset.write_text("tamper")

    with pytest.raises(RuntimeError, match="Published asset set mismatch"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)

    assert not (exports / "COMPLETE.json").exists()


def test_symlinked_published_asset_invalidates_complete(fake_run: tuple[Path, LlamaForCausalLM]):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    asset = exports / "step_20" / "tokenizer.json"
    outside = run_dir.parent / "outside-tokenizer.json"
    asset.rename(outside)
    asset.symlink_to(outside)

    with pytest.raises(RuntimeError, match="Symlinked assets are not allowed"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)

    assert not (exports / "COMPLETE.json").exists()


def test_manifest_trust_tamper_invalidates_complete(fake_run: tuple[Path, LlamaForCausalLM]):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    manifest_path = exports / "step_20" / "EXPORT.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trust_remote_code"] = True
    manifest_path.write_text(json.dumps(manifest))
    complete_path = exports / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["exports"][0]["manifest_sha256"] = _sha256(manifest_path)
    complete_path.write_text(json.dumps(complete))

    with pytest.raises(RuntimeError, match="trust_remote_code does not match resolved config"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)

    assert not (exports / "COMPLETE.json").exists()


def test_manifest_tamper_invalidates_complete(fake_run: tuple[Path, LlamaForCausalLM]):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    manifest_path = exports / "step_20" / "EXPORT.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["step"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="Published manifest hash mismatch"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)
    assert not (exports / "COMPLETE.json").exists()


def test_json_reader_rejects_invalid_utf8_and_duplicate_keys(tmp_path: Path):
    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"\xff")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key": 1, "key": 2}')

    with pytest.raises(RuntimeError, match="valid JSON"):
        hf_export._read_json(invalid_utf8)
    with pytest.raises(RuntimeError, match="valid JSON"):
        hf_export._read_json(duplicate)


def test_concurrent_exports_are_serialized_and_convert_once(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    convert = hf_export._convert_checkpoint
    guard = threading.Lock()
    calls = []
    active = 0
    max_active = 0

    def measured_conversion(**kwargs):
        nonlocal active, max_active
        with guard:
            calls.append(kwargs["step"])
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        try:
            return convert(**kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(hf_export, "_convert_checkpoint", measured_conversion)

    def run_export(_):
        return export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run_export, range(2))) == [True, True]
    assert calls == [20]
    assert max_active == 1


def test_eexist_winner_is_revalidated_and_published(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    exports = run_dir / "exports"
    original_rename = Path.rename

    def racing_rename(self, target):
        if self.parent == exports and target == exports / "step_20":
            shutil.copytree(self, target)
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", racing_rename)

    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    assert (exports / "COMPLETE.json").is_file()
    assert not list(exports.glob(".step_20.*.tmp"))


def test_eexist_winner_source_change_blocks_publication(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    exports = run_dir / "exports"
    original_rename = Path.rename
    revalidate = hf_export._revalidate_published_export

    def racing_rename(self, target):
        if self.parent == exports and target == exports / "step_20":
            shutil.copytree(self, target)
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)
        return original_rename(self, target)

    def revalidate_then_change_source(**kwargs):
        manifest = revalidate(**kwargs)
        payload = next((run_dir / "checkpoints" / "step_20" / "trainer").glob("*.distcp"))
        with payload.open("ab") as handle:
            handle.write(b"source changed during winner validation")
        return manifest

    monkeypatch.setattr(Path, "rename", racing_rename)
    monkeypatch.setattr(hf_export, "_revalidate_published_export", revalidate_then_change_source)

    with pytest.raises(RuntimeError, match="changed while exporting step 20"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)
    assert not (exports / "COMPLETE.json").exists()
    assert not (exports / "best").exists()
    assert not (exports / "latest").exists()


def test_cancellation_cleans_staging_preserves_complete_and_releases_lock(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True
    exports = run_dir / "exports"
    previous_complete = (exports / "COMPLETE.json").read_bytes()
    _write_dcp(run_dir, 40, model)
    convert = hf_export._convert_checkpoint

    def cancel_conversion(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(hf_export, "_convert_checkpoint", cancel_conversion)
    with pytest.raises(asyncio.CancelledError):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40)

    assert (exports / "COMPLETE.json").read_bytes() == previous_complete
    assert not list(exports.glob(".step_40.*.tmp"))

    monkeypatch.setattr(hf_export, "_convert_checkpoint", convert)
    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=40) is True


def test_dcp_payload_change_during_conversion_blocks_publication(
    fake_run: tuple[Path, LlamaForCausalLM], monkeypatch: pytest.MonkeyPatch
):
    run_dir, model = fake_run
    _write_dcp(run_dir, 20, model)
    convert = hf_export._convert_checkpoint

    def convert_then_change_source(**kwargs):
        manifest = convert(**kwargs)
        payload = next(kwargs["dcp_dir"].glob("*.distcp"))
        with payload.open("ab") as handle:
            handle.write(b"changed during conversion")
        return manifest

    monkeypatch.setattr(hf_export, "_convert_checkpoint", convert_then_change_source)

    with pytest.raises(RuntimeError, match="changed while exporting step 20"):
        export_best_and_latest(output_dir=run_dir, best_step=None, latest_step=20)

    exports = run_dir / "exports"
    assert not (exports / "step_20").exists()
    assert not (exports / "latest").exists()
    assert not (exports / "COMPLETE.json").exists()
    assert not list(exports.glob(".*.tmp"))


def test_provenance_tracks_template_tokenizer_code_and_converter_content(
    fake_run: tuple[Path, LlamaForCausalLM],
):
    run_dir, model = fake_run
    config_path = run_dir / "configs" / "resolved" / "trainer.json"
    run_config = json.loads(config_path.read_text())
    template = "{{ messages }}"
    run_config["tokenizer"]["chat_template"] = template
    config_path.write_text(json.dumps(run_config))
    source_dir = Path(run_config["model"]["name"])
    code_file = source_dir / "custom_code.py"
    code_file.write_text("# converter input v1")
    _write_dcp(run_dir, 20, model)

    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True

    exports = run_dir / "exports"
    manifest_path = exports / "step_20" / "EXPORT.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest["provenance"]
    assert "run_config" not in provenance
    assert len(provenance["run_config_sha256"]) == 64
    assert len(provenance["tokenizer_assets_sha256"]) == 64
    assert provenance["chat_template"]["content"] == template
    assert provenance["code_sha256"] == hf_export._hash_json({"custom_code.py": _sha256(code_file)})
    assert provenance["converter"]["version"] == hf_export._CONVERTER_VERSION
    assert set(provenance["converter"]["content_sha256"]) == {"hf_export.py", "weights.py"}
    asset_names = {record["path"] for record in manifest["assets"]}
    assert "custom_code.py" in asset_names
    assert "chat_template.jinja" in asset_names
    complete = json.loads((exports / "COMPLETE.json").read_text())
    assert complete["exports"][0]["manifest_sha256"] == _sha256(manifest_path)

    code_file.write_text("# converter input v2")
    with pytest.raises(RuntimeError, match="provenance does not match"):
        export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20)
    assert not (exports / "COMPLETE.json").exists()


def test_nemotron_h_dcp_layout_exports_through_builtin_hf_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise Mamba, MoE, latent projection, and attention conversion on CPU."""
    # Import only NemotronH, not every PrimeRL architecture. Kernel functions are
    # needed to construct the training model but are never called by this test.
    monkeypatch.setitem(sys.modules, "prime_kernels", types.ModuleType("prime_kernels"))
    import prime_rl.trainer

    models_package = types.ModuleType("prime_rl.trainer.models")
    models_package.__path__ = [str(Path(next(iter(prime_rl.trainer.__path__))) / "models")]
    models_package.__package__ = "prime_rl.trainer.models"
    monkeypatch.setitem(sys.modules, "prime_rl.trainer.models", models_package)

    def unused_kernel(*args, **kwargs):
        raise AssertionError("A kernel stub was called")

    stub_modules = {
        "fla": {},
        "fla.modules": {},
        "fla.modules.conv": {"causal_conv1d": unused_kernel},
        "fla.ops": {},
        "fla.ops.utils": {"prepare_sequence_ids": unused_kernel},
        "mamba_ssm": {},
        "mamba_ssm.ops": {},
        "mamba_ssm.ops.triton": {},
        "mamba_ssm.ops.triton.ssd_combined": {"mamba_chunk_scan_combined": unused_kernel},
    }
    for name, attributes in stub_modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=9,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        intermediate_size=32,
        expand=2,
        mamba_num_heads=4,
        mamba_head_dim=4,
        ssm_state_size=4,
        n_groups=1,
        conv_kernel=4,
        chunk_size=8,
        n_routed_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=8,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        hybrid_override_pattern="ME*E",
        attn_implementation="flash_attention_2",
    )
    model = NemotronHForCausalLM.from_config(config)
    assert "model.layers.1.mlp.experts.up_proj" in model.state_dict()

    run_dir = tmp_path / "nemotron-run"
    source_dir = tmp_path / "nemotron-model"
    source_dir.mkdir()
    model.config.save_pretrained(source_dir)
    _tiny_tokenizer().save_pretrained(source_dir)
    config_dir = run_dir / "configs" / "resolved"
    config_dir.mkdir(parents=True)
    (config_dir / "trainer.json").write_text(
        json.dumps({"model": {"name": str(source_dir)}, "tokenizer": {"name": str(source_dir)}})
    )
    _write_dcp(run_dir, 20, model)

    # The source has no original HF weights. Run the production custom
    # from_config loader, then remove import-only stubs after the real DCP load
    # so HF validation selects its built-in PyTorch Mamba implementation.
    dcp_load = hf_export.dcp_load

    def load_dcp_and_enable_hf_fallback(*args, **kwargs):
        result = dcp_load(*args, **kwargs)
        for name in stub_modules:
            monkeypatch.delitem(sys.modules, name, raising=False)
        return result

    monkeypatch.setattr(hf_export, "dcp_load", load_dcp_and_enable_hf_fallback)

    assert export_best_and_latest(output_dir=run_dir, best_step=20, latest_step=20) is True

    step_dir = run_dir / "exports" / "step_20"
    assert (step_dir / "generation_config.json").is_file()
    manifest = json.loads((step_dir / "EXPORT.json").read_text())
    expected_hf_keys = {
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.A_log",
        "backbone.layers.1.mixer.experts.0.up_proj.weight",
        "backbone.layers.1.mixer.fc1_latent_proj.weight",
        "backbone.layers.2.mixer.q_proj.weight",
    }
    assert expected_hf_keys <= set(manifest["tensors"])
    assert not any(".mlp.experts." in key for key in manifest["tensors"])

    original = model.state_dict()
    with safe_open(step_dir / "model.safetensors", framework="pt", device="cpu") as shard:
        for expert in range(config.n_routed_experts):
            actual = shard.get_tensor(f"backbone.layers.1.mixer.experts.{expert}.up_proj.weight")
            expected = original["model.layers.1.mlp.experts.up_proj"][expert].to(torch.bfloat16)
            assert torch.equal(actual, expected)
        latent = shard.get_tensor("backbone.layers.1.mixer.fc1_latent_proj.weight")
        assert torch.equal(latent, original["model.layers.1.mlp.fc1_latent_proj.weight"].to(torch.bfloat16))
        mamba = shard.get_tensor("backbone.layers.0.mixer.in_proj.weight")
        assert torch.equal(mamba, original["model.layers.0.mamba.in_proj.weight"].to(torch.bfloat16))
        for name in ("A_log", "D"):
            actual = shard.get_tensor(f"backbone.layers.0.mixer.{name}")
            assert actual.dtype is torch.float32
            assert torch.equal(actual, original[f"model.layers.0.mamba.{name}"])

    hf_config = AutoConfig.from_pretrained(step_dir, local_files_only=True)
    assert hf_config.model_type == "nemotron_h"
    assert type(hf_config).__module__ == "transformers.models.nemotron_h.configuration_nemotron_h"
    hf_model, loading_info = AutoModelForCausalLM.from_pretrained(
        step_dir,
        local_files_only=True,
        output_loading_info=True,
        dtype="auto",
        attn_implementation="eager",
    )
    assert type(hf_model).__module__ == "transformers.models.nemotron_h.modeling_nemotron_h"
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
