import pytest
import torch

from prime_rl.trainer import model as trainer_model


def test_converted_snapshot_cache_replaces_unmarked_target(monkeypatch, tmp_path):
    source_path = tmp_path / "snapshot"
    source_path.mkdir()
    target_path = source_path / "prime"
    target_path.mkdir()
    (target_path / "partial").write_text("stale", encoding="utf-8")

    def fake_load_state_dict(path):
        assert path == source_path
        return {"hf.weight": torch.empty(1)}

    def fake_save_state_dict(state_dict, path):
        path.mkdir()
        (path / "payload").write_text(",".join(sorted(state_dict)), encoding="utf-8")

    def fake_load_state_dict_keys(path):
        assert path.name.startswith(".prime.tmp")
        return ["prime.weight"]

    def convert_state_dict(state_dict):
        state_dict.clear()
        state_dict["prime.weight"] = torch.empty(1)

    monkeypatch.setattr(trainer_model, "load_state_dict", fake_load_state_dict)
    monkeypatch.setattr(trainer_model, "save_state_dict", fake_save_state_dict)
    monkeypatch.setattr(trainer_model, "load_state_dict_keys", fake_load_state_dict_keys)

    converted_path = trainer_model._ensure_converted_snapshot(
        source_path,
        "prime",
        convert_state_dict,
        {"prime.weight"},
        is_master=True,
    )

    assert converted_path == target_path
    assert (target_path / trainer_model.CONVERTED_SNAPSHOT_COMPLETE).is_file()
    assert (target_path / "payload").read_text(encoding="utf-8") == "prime.weight"
    assert not (target_path / "partial").exists()


def test_converted_snapshot_cache_rejects_missing_expected_keys(monkeypatch, tmp_path):
    source_path = tmp_path / "snapshot"
    source_path.mkdir()

    def fake_load_state_dict(path):
        assert path == source_path
        return {"hf.weight": torch.empty(1)}

    def fake_save_state_dict(state_dict, path):
        path.mkdir()
        (path / "payload").write_text(",".join(sorted(state_dict)), encoding="utf-8")

    def fake_load_state_dict_keys(path):
        assert path.name.startswith(".prime.tmp")
        return ["unexpected.weight"]

    monkeypatch.setattr(trainer_model, "load_state_dict", fake_load_state_dict)
    monkeypatch.setattr(trainer_model, "save_state_dict", fake_save_state_dict)
    monkeypatch.setattr(trainer_model, "load_state_dict_keys", fake_load_state_dict_keys)

    with pytest.raises(RuntimeError, match="missing expected key"):
        trainer_model._ensure_converted_snapshot(
            source_path,
            "prime",
            lambda state_dict: state_dict,
            {"prime.weight"},
            is_master=True,
        )

    assert not (source_path / "prime").exists()
    assert not list(source_path.glob(".prime.tmp.*"))


def test_converted_snapshot_cache_reuses_marked_target(monkeypatch, tmp_path):
    source_path = tmp_path / "snapshot"
    target_path = source_path / "prime"
    target_path.mkdir(parents=True)
    (target_path / trainer_model.CONVERTED_SNAPSHOT_COMPLETE).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        trainer_model,
        "load_state_dict",
        lambda path: pytest.fail("marked cache should not be reconverted"),
    )

    converted_path = trainer_model._ensure_converted_snapshot(
        source_path,
        "prime",
        lambda state_dict: state_dict,
        {"prime.weight"},
        is_master=True,
    )

    assert converted_path == target_path


def test_converted_snapshot_cache_uses_returned_state_dict(monkeypatch, tmp_path):
    source_path = tmp_path / "snapshot"
    source_path.mkdir()

    def fake_load_state_dict(path):
        assert path == source_path
        return {"hf.weight": torch.empty(1)}

    def fake_save_state_dict(state_dict, path):
        path.mkdir()
        (path / "payload").write_text(",".join(sorted(state_dict)), encoding="utf-8")

    def fake_load_state_dict_keys(path):
        assert path.name.startswith(".prime.tmp")
        return ["prime.weight"]

    monkeypatch.setattr(trainer_model, "load_state_dict", fake_load_state_dict)
    monkeypatch.setattr(trainer_model, "save_state_dict", fake_save_state_dict)
    monkeypatch.setattr(trainer_model, "load_state_dict_keys", fake_load_state_dict_keys)

    trainer_model._ensure_converted_snapshot(
        source_path,
        "prime",
        lambda state_dict: {"prime.weight": state_dict["hf.weight"]},
        {"prime.weight"},
        is_master=True,
    )

    assert (source_path / "prime" / "payload").read_text(encoding="utf-8") == "prime.weight"
