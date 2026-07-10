"""Unit tests for the standalone HF -> PrimeRL converter and its atomic write."""

import torch

from prime_rl.trainer.convert import convert_snapshot_to_prime
from prime_rl.trainer.weights import atomic_save_state_dict, is_state_dict_complete, load_state_dict


def test_atomic_save_roundtrips(tmp_path):
    save_dir = tmp_path / "out"
    state_dict = {"a": torch.arange(4, dtype=torch.float32), "b": torch.ones(2, 2)}
    expected = {key: tensor.clone() for key, tensor in state_dict.items()}

    atomic_save_state_dict(state_dict, save_dir)

    assert save_dir.is_dir()
    assert not list(tmp_path.glob(".out.tmp-*"))
    assert is_state_dict_complete(save_dir)
    loaded = load_state_dict(save_dir)
    assert set(loaded) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(loaded[key], tensor)


def test_atomic_save_repairs_incomplete_destination(tmp_path):
    save_dir = tmp_path / "out"
    save_dir.mkdir()
    (save_dir / "partial.bin").write_text("junk")

    atomic_save_state_dict({"a": torch.zeros(1)}, save_dir)

    assert is_state_dict_complete(save_dir)
    assert not (save_dir / "partial.bin").exists()


def test_atomic_save_accepts_complete_concurrent_winner(tmp_path):
    save_dir = tmp_path / "out"
    atomic_save_state_dict({"winner": torch.ones(1)}, save_dir)

    published = atomic_save_state_dict({"loser": torch.zeros(1)}, save_dir)

    assert published is False
    assert set(load_state_dict(save_dir)) == {"winner"}


class _FakeConfig:
    pass


class _FakeCausalLM:
    """Records format-classifier answers so tests can drive each branch."""

    prime = False
    hf = False
    converted = False

    @classmethod
    def is_prime_state_dict(cls, keys):
        return cls.prime

    @classmethod
    def is_hf_state_dict(cls, keys):
        return cls.hf

    @classmethod
    def convert_to_prime(cls, state_dict):
        cls.converted = True


def _patch_common(monkeypatch, *, cls, keys, save_calls=None):
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: _FakeConfig(), raising=False)
    # convert.py imports these lazily from their source modules, so patch there.
    monkeypatch.setattr("prime_rl.trainer.models.get_custom_causal_lm_cls", lambda config: cls)
    monkeypatch.setattr("prime_rl.trainer.weights.load_state_dict_keys", lambda snapshot: keys)
    monkeypatch.setattr("prime_rl.trainer.weights.load_state_dict", lambda snapshot: {"w": torch.zeros(1)})
    recorded = save_calls if save_calls is not None else []
    monkeypatch.setattr(
        "prime_rl.trainer.weights.atomic_save_state_dict",
        lambda state_dict, save_dir, **kw: recorded.append((state_dict, save_dir)),
    )


def test_exists_short_circuits(tmp_path):
    (tmp_path / "prime").mkdir()
    assert convert_snapshot_to_prime(tmp_path) == "exists"


def test_unsupported_architecture(tmp_path, monkeypatch):
    _patch_common(monkeypatch, cls=None, keys=["x"])
    assert convert_snapshot_to_prime(tmp_path) == "unsupported"


def test_no_safetensors(tmp_path, monkeypatch):
    _patch_common(monkeypatch, cls=_FakeCausalLM, keys=[])
    assert convert_snapshot_to_prime(tmp_path) == "no-safetensors"


def test_already_prime(tmp_path, monkeypatch):
    cls = type("PrimeCls", (_FakeCausalLM,), {"prime": True})
    _patch_common(monkeypatch, cls=cls, keys=["x"])
    assert convert_snapshot_to_prime(tmp_path) == "already-prime"


def test_not_hf(tmp_path, monkeypatch):
    cls = type("NeitherCls", (_FakeCausalLM,), {"prime": False, "hf": False})
    _patch_common(monkeypatch, cls=cls, keys=["x"])
    assert convert_snapshot_to_prime(tmp_path) == "not-hf"


def test_converted_writes_prime(tmp_path, monkeypatch):
    cls = type("HfCls", (_FakeCausalLM,), {"prime": False, "hf": True, "converted": False})
    save_calls: list = []
    _patch_common(monkeypatch, cls=cls, keys=["x"], save_calls=save_calls)

    assert convert_snapshot_to_prime(tmp_path) == "converted"
    assert cls.converted is True
    assert len(save_calls) == 1
    _, save_dir = save_calls[0]
    assert save_dir == tmp_path / "prime"
