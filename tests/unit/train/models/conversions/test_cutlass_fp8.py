"""Tests for cutlass FP8 e4m3 per-channel conversion + registry-extension plumbing.

Direct-loads the conversions package to bypass the heavy
``prime_rl.trainer`` import chain (CUDA + torchrun + ray + …) so the suite
runs on a plain CPU CI box with only ``torch`` installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent
_CONV_PKG_DIR = _REPO_ROOT / "src" / "prime_rl" / "trainer" / "models" / "conversions"
_FP8_PATH = _REPO_ROOT / "src" / "prime_rl" / "trainer" / "models" / "fp8.py"


def _direct_load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def conv_pkg():
    """Load the conversions package + its dependencies in isolation.

    Order matters: ``conversions/__init__.py`` registers ``cutlass_fp8`` as
    a late side-effect import; we need ``prime_rl.trainer.models.fp8`` to
    be importable first.
    """
    # Synthesize the prime_rl.trainer.models package hierarchy so the
    # relative imports inside the conversion modules resolve.
    for fqn, path in [
        ("prime_rl", _REPO_ROOT / "src" / "prime_rl"),
        ("prime_rl.trainer", _REPO_ROOT / "src" / "prime_rl" / "trainer"),
        ("prime_rl.trainer.models", _REPO_ROOT / "src" / "prime_rl" / "trainer" / "models"),
    ]:
        if fqn in sys.modules:
            continue
        pkg = types.ModuleType(fqn)
        pkg.__path__ = [str(path)]
        sys.modules[fqn] = pkg

    # Load fp8.py first — the conversion modules import from it.
    _direct_load("prime_rl.trainer.models.fp8", _FP8_PATH)

    # Now load the conversions package, then its submodules. We point at
    # the directory's __init__.py explicitly so we don't get the partial
    # package from a parent that's already half-loaded.
    pkg = _direct_load(
        "prime_rl.trainer.models.conversions", _CONV_PKG_DIR / "__init__.py"
    )
    return pkg


# ----------------------------------------------------------------------------
# Per-output-channel quantize helper (lives in fp8.py)
# ----------------------------------------------------------------------------


def test_fp8_per_channel_2d_round_trip_shape(conv_pkg):
    from prime_rl.trainer.models.fp8 import fp8_per_channel_quantize

    w = torch.randn(64, 256, dtype=torch.bfloat16)
    q, s = fp8_per_channel_quantize(w)
    assert q.shape == (64, 256)
    assert q.dtype == torch.float8_e4m3fn
    assert s.shape == (64,)
    assert s.dtype == torch.float32


def test_fp8_per_channel_3d_round_trip_shape(conv_pkg):
    from prime_rl.trainer.models.fp8 import fp8_per_channel_quantize

    w = torch.randn(8, 64, 256, dtype=torch.bfloat16)
    q, s = fp8_per_channel_quantize(w)
    assert q.shape == (8, 64, 256)
    assert q.dtype == torch.float8_e4m3fn
    assert s.shape == (8, 64)
    assert s.dtype == torch.float32


def test_fp8_per_channel_rejects_1d(conv_pkg):
    from prime_rl.trainer.models.fp8 import fp8_per_channel_quantize

    with pytest.raises(ValueError, match="2D or 3D"):
        fp8_per_channel_quantize(torch.randn(64))


def test_fp8_per_channel_dequant_close_to_original(conv_pkg):
    """Round-trip accuracy: per-channel scaling has ~1% error band on bf16 inputs."""
    from prime_rl.trainer.models.fp8 import fp8_per_channel_quantize

    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16) * 0.1
    q, s = fp8_per_channel_quantize(w)
    dequant = q.float() * s.unsqueeze(-1)
    # FP8 e4m3 has ~3-bit mantissa → relative error tolerance is generous
    rel = (dequant - w.float()).abs() / (w.float().abs() + 1e-6)
    assert rel.median().item() < 0.05  # 5 % median error is realistic for fp8 e4m3


def test_fp8_per_channel_into_writes_buffers(conv_pkg):
    from prime_rl.trainer.models.fp8 import fp8_per_channel_quantize_into

    w = torch.randn(16, 64, dtype=torch.bfloat16)
    out = torch.empty(16, 64, dtype=torch.float8_e4m3fn)
    sf = torch.empty(16, dtype=torch.float32)
    fp8_per_channel_quantize_into(w, out=out, sf=sf)
    # Both buffers should now reflect a real quantization (not the empty pattern).
    assert sf.gt(0).all()
    assert out.float().abs().max() <= 448.0  # fp8 e4m3 finite range


# ----------------------------------------------------------------------------
# Registry extensions: compile_target + compile_metadata + new entry
# ----------------------------------------------------------------------------


def test_conversion_entry_carries_compile_target(conv_pkg):
    entry = conv_pkg.get("bf16_cast")
    assert entry.compile_target == conv_pkg.COMPILE_TARGET_HF_RAW
    assert entry.compile_metadata == {"dtype": "bfloat16"}


def test_fp8_128x128_tagged_deep_gemm(conv_pkg):
    entry = conv_pkg.get("fp8_128x128")
    assert entry.compile_target == conv_pkg.COMPILE_TARGET_DEEPGEMM_FP8
    assert entry.compile_metadata["block_size"] == [128, 128]
    assert entry.compile_metadata["scale_layout"] == "blockwise"


def test_cutlass_fp8_entry_registered(conv_pkg):
    entry = conv_pkg.get("cutlass_fp8_e4m3_per_channel")
    assert entry.requires_scale is True
    assert entry.compile_target == conv_pkg.COMPILE_TARGET_CUTLASS_FP8
    assert entry.compile_metadata == {
        "dtype": "e4m3",
        "scale_layout": "per_channel",
        "scale_axis": -1,
        "activation_scheme": "dynamic",
    }


def test_cutlass_fp8_in_registered_names(conv_pkg):
    names = conv_pkg.registered_names()
    assert "cutlass_fp8_e4m3_per_channel" in names
    assert "fp8_128x128" in names
    assert "bf16_cast" in names


def test_register_default_rule_appends(conv_pkg):
    """register_default_rule appends by default and prepends with insert_first=True."""

    sentinel_name = "bf16_cast"  # we know this exists

    def predicate_a(quant):
        return quant.get("quant_method") == "test_a"

    def predicate_b(quant):
        return quant.get("quant_method") == "test_b"

    # These mutate module state; use unique enough names that they don't
    # collide with the real rules.
    conv_pkg.register_default_rule(predicate_a, sentinel_name)
    conv_pkg.register_default_rule(predicate_b, sentinel_name, insert_first=True)

    # We can't read the table directly without breaking the encapsulation,
    # but we can verify behaviorally: predicate_b should be matched before
    # the existing rules; predicate_a should be matched after.
    import prime_rl.trainer.models.conversions as conv

    rules = conv._DEFAULT_RULES
    # predicate_b should now be at index 0
    assert rules[0][0] is predicate_b
    # predicate_a should be at the end
    assert rules[-1][0] is predicate_a


def test_unknown_quant_raises_listing_registered(conv_pkg, fake_hf_config):
    """When no rule matches, the error message lists what IS registered."""
    fake_hf_config["quant"] = {"quant_method": "totally_unknown_method"}
    with pytest.raises(NotImplementedError, match="registered conversions"):
        conv_pkg.select_default_conversion("fake/model")


# ----------------------------------------------------------------------------
# select_default_conversion dispatch (the new table-driven path)
# ----------------------------------------------------------------------------


@pytest.fixture
def fake_hf_config(monkeypatch):
    """Stub ``transformers.AutoConfig`` for the test session so
    ``select_default_conversion`` runs without an HF download.

    Because the conversions module imports ``AutoConfig`` lazily inside
    the function, we have to populate ``sys.modules['transformers']``
    with our stub *before* the function call resolves the import.
    """
    holder = {"quant": None}

    class _Fake:
        @property
        def quantization_config(self):
            return holder["quant"]

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _Fake()

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoConfig = _FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    return holder


def test_default_no_quant_is_bf16(conv_pkg, fake_hf_config):
    fake_hf_config["quant"] = None
    assert conv_pkg.select_default_conversion("any/model") == "bf16_cast"


def test_default_deep_gemm_fp8(conv_pkg, fake_hf_config):
    fake_hf_config["quant"] = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    assert conv_pkg.select_default_conversion("any/model") == "fp8_128x128"


def test_default_cutlass_fp8_via_explicit_format(conv_pkg, fake_hf_config):
    fake_hf_config["quant"] = {
        "quant_method": "fp8",
        "quant_format": "cutlass",
    }
    assert (
        conv_pkg.select_default_conversion("any/model")
        == "cutlass_fp8_e4m3_per_channel"
    )


def test_default_cutlass_fp8_via_dynamic_no_block_size(conv_pkg, fake_hf_config):
    fake_hf_config["quant"] = {
        "quant_method": "fp8",
        "weight_block_size": None,
        "activation_scheme": "dynamic",
    }
    assert (
        conv_pkg.select_default_conversion("any/model")
        == "cutlass_fp8_e4m3_per_channel"
    )


def test_default_deep_gemm_wins_over_cutlass_when_block_size_set(
    conv_pkg, fake_hf_config
):
    """Both rules could plausibly fire for a config with block_size=[128,128]
    AND activation_scheme="dynamic"; the deep-gemm rule was registered first
    and must win."""
    fake_hf_config["quant"] = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    }
    assert conv_pkg.select_default_conversion("any/model") == "fp8_128x128"


# ----------------------------------------------------------------------------
# End-to-end fn dispatch: 2D + 3D shapes via the registered conversion entry
# ----------------------------------------------------------------------------


def test_cutlass_fp8_fn_dispatches_2d_linear(conv_pkg):
    entry = conv_pkg.get("cutlass_fp8_e4m3_per_channel")
    src = torch.randn(32, 128, dtype=torch.bfloat16)
    out = torch.empty(32, 128, dtype=torch.float8_e4m3fn)
    sf = torch.empty(32, dtype=torch.float32)
    entry.fn(src, out, sf)
    assert sf.gt(0).all()
    assert out.float().abs().max() <= 448.0


def test_cutlass_fp8_fn_dispatches_3d_moe(conv_pkg):
    entry = conv_pkg.get("cutlass_fp8_e4m3_per_channel")
    src = torch.randn(4, 32, 128, dtype=torch.bfloat16)  # E=4 experts
    out = torch.empty(4, 32, 128, dtype=torch.float8_e4m3fn)
    sf = torch.empty(4, 32, dtype=torch.float32)
    entry.fn(src, out, sf)
    assert sf.shape == (4, 32)
    assert sf.gt(0).all()
    assert out.shape == (4, 32, 128)


def test_cutlass_fp8_fn_requires_scale(conv_pkg):
    entry = conv_pkg.get("cutlass_fp8_e4m3_per_channel")
    src = torch.randn(8, 16, dtype=torch.bfloat16)
    out = torch.empty(8, 16, dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="scale_out"):
        entry.fn(src, out, None)
