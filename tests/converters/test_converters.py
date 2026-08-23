"""Byte-exactness matrix for the checkpoint converters, per mini_moe arch.

Chain per arch (see conftest): random tiny prime model -> source HF dir -> DCP
checkpoint -> {bf16, fp8} exports -> fp8 -> bf16 dequant. Every hop is checked
byte-for-byte where the math is exact; the lossy dequant hop checks byte-identity
on non-quantized tensors and a bounded error on quantized ones.
"""

import json
from pathlib import Path

import pytest
import torch

from prime_rl.utils.weights import load_state_dict

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_dcp_to_bf16_matches_source(source_dir: Path, bf16_dir: Path):
    """The bf16 export is byte-identical to the source weights (modulo dropped tied keys)."""
    source = load_state_dict(source_dir)
    export = load_state_dict(bf16_dir)

    assert not set(export) - set(source), f"unexpected keys: {sorted(set(export) - set(source))[:5]}"
    mismatch = [key for key in export if not tensors_equal(source[key], export[key])]
    assert not mismatch, f"{len(mismatch)} tensor mismatches, e.g. {mismatch[:5]}"

    # Keys only in the source must be tied duplicates of the embedding.
    embed = next(source[key] for key in source if key.endswith("embed_tokens.weight"))
    for key in set(source) - set(export):
        assert tensors_equal(source[key], embed), f"dropped key {key} is not a tied embedding duplicate"


def test_dcp_to_fp8_matches_chained(fp8_dir: Path, fp8_chained_dir: Path):
    """Direct dcp->fp8 and chained dcp->bf16->fp8 produce byte-identical tensors."""
    direct = load_state_dict(fp8_dir)
    chained = load_state_dict(fp8_chained_dir)

    assert set(direct) == set(chained), (sorted(set(direct) - set(chained))[:5], sorted(set(chained) - set(direct))[:5])
    mismatch = [key for key in direct if not tensors_equal(direct[key], chained[key])]
    assert not mismatch, f"{len(mismatch)} tensor mismatches, e.g. {mismatch[:5]}"

    direct_config = json.loads((fp8_dir / "config.json").read_text())["quantization_config"]
    chained_config = json.loads((fp8_chained_dir / "config.json").read_text())["quantization_config"]
    assert direct_config == chained_config


def test_bf16_to_fp8_structure(bf16_dir: Path, fp8_chained_dir: Path):
    """Skipped modules stay byte-identical bf16 and are listed in the quantization config."""
    bf16 = load_state_dict(bf16_dir)
    fp8 = load_state_dict(fp8_chained_dir)
    quantization_config = json.loads((fp8_chained_dir / "config.json").read_text())["quantization_config"]

    quantized = {key.removesuffix("_scale_inv") for key in fp8 if key.endswith("_scale_inv")}
    assert quantized, "no tensors were quantized"
    for key in bf16:
        if key in quantized:
            assert fp8[key].dtype == torch.float8_e4m3fn
        else:
            assert tensors_equal(bf16[key], fp8[key]), f"skipped tensor changed: {key}"
    skipped_linears = {
        key.removesuffix(".weight")
        for key, tensor in bf16.items()
        if key.endswith(".weight") and tensor.ndim == 2 and key not in quantized
    }
    assert skipped_linears == set(quantization_config["modules_to_not_convert"])

    index = json.loads((fp8_chained_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(fp8)


def test_fp8_to_bf16_roundtrip(bf16_dir: Path, fp8_dir: Path, dequant_dir: Path):
    """Dequant restores the bf16 layout: skipped tensors byte-identical, quantized ones bounded."""
    bf16 = load_state_dict(bf16_dir)
    fp8 = load_state_dict(fp8_dir)
    dequant = load_state_dict(dequant_dir)

    assert set(dequant) == set(bf16), (sorted(set(dequant) - set(bf16))[:5], sorted(set(bf16) - set(dequant))[:5])
    quantized = {key.removesuffix("_scale_inv") for key in fp8 if key.endswith("_scale_inv")}
    for key, tensor in dequant.items():
        if key in quantized:
            reference = bf16[key].float()
            error = (tensor.float() - reference).abs() / reference.abs().clamp(min=1e-6)
            assert error.median() < 0.05, f"dequant error too large for {key}: {error.median():.4f}"
        else:
            assert tensors_equal(bf16[key], tensor), f"non-quantized tensor changed: {key}"

    config = json.loads((dequant_dir / "config.json").read_text())
    assert "quantization_config" not in config
