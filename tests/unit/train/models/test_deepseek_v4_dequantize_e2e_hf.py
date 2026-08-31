"""HF-vs-PrimeRL cross-checks for DeepSeek V4's fp8/MXFP4 dequantization.

Both sides dequantize the *same* quantized tensors, each with its own implementation
(`transformers.integrations.finegrained_fp8.Fp8Dequantize` for HF, prime-rl's own
`dequantize_state_dict_` for the trainer), then load the resulting plain `bfloat16` weights
into ordinary, non-quantized model instances and compare forward passes. Neither side ever
runs a quantized-weight kernel (`FP8Linear`/`FP8Experts`/DeepGEMM/Triton): that machinery is
an inference-serving optimization, keeps weights compressed at rest, and is unrelated to
whether the dequantization math itself is correct -- which is what these tests check.

Both sides of the comparison need `transformers.models.deepseek_v4`, which only exists from
transformers 5.15, and the repo pins an older version so the DS V4 work does not drag an
unrelated dependency bump along with it. Run them explicitly:

    uv run --with 'transformers==5.15.0' pytest tests/unit/train/models/test_deepseek_v4_dequantize_e2e_hf.py -v

Under the pinned version the module skips rather than erroring. `Fp8Dequantize` itself exists on
the pinned version; it is HF's model class that does not.
"""

import pytest

pytest.importorskip("transformers.models.deepseek_v4")

import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4ForCausalLM as PrimeRLDeepseekV4ForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

from .deepseek_v4_helpers import (
    _assert_close,
    _IdentityMLP,
    _randomize,
    _run_pair,
    _torch_rms_norm,  # noqa: F401 -- pytest fixture, referenced by name in test signatures
)
from .test_deepseek_v4_hf import _configs, _on_disk_state_dict

pytestmark = [pytest.mark.gpu]

_DENSE_FP8_SUFFIXES = {"wq_a", "wq_b", "wo_a", "wo_b", "wkv"}
_FP8_E4M3_MAX = 448.0
_MXFP4_MAX = 6.0
_MXFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_QUANTIZED_DTYPES = (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8)


def _is_dense_fp8_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    parts = key.split(".")
    if len(parts) >= 3 and parts[-3] == "attn" and parts[-2] in _DENSE_FP8_SUFFIXES:
        return True
    return key.endswith("indexer.wq_b.weight") or ".shared_experts." in key


def _is_routed_expert_key(key: str) -> bool:
    return bool(re.search(r"\.experts\.\d+\.w[123]\.weight$", key))


def _block_scale_bytes(absmax: torch.Tensor, max_representable: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest power-of-two block scale, as `(float32 value, raw e8m0 byte)`."""
    exponent = torch.ceil(torch.log2(absmax.clamp(min=1e-12) / max_representable))
    byte = (exponent + 127.0).clamp(0, 254).to(torch.uint8)
    return torch.exp2(exponent), byte


def _quantize_dense_fp8(weight: torch.Tensor, block: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    # Real checkpoints use 128x128 blocks; `_BASE`'s dims (chosen for a fast, small test model,
    # not to mirror real kernel-constrained shapes) are too small for that, so this test uses a
    # smaller block instead. `dequantize_weight` derives block size from the scale tensor's
    # shape dynamically, so the exact size doesn't matter -- only that both sides agree, and
    # that it's non-trivial (>1 block along an axis) to actually exercise block orientation.
    rows, cols = weight.shape
    blocked = weight.float().reshape(rows // block, block, cols // block, block)
    absmax = blocked.abs().amax(dim=(1, 3))
    scale, byte = _block_scale_bytes(absmax, _FP8_E4M3_MAX)
    scale_expanded = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    quantized = (weight.float() / scale_expanded).to(torch.float8_e4m3fn)
    return quantized, byte.view(torch.float8_e8m0fnu)


def _quantize_expert_mxfp4(weight: torch.Tensor, block: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    blocked = weight.float().reshape(rows, cols // block, block)
    absmax = blocked.abs().amax(dim=-1)
    scale, byte = _block_scale_bytes(absmax, _MXFP4_MAX)
    scale_expanded = scale.repeat_interleave(block, dim=-1)
    normalized = weight.float() / scale_expanded

    magnitudes = _MXFP4_MAGNITUDES.to(normalized.device)
    magnitude_idx = (normalized.abs().unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    sign_bit = (normalized < 0).to(torch.uint8) * 8
    nibble = magnitude_idx.to(torch.uint8) + sign_bit

    low, high = nibble[:, 0::2], nibble[:, 1::2]
    packed = (low | (high << 4)).contiguous().view(torch.int8)
    return packed, byte.view(torch.float8_e8m0fnu)


def _quantize_selected(state_dict: dict[str, torch.Tensor], quantize_experts: bool) -> dict[str, torch.Tensor]:
    """Quantize dense fp8 keys, plus routed-expert keys to packed MXFP4 if requested."""
    quantized = {}
    for key, tensor in state_dict.items():
        if quantize_experts and _is_routed_expert_key(key):
            weight, scale = _quantize_expert_mxfp4(tensor)
        elif _is_dense_fp8_key(key):
            weight, scale = _quantize_dense_fp8(tensor)
        else:
            quantized[key] = tensor
            continue
        quantized[key] = weight
        quantized[key.removesuffix(".weight") + ".scale"] = scale
    return quantized


def _hf_dequantize(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize with `transformers`' own `Fp8Dequantize`, independent of prime-rl's code."""
    dequantize_one = Fp8Dequantize(hf_quantizer=None)._dequantize_one
    out = {}
    for key, tensor in state_dict.items():
        if key.endswith(".scale"):
            continue
        scale_key = key.removesuffix(".weight") + ".scale"
        if key.endswith(".weight") and scale_key in state_dict:
            out[key] = dequantize_one(tensor, state_dict[scale_key])
        else:
            out[key] = tensor
    return out


def _reload_hf_model_from_dequantized(hf_config, on_disk_dequantized: dict[str, torch.Tensor], save_dir: Path):
    """Round-trip through disk so `from_pretrained` applies the real on-disk -> HF-native
    key/layout conversion (e.g. fusing routed experts' `w1`/`w3` into `gate_up_proj`).

    The reloaded model carries no `quantization_config`, so it is built from ordinary
    `nn.Linear`/plain-expert modules -- no `FP8Linear`/`FP8Experts`, no kernel involved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(on_disk_dequantized, save_dir / "model.safetensors", metadata={"format": "pt"})
    hf_config.save_pretrained(save_dir)
    model = HFDeepseekV4ForCausalLM.from_pretrained(str(save_dir), config=hf_config)
    return model.to(device="cuda", dtype=torch.float32)


def _prime_model_from_quantized(prime_config, quantized_state_dict: dict[str, torch.Tensor]):
    """Load prime-rl's model straight from the quantized on-disk dict via `convert_to_prime`,
    which dequantizes internally through `dequantize_state_dict_` -- the same code path
    `load_dcp_from_hf` uses.
    """
    with torch.device("cuda"), default_dtype(torch.float32):
        prime_model = PrimeRLDeepseekV4ForCausalLM._from_config(prime_config)
    with torch.no_grad():
        state_dict = {
            k: v.to(device="cuda") if v.dtype in _QUANTIZED_DTYPES else v.to(device="cuda", dtype=torch.float32)
            for k, v in quantized_state_dict.items()
        }
        prime_model.convert_to_prime(state_dict)
        prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    return prime_model


def _get_dequantized_model_pairs(tmp_path: Path, quantize_experts: bool):
    hf_config, prime_config = _configs()
    with torch.device("cuda"), default_dtype(torch.float32):
        hf_model = HFDeepseekV4ForCausalLM._from_config(hf_config)
    _randomize(hf_model)

    on_disk = _on_disk_state_dict(hf_model)
    quantized = _quantize_selected(on_disk, quantize_experts=quantize_experts)

    hf_dequantized = _hf_dequantize(quantized)
    hf_model = _reload_hf_model_from_dequantized(hf_config, hf_dequantized, tmp_path / "hf")
    prime_model = _prime_model_from_quantized(prime_config, quantized)
    return hf_model, prime_model


def test_dequantized_attention_matches_hf(_torch_rms_norm, tmp_path):  # noqa: F811
    """Dense-fp8 attention only, MLP swapped for an identity so it can't mask anything.

    `_BASE`'s layer stack carries all three attention types (sliding, CSA, HCA), so this
    covers every attention variant in one pass. Dense fp8 covers the attention projections
    (`wq_a`/`wq_b`/`wo_a`/`wo_b`/`wkv`) and the CSA/HCA indexer's `wq_b` -- the compressors'
    own inner projections are left unquantized, matching the real checkpoint.
    """
    hf_model, prime_model = _get_dequantized_model_pairs(tmp_path, quantize_experts=False)
    for model in (hf_model, prime_model):
        for layer in model.model.layers:
            layer.mlp = _IdentityMLP()

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=1e-4, grad_rtol=1e-4)


def test_dequantized_full_model_matches_hf(_torch_rms_norm, tmp_path):  # noqa: F811
    """Full model: dense fp8 (attention, shared experts) + packed MXFP4 (routed experts).

    `_BASE` mixes hash-routed and gate-routed MoE layers, so this exercises the packed-MXFP4
    unpack/dequant path for both. Looser tolerance than the attention-only test: MXFP4's
    8-level magnitude LUT is much coarser than fp8's, and the noise compounds across 5 layers.
    """
    hf_model, prime_model = _get_dequantized_model_pairs(tmp_path, quantize_experts=True)

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=1e-4, grad_rtol=1e-4)
