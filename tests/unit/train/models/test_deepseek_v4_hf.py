"""HF-oracle parity checks for the whole DeepSeek V4 model.

These tests use `transformers.models.deepseek_v4` as the correctness oracle. That package only
exists from transformers 5.15, and the repo pins an older version so the DS V4 work does not
drag an unrelated dependency bump along with it. Run them explicitly:

    uv run --with 'transformers==5.15.0' pytest tests/unit/train/models/test_deepseek_v4_hf.py -v

Under the pinned version the module skips rather than erroring. Everything these tests share
with the HF-free half lives in `deepseek_v4_helpers.py`.
"""

import pytest

pytest.importorskip("transformers.models.deepseek_v4")

import torch
from torch import nn
from transformers.core_model_loading import revert_weight_conversion
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

from .deepseek_v4_helpers import (
    _BASE,
    _assert_close,
    _identity_attention,
    _IdentityMLP,
    _prime_config,
    _randomize,
    _run_pair,
    _seed_rng,  # noqa: F401 -- pytest fixture, applied by name
    _to_on_disk_naming,
    _torch_rms_norm,  # noqa: F401 -- pytest fixture, applied by name
)

pytestmark = [pytest.mark.gpu]


def _configs() -> tuple[HFDeepseekV4Config, DeepseekV4Config]:
    hf_config = HFDeepseekV4Config(**_BASE)
    # Force the eager path so HF actually runs its sink softmax, and keep the compressors'
    # rolling-window caches out of a training-shaped single forward.
    hf_config._attn_implementation = "eager"
    hf_config.use_cache = False
    return hf_config, _prime_config()


def _on_disk_state_dict(hf_model: nn.Module) -> dict[str, torch.Tensor]:
    """An HF model's weights under the key naming a real DeepSeek V4 checkpoint uses.

    `conversion_chain` converts *on-disk* names, which for this model are not the names
    `hf_model.state_dict()` returns: transformers carries a conversion registry entry for
    deepseek_v4 and applies it inside `from_pretrained` / `save_pretrained`, so the on-disk
    names are the compact DeepSeek-native ones (`attn`, `ffn`, `wkv`, per-expert `w1`/`w2`/`w3`,
    no `model.` prefix). The trainer reads raw on-disk state dicts in `load_dcp_from_hf` and
    never goes through `from_pretrained`, so that is the naming the chain has to handle.

    `revert_weight_conversion` is transformers' own reverse pass, the one `save_pretrained`
    runs, so this stays authoritative rather than restating the mapping here.
    """
    reverted = revert_weight_conversion(hf_model, dict(hf_model.state_dict()))
    return _to_on_disk_naming(reverted)


def get_model_pairs(dtype: torch.dtype = torch.bfloat16) -> tuple[nn.Module, nn.Module]:
    """Build an HF and a prime-rl model carrying identical weights."""
    hf_config, prime_config = _configs()
    with torch.device("cuda"), default_dtype(dtype):
        hf_model = HFDeepseekV4ForCausalLM._from_config(hf_config)
        prime_model = DeepseekV4ForCausalLM._from_config(prime_config)
    _randomize(hf_model)

    with torch.no_grad():
        state_dict = _on_disk_state_dict(hf_model)
        prime_state_keys = set(prime_model.state_dict())
        prime_model.convert_to_prime(state_dict)
        assert set(state_dict) == prime_state_keys, "the converted HF key set must equal prime-rl's exactly"
        prime_model.load_state_dict(state_dict)

    # Training code wraps the LM head; tests mirror that so forward takes labels/temperature.
    inject_prime_lm_head(prime_model, chunk_size=None)
    return hf_model, prime_model


def test_deepseek_v4_attn_only(_torch_rms_norm):  # noqa: F811
    hf_model, prime_model = get_model_pairs()
    for model in (hf_model, prime_model):
        for layer in model.model.layers:
            layer.mlp = _IdentityMLP()

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=0.02, grad_rtol=0.02)


def test_deepseek_v4_mlp_only(_torch_rms_norm):  # noqa: F811
    hf_model, prime_model = get_model_pairs()
    for model in (hf_model, prime_model):
        for layer in model.model.layers:
            layer.self_attn.forward = _identity_attention

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=0.02, grad_rtol=0.02)


def test_deepseek_v4(_torch_rms_norm):  # noqa: F811
    hf_model, prime_model = get_model_pairs()

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    # Loose by design, and the loosest assertion in this file. prime-rl's router scores in
    # float32 (`TokenChoiceTopKRouter` upcasts to keep the training loss from exploding)
    # while HF scores in the activation dtype, so in bfloat16 a few percent of the tokens
    # in the deeper layers pick a different expert set and their outputs then legitimately
    # diverge. `test_deepseek_v4_float32` runs the same comparison with that one difference
    # removed and holds to 1e-5; the isolation tests above carry the tight bfloat16 bound.
    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=0.2, grad_rtol=0.1)


def test_deepseek_v4_float32(_torch_rms_norm):  # noqa: F811
    """Full-model parity with the router's dtype difference removed.

    Not exact even so: `GroupedExperts` pushes the routed experts through
    `torch._grouped_mm` in bfloat16 whatever dtype the model runs in, so float32 buys
    parity on everything except the expert GEMMs. Measured deviation 2.1e-3 for the logits
    and 1.3e-3 for the embedding gradient, both relative to their own scale.
    """
    hf_model, prime_model = get_model_pairs(dtype=torch.float32)

    hf_logits, prime_logits = _run_pair(hf_model, prime_model)

    _assert_close(prime_logits, hf_logits, hf_model, prime_model, logits_rtol=2e-2, grad_rtol=2e-2)


def test_deepseek_v4_conversion_matches_the_hf_key_set():
    """The converted HF checkpoint has to land on prime-rl's keys with nothing left over."""
    hf_config, prime_config = _configs()
    with torch.device("meta"):
        hf_model = HFDeepseekV4ForCausalLM._from_config(hf_config)
        prime_model = DeepseekV4ForCausalLM._from_config(prime_config)

    state_dict = _on_disk_state_dict(hf_model)
    # A real checkpoint ships multi-token-prediction heads that neither side instantiates,
    # at the top level (`mtp.0.hc_attn_base`, ...) rather than nested inside a layer.
    state_dict["mtp.0.embed.weight"] = torch.empty(0, device="meta")
    prime_model.convert_to_prime(state_dict)

    assert set(state_dict) == set(prime_model.state_dict())


def test_deepseek_v4_config_translates_legacy_compress_ratios():
    """Real checkpoints ship the V3-flavoured legacy `compress_ratios`/`num_hash_layers` schema
    instead of `layer_types`/`mlp_layer_types` (see NOTES-ds-v4-inference-preflight.md). HF's own
    config translates these; prime-rl's must too, since prime-rl's model code reads
    `layer_types`/`mlp_layer_types` directly rather than `compress_ratios`.
    """
    legacy_kwargs = dict(
        num_hidden_layers=6,
        compress_ratios=[0, 0, 4, 128, 4, 128],
        num_hash_layers=2,
    )
    hf_config = HFDeepseekV4Config(**legacy_kwargs)
    prime_config = DeepseekV4Config(**legacy_kwargs)

    assert prime_config.layer_types == hf_config.layer_types
    assert prime_config.mlp_layer_types == hf_config.mlp_layer_types
