# DeepSeek V4 RoPE diverges from DeepSeek's reference implementation

Draft of an upstream vLLM issue. Not filed. Both defects are worked around locally in
`src/prime_rl/inference/patches.py` by
`monkey_patch_deepseek_v4_rope_disable_yarn_on_sliding_layers`, pinned by
`tests/unit/inference/test_deepseek_v4_rope_patch.py`.

## Summary

`vllm/models/deepseek_v4/common/rope.py`'s `build_deepseek_v4_rope` is the only rotary-embedding
builder for DeepSeek V4. It branches the RoPE base on `compress_ratio` but never branches the
scaling type, and it reads the rope configuration as a single flat dict. Two consequences:

1. On `deepseek-ai/DeepSeek-V4-Flash-0731`, whose `config.json` ships a flat legacy `rope_scaling`,
   YaRN is applied to the pure sliding-window layers (0 and 1 of 43). DeepSeek's reference
   implementation disables it there.
2. On any checkpoint that has been through `save_pretrained`, whose `rope_parameters` is nested
   under `main` / `compress` keys, the builder cannot read the nested form and falls back to
   unscaled RoPE for all 43 layers, silently dropping YaRN from the 41 compressed ones.

## The reference

DeepSeek ships its own inference code inside the checkpoint, at
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference

`model.py:481-487` chooses the frequencies per layer:

```python
if self.compress_ratio:
    original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
else:
    # disable YaRN and use base rope_theta in pure sliding-window attention
    original_seq_len, rope_theta = 0, args.rope_theta
freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                 rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
```

and `precompute_freqs_cis` (`model.py:206-235`) applies the NTK-by-parts ramp only when
`original_seq_len > 0`:

```python
freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
if original_seq_len > 0:
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
```

So passing `original_seq_len = 0` is how the reference disables YaRN. A layer with no compressor
gets plain RoPE at `rope_theta`; every compressed layer gets YaRN at `compress_rope_theta`.
Hugging Face's `DeepseekV4Config` encodes the same split, keying `rope_parameters` by the rope-type
labels `main` (always `rope_type="default"`) and `compress` (the checkpoint's YaRN), with
`DeepseekV4Attention` selecting `"main" if layer_type == "sliding_attention" else "compress"`.

The checkpoint's `inference/config.json` confirms which layers those are:

```json
"compress_ratios": [0, 0, 4, 128, 4, 128, ..., 4, 0, 0, 0],
"rope_theta": 10000, "compress_rope_theta": 160000, "rope_factor": 16,
"original_seq_len": 65536, "beta_fast": 32, "beta_slow": 1, "window_size": 128
```

With `num_hidden_layers = 43`, the leading two entries are the sliding-window layers (the trailing
three zeros are MTP layers, which are not built).

## Bug 1: YaRN applied to the sliding-window layers

```python
rope_parameters["rope_theta"] = (
    config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
)
if rope_parameters["rope_type"] != "default":
    rope_parameters["rope_type"] = (
        "deepseek_yarn" if rope_parameters.get("apply_yarn_scaling", True)
        else "deepseek_llama_scaling"
    )
```

`rope_type` comes from the single global config field, so layers 0 and 1 get the correct base
(10000, since `max(1, compress_ratios[i]) == 1`) but the YaRN scaling the reference disables.
`apply_yarn_scaling: false` does not avoid it: `deepseek_llama_scaling` reaches the same `get_rope`
branch and builds the same class.

Comparing the built `cos_sin_cache` against the reference's own `precompute_freqs_cis` at the
checkpoint's values:

```
SWA layers 0,1  vs reference: max abs inv_freq error 2.885e-04  DIVERGES
CSA layers      vs reference: max abs inv_freq error 5.960e-08  MATCH
```

11 of the 32 channel pairs have their frequency scaled, by up to 4.8x. The affected channels are
the slowest ones and these layers only attend within `sliding_window = 128`, so the worst
per-channel phase error inside the window is 0.037 rad. Small, but it is a systematic difference
from the weights' own reference implementation, and it shows up as a train/inference mismatch for
anyone whose trainer follows the HF model.

## Bug 2: nested `rope_parameters` is not parsed

`vllm/transformers_utils/configs/deepseek_v4.py` is the whole config class:

```python
self.rope_parameters = rope_scaling or rope_parameters
```

which assumes a flat dict. Hugging Face's `DeepseekV4Config.to_dict()` writes the nested form, so
for any resaved checkpoint transformers finds no top-level `type` / `rope_type` and injects
`rope_type="default"` beside the two sub-dicts, warning:

```
Unrecognized keys in `rope_parameters` for 'rope_type'='default': {'compress', 'main'}
```

`build_deepseek_v4_rope` then reads that `"default"`, skips the YaRN rewrite, and `get_rope` builds
a plain `RotaryEmbedding` for every layer. Layers 0 and 1 happen to come out correct; layers 2
through 42 lose YaRN entirely: 16 of 32 channel pairs off by up to 16x, giving 0.34 rad of phase
error at a 1024-token offset, 2.72 rad at 8192, and 21.8 rad at 65536.

Two secondary symptoms of the same path:

- `get_rope`'s cache-key builder only tuple-izes top-level list values, so the nested dict raises
  `TypeError: unhashable type: 'dict'` at `if key in _ROPE_DICT`.
- The `"default"` branch builds `RotaryEmbedding` at the ambient dtype (bf16 during model load),
  which trips the unconditional `cos_sin_cache.dtype == torch.float32` assert in
  `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`.

vLLM already has `is_rope_parameters_nested` (`vllm/transformers_utils/config.py:174-179`), but it
tests membership in transformers' `ALLOWED_LAYER_TYPES` and V4's labels are `main` / `compress`, so
it returns False here and the V4 path never consults it regardless.

## Reproduction

```python
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

flat = {"type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
        "original_max_position_embeddings": 65536}
config = DeepseekV4Config(rope_scaling=dict(flat), rope_theta=10000.0,
                          compress_rope_theta=160000.0, max_position_embeddings=1048576)
with set_current_vllm_config(VllmConfig()):
    swa = build_deepseek_v4_rope(config, head_dim=512, rope_head_dim=64,
                                 max_position_embeddings=1048576, compress_ratio=1)

cos, sin = swa.cos_sin_cache[1].chunk(2)
got = torch.atan2(sin, cos)
plain = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
print((got - plain).abs().max())          # 2.885e-04, expected ~0 per the reference
```

For bug 2, pass `rope_parameters={"main": {...}, "compress": {...}}` instead of `rope_scaling` and
observe that both `compress_ratio=1` and `compress_ratio=4` return a plain `RotaryEmbedding`.

## Suggested fix

In `build_deepseek_v4_rope`:

1. Branch the scaling the same way the base is already branched. For `compress_ratio <= 1`, build
   the layer without YaRN, matching `original_seq_len = 0` in the reference.
2. When `config.rope_parameters` carries `main` / `compress` sub-dicts, read the parameters from
   `compress` rather than the top level.

One caution for (1): substituting `rope_type="default"` is not equivalent, because `get_rope` then
returns a plain `RotaryEmbedding`, which rotates the leading `rotary_dim` channels instead of the
trailing ones and takes no `inverse` argument, both of which the V4 attention path relies on.
Setting `factor = 1` on the existing `deepseek_yarn` path keeps `DeepseekV4ScalingRotaryEmbedding`
and makes interpolation and extrapolation identical, which collapses the ramp to plain RoPE exactly
and leaves `yarn_get_mscale` at 1.0. `original_max_position_embeddings` has to be widened to
`max_position_embeddings` alongside it, since the cache is `original_max * factor` rows.

Separately, `rope_parameters = config.rope_parameters` is a live reference that lines 18 to 30
mutate in place, so after model init the config permanently carries the last-built layer's
`rope_theta` plus the injected `mscale`, `is_deepseek_v4` and `rope_dim` keys.

## Environment

- vLLM 0.26.0+cu129
- transformers 5.6.2
- `deepseek-ai/DeepSeek-V4-Flash-0731`
