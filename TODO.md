# TODO

## Qwen3.5 GatedDeltaNet varlen patch is broken on transformers >= 5.13

`_patch_qwen3_5_linear_attn_varlen()` in `src/prime_rl/trainer/model.py` monkey-patches
`Qwen3_5DecoderLayer`/`Qwen3_5GatedDeltaNet` so packed RL/SFT batches don't leak Mamba-style
recurrent state across sequence boundaries. It has not been updated for the transformers 5.15.0
bump and needs real work before Qwen3.5 (dense or MoE) can train safely on packed sequences with
this transformers version. No existing test exercises this code path at all today
(`_patch_qwen3_5_linear_attn_varlen`, `Qwen3_5GatedDeltaNet`, `GatedDeltaNet` have zero hits under
`tests/`), which is why none of this has been caught by CI.

Originally flagged (partially) by reviewer `dzautner` on PR #3055 (`chore/bump-transformers`,
2026-08-11); the deeper issue below was found during follow-up investigation on this branch.

### Issue 1 (dzautner, confirmed): decoder-layer dispatch reads a renamed attribute

The patched `Qwen3_5DecoderLayer.forward` checks `if self.layer_type != "linear_attention":` to
decide whether to dispatch to the GatedDeltaNet path. Transformers >= 5.13 renamed this attribute:
`Qwen3_5DecoderLayer.__init__` now sets `self.block_type = config.layer_types[layer_idx]`, not
`self.layer_type`. Confirmed directly against the installed 5.15.0 source -- raises
`AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'layer_type'` the moment the patch
runs. Fix: `getattr(self, "block_type", getattr(self, "layer_type", None))`.

### Issue 2 (found on this branch): the GDN forward's kernel references no longer exist

Even after fixing Issue 1, the patched `Qwen3_5GatedDeltaNet.forward` calls
`self.causal_conv1d_fn(...)` and `self.chunk_gated_delta_rule(...)` as if they were bound instance
attributes. Confirmed on the installed 5.15.0: neither exists on a real instance
(`hasattr(gdn, "causal_conv1d_fn")` / `hasattr(gdn, "chunk_gated_delta_rule")` are both `False`).
Transformers moved these to free functions at module scope in
`transformers.models.qwen3_5.modeling_qwen3_5` (`causal_conv1d_fn`, `torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule`), each wrapped in `@use_kernel_func_from_hub_with_fallback`,
which resolves at decoration time to either an accelerated external package's kernel or a plain
PyTorch reference implementation.

This isn't just a rename -- the reference implementations behave differently from what the patch
assumes:

- `causal_conv1d_fn`'s PyTorch reference (used when the `causal_conv1d` pip package isn't
  installed -- true in this dev env) ignores `seq_idx` entirely: a single `F.conv1d` over the whole
  tensor, no packing awareness.
- `torch_chunk_gated_delta_rule`'s reference implementation similarly never references `cu_seqlens`
  in its body -- it's accepted into `**kwargs` and silently dropped.

**Important nuance found while investigating a "detect and error out" guard**: you cannot reliably
tell from outside which implementation (real kernel vs. reference) is active by introspecting the
function object -- `@functools.wraps(torch_function)` copies `__module__`/`__qualname__`/signature
from the *reference* implementation onto the wrapper regardless of what actually got resolved, so
naive checks (`fn.__module__`, `inspect.signature(fn)`) always look like the reference and prove
nothing either way.

The precise, fully static way to check resolution (no model math, no randomness) is to reuse
transformers' own resolution helper directly:

```python
import importlib
from transformers.utils.import_utils import resolve_internal_import

def resolves_to_kernel(package: str, chained_path: str) -> bool:
    try:
        module = importlib.import_module(package)
    except ImportError:
        return False
    return resolve_internal_import(module, chained_path) is not None
```

Empirically in this dev environment:

- `resolves_to_kernel("causal_conv1d", "causal_conv1d_fn")` -> `False`. `causal_conv1d` isn't
  installed at all, so this is *guaranteed* to be the unpacked reference implementation.
- `resolves_to_kernel("fla", "ops.gated_delta_rule.chunk_gated_delta_rule")` -> `True`. It resolves
  to `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`, which has a real `cu_seqlens` (and
  `cu_seqlens_cpu`) parameter -- this one may already be packing-aware, but:
  - its signature differs from what the current patch assumes (`scale`, `cu_seqlens_cpu`,
    `use_beta_sigmoid_in_kernel`, `allow_neg_eigval`, `state_v_first`, `cp_context`, no positional
    `chunk_size` -- not a drop-in match for `self.chunk_gated_delta_rule(...)`'s current call site).
  - fla's semantics/correctness for `cu_seqlens` haven't been verified here (no audit of fla's
    implementation, no numeric parity check against a known-correct per-segment reference).

### What still needs to happen

1. Fix Issue 1 (`layer_type` -> `block_type`, with a `getattr` fallback for safety).
2. Rewrite the `causal_conv1d_fn` branch to call the module-level function directly. Since it's
   confirmed to always be the unpacked reference in this environment (and likely in most training
   environments unless `causal_conv1d` is explicitly installed), keep prime-rl's existing per-segment
   manual conv1d loop as the (only) path rather than trying to conditionally use a "fast path" that
   can't be reliably detected as safe from outside.
3. Re-derive the `chunk_gated_delta_rule` call against fla's actual current signature
   (`fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`). Decide, with an actual correctness
   check (e.g. numeric parity: packed-with-cu_seqlens vs. per-segment-with-initial_state=None on
   synthetic multi-segment input), whether fla's `cu_seqlens` support is sufficient on its own or
   whether prime-rl still needs to do its own per-segment looping for this step too.
4. Once the above is settled, add regression tests (none exist today):
   - decoder-layer dispatch survives the `block_type` rename (should fail red today).
   - packed vs. unpacked GDN output parity, to actually pin the correctness property the patch
     exists for.
5. Separately (dzautner, PR #3055, still unaddressed anywhere): `get_model()` in
   `trainer/model.py` decides whether to apply the Qwen3.5 patches via a name-string check plus a
   `model_type`-based fallback that only matches `"qwen3_5_moe*"`. A dense Qwen3.5 checkpoint loaded
   through a generic local alias (e.g. `/checkpoints/model4b`) silently skips the patch entirely --
   no crash, just reintroduces the state-leak bug, showing up only as elevated mismatch-KL. Fix:
   extract the detection into a small pure function checking both `model_type` and nested
   `text_config.model_type` for `qwen3_5`/`qwen3_5_moe` prefixes, and add a unit test for the dense
   generic-alias case.

## DeepSeek V4 port

Step-by-step plan, one commit each:

- [x] 1. Config + manifold-constrained hyper-connections (mHC)
- [x] 2. Rotary embedding + sliding-window attention (one of three attention layer types)
- [x] 3. Compressed Sparse Attention (CSA): compressor + Lightning Indexer (second attention layer type)
- [x] 4. Heavily Compressed Attention (HCA): compressor, no indexer (third and last attention layer type)
- [ ] 5. Standard MoE (router, experts, shared expert)
- [ ] 6. Hash-routed MoE (bootstrap layers)
- [ ] 7. Decoder layer + model classes + state-dict conversion chain, wiring everything above together

Attention is complete as of step 4: all three per-layer variants (`sliding_attention`,
`compressed_sparse_attention`, `heavily_compressed_attention`) share the core built in
step 2, CSA adds the step-3 compressor plus Lightning Indexer, and HCA adds the step-4
compressor (no indexer: it attends over every compressed entry its query position has
made causally readable). `COMPRESSOR_CLASSES` now covers every entry of
`DEEPSEEK_V4_LAYER_TYPES`, so `DeepseekV4Attention` accepts any config the config's own
`validate_architecture` accepts.

Deliberate scope reductions taken in steps 1-4 that later steps must revisit:

- **YaRN on the compress RoPE branch is not wired up.** `DeepseekV4Config._nest_rope_parameters`
  forces `rope_type="default"` for both the `main` and `compress` parameter sets. Real
  DeepSeek-V4 checkpoints use YaRN (`factor=16`, `attention_factor=1.0`) for `compress`,
  with `rope_theta=160000.0`. Any `rope_scaling` keys passed in are preserved in the
  `compress` dict but ignored while `rope_type` stays `"default"`. Wiring this up means
  dropping the forced override, restoring HF's `attention_factor=1.0` default for the
  YaRN case, and porting HF's `validate_rope` override (the base class validator keys
  off `layer_types`, not the `main`/`compress` rope-type labels, so it warns about
  unrecognized keys otherwise).
- **No MTP.** `num_nextn_predict_layers` is not carried over. HF does not instantiate the
  MTP layers either, and the conversion chain will need to drop `mtp.*` keys.
- **`_init_weights` is not wired up.** `DeepseekV4HyperConnection` and `DeepseekV4HyperHead`
  expose `init_weights(init_std)` matching HF's `_init_weights` (normal for `fn`/`hc_fn`,
  zeros for `base`/`hc_base`, ones for `scale`/`hc_scale`), but nothing calls them until
  the `PreTrainedModel` subclass lands in step 7.
- **`tests/unit/train/models/test_deepseek_v4_temp.py` is scaffolding.** It isolates the
  hyper-connections, the rotary, and all three attention layer types against HF's
  reference classes. Fold it into a proper `test_deepseek_v4.py` full-model parity test
  once the model classes exist.
- **`build_sliding_window_mask` is single-document only.** It builds a dense
  `[1, 1, S, S]` additive mask from the causal + local-window predicate, with no
  `cu_seqlens` awareness, so a packed multi-document row would let the window bleed
  across document boundaries. Every other prime-rl model threads `seq_lens` down to a
  varlen flash-attention call instead (see `layers/attn.py`), which is also what makes
  the `O(S^2)` mask affordable to skip. Step 7 needs to decide between a
  `cu_seqlens`-aware mask builder and a flash-attention path with `window_size`.
- **Attention runs eagerly.** `eager_attention_with_sinks` is a direct port of GPT-OSS's
  reference softmax, chosen because the per-head sink logit has no flash-attention
  equivalent in the kernels prime-rl currently vendors. It is correct but materializes
  the full `[B, H, S, S + 1]` logit tensor, `[B, H, S, S + S / m + 1]` on a CSA layer.
- **The compressors are stateless (no KV cache).** prime-rl only ever runs a single
  forward + backward over a full sequence, never `generate()`, so
  `DeepseekV4HCACache` / `DeepseekV4CSACache` are not ported and nothing threads
  `past_key_values`. This is not just "delete the cache branch": it collapses HF's
  incremental window bookkeeping into a plain reshape. `first_window_position` is always
  `0`, `store_compression_weights`'s leftover buffer becomes "drop the trailing
  `S % m` tokens", and `update_overlap_state` (which carries the previous call's `Ca`
  slice across a forward boundary) becomes a one-window shift of the `Ca` series inside
  the same tensor, with window 0's slot left zero-valued and `-inf`-gated exactly as HF
  leaves it on a first call. Re-deriving any of this is only needed if prime-rl ever
  grows an incremental decode path.
- **The Lightning Indexer gets no gradient.** Its parameters
  (`compressor.indexer.*`) reach the loss only through integer top-k indices, so a
  backward pass leaves every one of them with `grad is None` (true of HF's
  implementation too, and pinned by `test_csa_indexer_selection_is_not_differentiable`).
  DeepSeek trains the indexer with a separate auxiliary loss that distills the dense
  attention distribution into the indexer scores; nothing here implements it, so an
  RL/SFT run would leave the indexer frozen at its checkpoint values. Fine for
  fine-tuning a released checkpoint, not fine for pre-training.
- **Top-k selection is plain PyTorch.** `torch.topk` over a dense `[B, S, T]` score
  tensor plus a dense `[B, 1, S, T]` additive block bias, `T = S / m`. GLM-MoE-DSA's
  fp8/tilelang indexer kernel is not reusable: it scores with a different formula. This
  is `O(S^2 / m)` memory on top of the eager attention above.
- **The compressor is single-document only**, like `build_sliding_window_mask`. It pools
  fixed windows of `m` tokens off the raw sequence and derives readability from
  `position_ids`, so a packed multi-document row would pool across document boundaries
  and let a later document read the compressed history of an earlier one. Same step-7
  decision as the sliding mask.
- **`position_ids` defaults to `arange`.** `DeepseekV4Attention.forward` synthesizes
  sequential positions when the caller passes none, matching the single-document
  assumption above. Step 7's decoder layer should always pass real ones.
- **Compressed-position cos/sin are recomputed per module.** The CSA compressor, its
  indexer and the HCA compressor each own a `DeepseekV4RotaryEmbedding` (as in HF) and
  each calls it every forward, even though the compressed positions are the deterministic
  `arange(S / m) * m` and identical across every layer of the same type. Cheap, but step 7
  could hoist it next to the model-level rotary.
- **HCA's `block_bias` is always a tensor, never `None`.** HF returns `None` from
  `DeepseekV4HCACompressor.forward` when `seq_len == 1` or no window is full, and its
  attention then zero-pads the mask instead. Stateless, both cases imply
  `compressed_len == 0` (a single token cannot fill a window of `m >= 4`), so prime-rl
  returns a `[B, 1, S, 0]` bias and the concatenation is a no-op, which is exactly what
  the zero-pad degenerates to. An incremental decode path would have to restore the
  distinction, since there `compressed_len > 0` with `seq_len == 1` is the normal case.
- **Rotary buffers are computed eagerly in `__init__`.** `DeepseekV4RotaryEmbedding`
  writes `<type>_inv_freq` on whatever device it is constructed on. Meta-device loading
  needs an `init_buffers_post_meta` on the step-7 `PreTrainedModel` subclass to
  re-derive them, mirroring HF's `_init_weights` branch for the rotary.

One structural departure from HF worth knowing about before step 7: `DeepseekV4Indexer`
subclasses a `DeepseekV4DualSeriesCompressor` base that also backs
`DeepseekV4CSACompressor`, because HF's two classes run byte-identical compression code
differing only in `head_dim` (`config.head_dim` vs `config.index_head_dim`). Every
parameter name is unchanged (`compressor.kv_proj`, `compressor.indexer.kv_proj`, ...), so
state-dict conversion is unaffected. `DeepseekV4HCACompressor` deliberately does *not*
share that base: it compresses non-overlapping windows from `head_dim`-wide (not
`2 * head_dim`-wide) projections, so the only code it could have inherited is the
`(position_ids + 1) // compress_rate` readability threshold, which it spells out itself.

Considered and rejected: reusing GLM-MoE-DSA's `apply_rope_interleave_single`
(`glm_moe_dsa/sparse_mla_attention.py:56-63`) or its reshape-then-shared-`rotate_half`
trick for `rotate_half_interleaved`. Worked out the math by hand: the reshape trick
returns its output in a permuted ("de-interleaved") channel order and never permutes
back, which GLM-MoE-DSA gets away with because its only consumer is a Q.K dot product
(invariant to a shared relabeling of both operands). DeepSeek V4 also rotates the value
stream (K==V here) and feeds the attention output straight into `o_a_proj`/`o_b_proj`,
which expect the true HF channel order, so reusing the trick without adding a
compensating re-interleave step (plus switching cos/sin prep from `repeat_interleave` to
`cat`-style duplication) would silently compute the wrong thing. Keeping the current
self-contained `rotate_half_interleaved`.

## `convert_rope_params_to_dict` overrides are dead code

`LagunaConfig.convert_rope_params_to_dict` (`laguna/configuration_laguna.py`) and
`DeepseekV4Config.convert_rope_params_to_dict` (`deepseek_v4/configuration_deepseek_v4.py`)
are both no-ops (`return kwargs`) in practice, not just in intent. Verified empirically for
both: deleting the method and reconstructing a config produces byte-identical
`rope_parameters` output. The reason is structural in both configs' `__init__`: `rope_theta`
and `rope_scaling` are consumed as named parameters and never forwarded into
`super().__init__()`'s `**kwargs`, and `self.rope_parameters` is unconditionally overwritten
by each config's own normalization method (`_nest_rope_parameters` / `_normalize_rope_parameters`)
immediately after `super().__init__()` returns, regardless of anything the base class's real
`convert_rope_params_to_dict` (in `transformers.modeling_rope_utils.RotaryEmbeddingConfigMixin`)
might have done to `self.rope_parameters` while `super().__init__()` was running.

DeepSeek V4's override was copied from Laguna as the closest existing precedent for a model
with per-layer-type nested rope parameters, inheriting the same dead code.

Decision: keep both for now, don't touch either in isolation. Remove both overrides together
in a single follow-up commit once there's time to also re-verify against `from_pretrained`/
`from_dict` checkpoint-loading paths (which pass arbitrary `**kwargs` differently than direct
`__init__` construction does, and weren't part of the empirical check above).
