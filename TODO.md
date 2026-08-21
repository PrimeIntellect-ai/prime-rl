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
