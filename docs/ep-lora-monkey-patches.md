# Expert Parallelism + LoRA: Monkey Patches

vLLM (as of v0.16) does not support combining Expert Parallelism (EP) with LoRA on
Mixture-of-Experts models. `FusedMoEWithLoRA.__init__` explicitly blocks it:

```python
assert not self.base_layer.use_ep, "EP support for Fused MoE LoRA is not implemented yet."
```

We work around this with seven monkey patches in
`src/prime_rl/inference/patches.py::monkey_patch_fused_moe_lora_ep()`.

## Background

In a standard TP-only setup, every GPU holds a shard of **every** expert's weights.
With EP, each GPU only holds a subset of experts (e.g. 10 out of 160 for GLM-4.7 on
16 GPUs). Tokens are routed to the correct GPU via all-to-all communication
(dispatch), processed by the local experts, then sent back (combine).

The LoRA code was written assuming every GPU has every expert. The patches fix the
places where this assumption breaks.

### Key data structures

- **`expert_map`**: A tensor of shape `[global_num_experts]` on each rank. Entry `i`
  is the local index of global expert `i` if this rank owns it, or `-1` otherwise.
  For example, on a rank owning experts 20-29: `expert_map[25] = 5`,
  `expert_map[0] = -1`.

- **`packed_modules_mapping["experts"]`**: A list with one entry per
  `(expert, weight_type)` pair. For GLM-4.7 (160 experts, 3 weight matrices: w1, w2,
  w3), this has 480 entries. This mapping is global and shared across all ranks.

- **`lora_a_stacked` / `lora_b_stacked`**: Flat lists on `FusedMoEWithLoRA` holding
  the LoRA weight tensors. With EP, these are sized for **local** experts only
  (e.g. 10 experts × 3 weights × max_loras = 60 entries, not 480).

## Patch 1: Remove the EP assertion

**Problem**: `FusedMoEWithLoRA.__init__` raises an assertion when EP is active.

**Fix**: Replace `__init__` with a version that skips the assertion while reproducing
all other initialization (setting `base_layer`, `tp_size`, `tp_rank`, `device`,
`_w13_slices`, and calling `_inject_lora_into_fused_moe()`).

## Patch 2: Fix modular kernel initialization for EP

**Problem**: `_inject_lora_into_fused_moe` creates the modular MoE kernel that the
LoRA decorators will wrap. When `supports_internal_mk` is False (the common
unquantized path), it hardcodes `MoEPrepareAndFinalizeNoEP` as the prepare/finalize
strategy. With EP, the kernel needs the EP-aware prepare/finalize (dispatch/combine
via all-to-all), not the no-EP one.

**Fix**: Before calling the original `_inject_lora_into_fused_moe`, call
`self.base_layer.maybe_init_modular_kernel()`. This initializes the modular kernel
with the correct EP-aware prepare/finalize strategy. The LoRA decorators then wrap
this correct kernel instead of creating a broken no-EP one.

## Patch 3: Expert-slicing helpers

**Problem**: LoRA weights are loaded and passed around with the global expert
dimension (e.g. shape `[160, ...]`), but with EP each rank only stores local experts
(e.g. 10).

**Fix**: Two helper methods added to `FusedMoEWithLoRA`:

- `_get_local_expert_indices()`: Reads `expert_map` and returns the global indices
  of experts owned by this rank (e.g. `tensor([20, 21, 22, ..., 29])`).

- `_slice_experts(weight, local_expert_indices)`: Slices a weight tensor from global
  to local expert dimension. Includes a guard: if the weight is already local-sized
  (e.g. from a dummy LoRA), it skips slicing. Also handles device mismatches between
  the weight and `expert_map` indices.

## Patch 4 & 5: Slice LoRA weights in `set_lora`

**Problem**: `FusedMoEWithLoRA.set_lora(index, lora_a, lora_b)` receives lists of
LoRA weight tensors with the **global** expert count in dimension 0. It copies them
into `lora_a_stacked` / `lora_b_stacked`, which are sized for **local** experts.
This causes shape mismatches or silent corruption.

**Fix**: Before calling the original `set_lora`, slice each weight in `lora_a` and
`lora_b` from global to local experts using the helpers from Patch 3. This applies
to both `FusedMoEWithLoRA` (patch 4) and `FusedMoE3DWithLoRA` (patch 5).

```
# Before: lora_a[i].shape = [160, rank, hidden_size]  (global)
# After:  lora_a[i].shape = [10, rank, hidden_size]   (local)
```

## Patch 6: Zero-initialize `expert_ids` in block alignment

**Problem**: `PunicaWrapperGPU.moe_lora_align_block_size` allocates `expert_ids`
with `torch.empty` (uninitialized memory). The C++ kernel
`ops.moe_lora_align_block_size` fills in entries for active LoRA slots but skips
inactive ones, leaving garbage values. Later, `expert_map[expert_ids]` uses these
as indices — garbage values like `741924` index out of bounds in `expert_map`
(size 160).

Without EP this is harmless because there's no `expert_map` remapping. With EP,
the remapping step crashes.

**Fix**: Replace `torch.empty` with `torch.zeros` for `expert_ids`. Zero is always
a valid index into `expert_map` (expert 0 exists on some rank). The values in
inactive slots are never used for actual computation, so the zero is just a safe
placeholder.

## Patch 7: Fix `create_dummy_lora` for EP

**Problem**: `LoRAModelManager.create_dummy_lora` creates zero-weight LoRA adapters
for CUDA graph warmup and profiling. For packed modules like MoE experts, it
iterates over `packed_modules_mapping["experts"]` and accesses
`module.lora_a_stacked[i]` for each entry.

The mapping has 480 entries (160 experts × 3 weights, global), but with EP,
`lora_a_stacked` only has entries for local experts (e.g. 30 entries for 10
experts × 3 weights). When `i >= 30`, it's out of bounds.

**Fix**: Before calling the original `create_dummy_lora`, temporarily truncate
`packed_modules_mapping["experts"]` to match the local expert count. After the
call, restore the original mapping. This is safe because:

- Only the iteration count changes; the dummy weights are all zeros regardless.
- The mapping is restored immediately after, so subsequent operations see the
  full global mapping.
- This only runs once at startup (during warmup), not on the inference hot path.

## Known limitations

- **CUDA graph capture with EP+LoRA crashes** with `cudaErrorIllegalAddress` in
  vLLM's fused MoE LoRA Triton kernel at ~66% of graph capture. This appears to be
  a bug in the upstream kernel, not in our patches. Workaround: use
  `enforce_eager = true`.

- **All patches are applied unconditionally** at import time
  (`src/prime_rl/inference/vllm/worker/__init__.py`) because the worker `__init__`
  runs before the vLLM config is available. The patches are safe no-ops when EP
  is not active (all guards check `self.base_layer.use_ep`).
