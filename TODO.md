# TODO

## Qwen3.5 GatedDeltaNet varlen patch is broken on transformers >= 5.13

`_patch_qwen3_5_linear_attn_varlen()` in `src/prime_rl/trainer/model.py` monkey-patches
`Qwen3_5DecoderLayer`/`Qwen3_5GatedDeltaNet` so packed RL/SFT batches don't leak Mamba-style
recurrent state across sequence boundaries. It's broken on transformers 5.15.0 in two ways,
and no existing test exercises this code path (`_patch_qwen3_5_linear_attn_varlen`,
`Qwen3_5GatedDeltaNet`, `GatedDeltaNet` have zero hits under `tests/`), so CI won't catch it.
Originally flagged by reviewer `dzautner` on PR #3055; confirmed and extended here.

1. **Renamed attribute.** The patch checks `self.layer_type != "linear_attention"`, but
   transformers >= 5.13 renamed this to `self.block_type`. Raises `AttributeError` the moment
   the patch runs. Fix: `getattr(self, "block_type", getattr(self, "layer_type", None))`.
2. **Kernel references no longer exist.** The patch calls `self.causal_conv1d_fn(...)` /
   `self.chunk_gated_delta_rule(...)` as bound methods; transformers moved these to
   module-level functions in `transformers.models.qwen3_5.modeling_qwen3_5`, each resolved at
   decoration time to a real kernel or a plain PyTorch reference
   (`@use_kernel_func_from_hub_with_fallback`). In this dev env, `causal_conv1d` isn't
   installed, so `causal_conv1d_fn` is guaranteed to be the reference implementation, which
   ignores `seq_idx`/packing entirely. `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (an
   installed alternative) does take `cu_seqlens`, but its signature doesn't match the current
   call site and its `cu_seqlens` correctness hasn't been verified here. (Checking which
   implementation resolved requires `transformers.utils.import_utils.resolve_internal_import`
   directly: `functools.wraps` makes the wrapper's `__module__`/signature always look like the
   reference impl regardless of what actually resolved.)

What still needs to happen: fix the rename with a `getattr` fallback; call the module-level
`causal_conv1d_fn` directly (keep prime-rl's existing per-segment conv1d loop rather than a
fast path that can't be reliably detected as safe); re-derive the `chunk_gated_delta_rule` call
against fla's real signature with an actual packed-vs-unpacked parity check; then add regression
tests for both (none exist today). Separately (dzautner, PR #3055, still open): `get_model()` in
`trainer/model.py` detects Qwen3.5 via a name-string check plus a `model_type` fallback that
only matches `"qwen3_5_moe*"`, so a dense Qwen3.5 checkpoint loaded through a generic local
alias silently skips the patch (no crash, just the state-leak bug back, showing up only as
elevated mismatch-KL).

## DeepSeek V4 port

Step-by-step plan, one commit each:

- [x] 1. Config + manifold-constrained hyper-connections (mHC)
- [x] 2. Rotary embedding + sliding-window attention
- [x] 3. Compressed Sparse Attention (CSA): compressor + Lightning Indexer
- [x] 4. Heavily Compressed Attention (HCA): compressor, no indexer
- [x] 5. Standard MoE (router, experts, shared expert)
- [x] 6. Hash-routed MoE (bootstrap layers)
- [ ] 7. Decoder layer + model classes + state-dict conversion chain, wiring everything above together

Open items step 7 needs to handle:

- **YaRN on the compress RoPE branch is not wired up.** `DeepseekV4Config._nest_rope_parameters`
  forces `rope_type="default"` for both `main` and `compress`. Real checkpoints use YaRN
  (`factor=16`, `attention_factor=1.0`, `rope_theta=160000.0`) for `compress`. Wiring it up also
  needs HF's `validate_rope` override (the base validator keys off `layer_types`, not the
  `main`/`compress` labels).
- **No MTP.** `num_nextn_predict_layers` is not carried over (HF doesn't instantiate it either);
  the conversion chain needs to drop `mtp.*` keys, same as `nemotron_h`.
- **Everything is single-document.** `build_sliding_window_mask`, both compressors, and
  `DeepseekV4Attention`'s default `position_ids` all assume one document per row (no
  `cu_seqlens` awareness). A packed multi-document batch would let windows/compression bleed
  across document boundaries. Step 7 needs to decide between a `cu_seqlens`-aware mask/compressor
  or a flash-attention path with `window_size` (currently attention is eager-only, since the
  per-head sink logit has no flash-attention equivalent in prime-rl's vendored kernels).
- **The compressors and attention are stateless (no KV cache), by design.** prime-rl only runs a
  single forward + backward over a full sequence, never `generate()`, so `DeepseekV4HCACache`/
  `DeepseekV4CSACache` are not ported. Only relevant again if prime-rl grows incremental decode.
- **The Lightning Indexer gets no gradient**, in both HF and prime-rl (its parameters only reach
  the loss through non-differentiable top-k indices; pinned by
  `test_csa_indexer_selection_is_not_differentiable`). DeepSeek trains it with a separate
  auxiliary distillation loss not implemented here, so an RL/SFT run leaves it frozen at
  checkpoint values. Fine for fine-tuning, not for pre-training.
- **Rotary buffers are computed eagerly in `__init__`.** `DeepseekV4RotaryEmbedding` needs an
  `init_buffers_post_meta` on step 7's `PreTrainedModel` subclass to re-derive them after
  meta-device loading, mirroring HF's `_init_weights` branch.
- **`_init_weights` is not wired up anywhere yet.** Every V4 module exposes its own
  `init_weights(init_std)`, but nothing calls them until step 7's `PreTrainedModel` subclass.
- **Hash layers need `input_ids` threaded down to them.** `DeepseekV4MoE.forward(x, input_ids,
  routed_experts)` asserts `input_ids is not None` when `mlp_layer_types[layer_idx] ==
  "hash_moe"`, so step 7's decoder layer and model must pass the ids through every block, as
  HF's do. A standard layer accepts and ignores them.
- **Router replay wins over the hash table.** An explicit `routed_experts` (recorded by the
  inference engine) takes precedence over `tid2eid[input_ids]` in a hash layer. The two agree as
  long as the engine implements hash routing; if a future engine reports zeros for those layers
  instead, the trainer would silently follow the zeros. Step 7 could drop `routed_experts` for
  hash layers instead, at the cost of a per-layer special case in the model forward.
- **A missing `tid2eid` fails silently.** It is a persistent buffer that no `init_weights` can
  reconstruct: zeros mean every token routes to expert 0. Step 7's loading path (meta device,
  then conversion) has to guarantee it comes from the checkpoint, and should say so loudly if it
  does not.
- **Hash layers have no load balancing.** They pass `load_balance_coeff=None`, so no
  `expert_bias` buffer exists (a frozen selection cannot be steered, and HF's
  `DeepseekV4HashRouter` has no `e_score_correction_bias` to load into one). `tokens_per_expert`
  is still accumulated, so `get_load_balance_stats` reports a `max_vio` for them that no
  mechanism can act on.
- **Three `DeepseekV4Config` fields are asserted, not supported, in `DeepseekV4MoE`**:
  `hidden_act` must be `"silu"`, `mlp_bias` must be `False` (the shared `MLP` never adds a bias
  regardless of the flag), `fp8` is rejected (the fp8 grouped GEMM assumes a different weight
  layout).
- **Expert parallelism and LoRA don't support `DeepseekV4Experts`'s fused weight layout yet.**
  torchtitan's `ExpertParallel._partition_fn` shards by the literal names `w1`/`w2`/`w3`, so
  `ep=True` needs a partition function for `gate_up_proj`/`down_proj`; `lora.py` dispatches on
  the three known expert classes, so a LoRA run would silently leave the routed experts frozen.
  `GptOssGroupedExperts` has the same EP gap.
- **No router aux loss.** `output_router_logits`, `router_aux_loss_coef`, `router_jitter_noise`
  are carried by the config and read by nothing.
- **`tests/unit/train/models/test_deepseek_v4_temp.py` is scaffolding.** Fold it into a proper
  `test_deepseek_v4.py` full-model parity test once the model classes exist.

State-dict deltas step 7's conversion chain needs (all forced by prime-rl's own `MoE`/router
naming, identical to what `glm4_moe`/`laguna` already do): `mlp.gate.weight` ->
`mlp.router.gate.weight`, `mlp.gate.e_score_correction_bias` -> `mlp.expert_bias`,
`mlp.shared_experts.*` -> `mlp.shared_expert.*`, and, on the hash layers only,
`mlp.gate.tid2eid` -> `mlp.tid2eid`. The routed experts need **no** conversion:
`mlp.experts.gate_up_proj`/`down_proj` already match HF's own names and shapes, unlike every
other prime-rl MoE. The two MoE layer types have different key sets: a hash layer has
`mlp.tid2eid` and no `mlp.expert_bias`, a standard one the other way round.

One structural note for step 7: `DeepseekV4Indexer` subclasses a `DeepseekV4DualSeriesCompressor`
base shared with `DeepseekV4CSACompressor` (HF's two classes run byte-identical compression code
at different `head_dim`s). `DeepseekV4HCACompressor` deliberately does **not** share that base:
non-overlapping windows, `head_dim`-wide (not `2*head_dim`) projections.

Considered and rejected: reusing GLM-MoE-DSA's `apply_rope_interleave_single`
(`glm_moe_dsa/sparse_mla_attention.py:56-63`) for `rotate_half_interleaved`. Its reshape trick
returns output in a permuted channel order and never permutes back, which is safe for GLM-DSA
(its only consumer is a Q.K dot product, invariant to a shared relabeling) but not for V4, which
also rotates the value stream and feeds the result straight into `o_a_proj`/`o_b_proj` expecting
true HF channel order.

## `GptOssGroupedExperts` crashes with `use_grouped_mm=False`

Found by accident while porting DeepSeek V4's experts (which hit the same constraint and worked
around it correctly). `layers/moe.py`'s `expert_parallel` decorator wraps functions with a fixed
`(w1, w2, w3, x, num_tokens_per_expert)` signature, but `GptOssGroupedExperts`'s for-loop path
calls its wrapped function with 6 positional args (`gate_up_proj`, `gate_up_proj_bias`,
`down_proj`, `down_proj_bias`, `x`, `num_tokens_per_expert`). Confirmed empirically: raises
`TypeError: wrapper() takes from 4 to 5 positional arguments but 6 were given` immediately, every
time. No existing test exercises `GptOssGroupedExperts` with `use_grouped_mm=False` (`gpt_oss`
has no test file at all), so nothing has caught it. Unrelated to the DeepSeek V4 port; needs its
own fix in shared code.

## `convert_rope_params_to_dict` overrides are dead code

`LagunaConfig.convert_rope_params_to_dict` and `DeepseekV4Config.convert_rope_params_to_dict`
are both no-ops (`return kwargs`) in practice: `rope_theta`/`rope_scaling` are consumed as named
`__init__` params and never reach `super().__init__()`'s `**kwargs`, and `self.rope_parameters`
is unconditionally overwritten by each config's own normalization method right after
`super().__init__()` returns. Verified empirically for both (deleting the method changes
nothing). DeepSeek V4's copy was inherited from Laguna's precedent. Decision: keep both for now;
remove together in one follow-up commit, after also checking `from_pretrained`/`from_dict`
checkpoint-loading paths (not covered by the empirical check above).
