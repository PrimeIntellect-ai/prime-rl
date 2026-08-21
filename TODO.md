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
- [x] 7. Decoder layer + model classes + state-dict conversion chain, wiring everything above together

All seven steps are done, which gets a minimal working model: `DeepseekV4ForCausalLM` is
registered in `trainer/models/__init__.py`, dispatches through `AutoModelForCausalLMPrimeRL` /
`get_model()`, loads an HF checkpoint through `converting_deepseek_v4.conversion_chain` with no
missing or unexpected keys, and matches HF's forward and backward to the float32 floor
(`tests/unit/train/models/test_deepseek_v4.py`). What follows is what it still takes to call it
production ready.

Fixed during review: `init_buffers_post_meta` unconditionally zeroed `MoE`'s persistent
`expert_bias` buffer, which by that point already holds the real value `dcp_load` loaded from a
checkpoint (`to_empty` -> `dcp_load` -> `init_buffers_post_meta`, per `trainer/model.py`), so it
silently discarded a checkpoint's load-balancing bias on every load. `tokens_per_expert` (a
non-persistent buffer, never in a checkpoint) is still correctly reset. The same bug exists in
`laguna/modeling_laguna.py` (copied from there) and is tracked/fixed on its own branch
(`fix/laguna-expert-biases`), independent of this port.

Open items:

- **YaRN on the compress RoPE branch is not wired up.** `DeepseekV4Config._nest_rope_parameters`
  forces `rope_type="default"` for both `main` and `compress`. Real checkpoints use YaRN
  (`factor=16`, `attention_factor=1.0`, `rope_theta=160000.0`) for `compress`. Wiring it up also
  needs HF's `validate_rope` override (the base validator keys off `layer_types`, not the
  `main`/`compress` labels). This is the one open item that changes the numbers a real
  checkpoint produces, so it blocks any run against real V4 weights.
- **No MTP.** `num_nextn_predict_layers` is not carried over and no multi-token-prediction head is
  built (HF does not build one either). The conversion chain drops `mtp.*` keys at either nesting
  depth, mirroring HF's `_keys_to_ignore_on_load_unexpected`.
- **Everything is single-document.** `build_sliding_window_mask`, both compressors, and
  `DeepseekV4Model`'s default `position_ids` all assume one document per row (no `cu_seqlens`
  awareness). `DeepseekV4Model.forward` accepts `seq_lens` / `seq_lens_are_pre_shard` to satisfy
  the trainer's contract and ignores them, so a packed multi-document batch would let windows and
  compression bleed across document boundaries. Fixing it means either a `cu_seqlens`-aware
  mask and compressor, or a flash-attention path with `window_size` (attention is eager-only
  today, since the per-head sink logit has no flash-attention equivalent in prime-rl's vendored
  kernels).
- **The compressors and attention are stateless (no KV cache), by design.** prime-rl only runs a
  single forward + backward over a full sequence, never `generate()`, so `DeepseekV4HCACache`/
  `DeepseekV4CSACache` are not ported. Only relevant again if prime-rl grows incremental decode.
- **The Lightning Indexer gets no gradient**, in both HF and prime-rl (its parameters only reach
  the loss through non-differentiable top-k indices; pinned by
  `test_csa_indexer_selection_is_not_differentiable`). DeepSeek trains it with a separate
  auxiliary distillation loss not implemented here, so an RL/SFT run leaves it frozen at
  checkpoint values. Fine for fine-tuning, not for pre-training.
- **Router replay wins over the hash table.** An explicit `routed_experts` (recorded by the
  inference engine) takes precedence over `tid2eid[input_ids]` in a hash layer. The two agree as
  long as the engine implements hash routing; if a future engine reports zeros for those layers
  instead, the trainer would silently follow the zeros. The alternative is dropping
  `routed_experts` for hash layers, at the cost of a per-layer special case in the model forward.
- **A missing `tid2eid` fails silently.** It is a persistent buffer that no `init_weights` can
  reconstruct: zeros mean every token routes to expert 0. Nothing in the loading path checks that
  it actually came from the checkpoint, and it should say so loudly when it did not.
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
- **No vLLM kernel weight transfer.** `convert_layer_to_vllm_kernel` is not overridden, so the
  base class's `NotImplementedError` stands, as it does for `nemotron_h` and `laguna`. Serving a
  trained V4 through the NIXL transport needs a real implementation, and it has no precedent to
  copy: the fused `gate_up_proj` layout and the grouped output projection are both new.
- **`n_shared_experts=0` diverges from HF.** HF's `DeepseekV4SparseMoeBlock` always builds a
  shared expert; `n_shared_experts` is carried by its config and read nowhere. prime-rl's
  `DeepseekV4MoE` builds one only when the field is positive, so at zero the two key sets
  disagree. Harmless for real checkpoints (V4 ships `n_shared_experts=1`), wrong for a
  hand-written config.
- **bfloat16 routing drifts from HF's.** prime-rl's router upcasts its scores to float32 while HF
  scores in the activation dtype, so in bfloat16 a few percent of tokens pick a different expert
  set and the logits diverge by ~10% of their scale. `test_deepseek_v4_float32` pins that this is
  the *only* remaining difference; `test_deepseek_v4` documents the bfloat16 bound.
- **`tests/unit/train/models/test_deepseek_v4_temp.py` is per-mechanism scaffolding.** It predates
  the model classes and still carries the only coverage of several internals (compressor window
  structure, indexer selection, grouped-mm experts). `test_deepseek_v4.py` covers the assembled
  model. Fold the still-useful half of the scratch file into it and delete the rest.

State-dict deltas, all forced by prime-rl's own `MoE`/router naming and all implemented in
`converting_deepseek_v4.py`: `mlp.gate.weight` -> `mlp.router.gate.weight`,
`mlp.gate.e_score_correction_bias` -> `mlp.expert_bias`, `mlp.shared_experts.*` ->
`mlp.shared_expert.*`, and, on the hash layers only, `mlp.gate.tid2eid` -> `mlp.tid2eid`. The
routed experts need **no** conversion: `mlp.experts.gate_up_proj`/`down_proj` already match HF's
own names and shapes, unlike every other prime-rl MoE. The two MoE layer types have different key
sets: a hash layer has `mlp.tid2eid` and no `mlp.expert_bias`, a standard one the other way round.

One structural note: `DeepseekV4Indexer` subclasses a `DeepseekV4DualSeriesCompressor`
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
