# TODO

## Qwen3.5 patch detection misses a dense checkpoint behind a generic alias

`get_model()` in `trainer/model.py` decides whether to apply the Qwen3.5 patches via a
name-string check plus a `model_type` fallback that only matches `"qwen3_5_moe*"`. A dense
Qwen3.5 checkpoint loaded through a generic local alias (e.g. `/checkpoints/model4b`) silently
skips them: no crash, just the state-leak bug that `_patch_qwen3_5_linear_attn_varlen` exists to
prevent, showing up only as elevated mismatch-KL. Flagged by reviewer `dzautner` on PR #3055 and
still unaddressed. Fix: extract the detection into a small pure function checking both
`model_type` and nested `text_config.model_type` for `qwen3_5`/`qwen3_5_moe` prefixes, and add a
unit test for the dense generic-alias case.

Nothing under `tests/` exercises the patched path at all (`_patch_qwen3_5_linear_attn_varlen`,
`Qwen3_5GatedDeltaNet`, `GatedDeltaNet` have zero hits), so CI would not catch a regression in
it either. Worth a packed-vs-unpacked GDN parity test alongside the fix above.

## `mini_moe.py`'s DeepSeek V4 preset was pulled, and needs a different writer to come back

`scripts/mini_moe.py` is back to zero diff against `main`, so `--arch deepseek_v4` no longer
exists. It worked by building an HF model, calling `save_pretrained`, then rewriting four key
families in the result, because `save_pretrained` routes weights through `transformers`' reverse
conversion and the `"deepseek_v4"` entry in `transformers/conversion_mapping.py` mis-reverts them:
its `^embed\.weight$` / `^hc_head_*$` patterns never match `state_dict()` keys, which carry the
`model.` prefix that on-disk names do not, and its broad `.norm.` rule (meant for the compressor's
norm) turns attention's `kv_norm` into `norm`. vLLM's `hf_to_vllm_mapper` assumes the real format
and fails with `KeyError: 'hc_head.hc_base'` without the repair.

Reinstating the preset should not reinstate the repair. Write the weights with prime-rl's own
`convert_to_hf`, which is what `convert_state_dict_to_hf` (`utils/weights.py`) already broadcasts
every real checkpoint with, and `save_pretrained` is out of the loop entirely. The bug is still
present on `transformers` `main`, so a version bump will not fix it; worth filing upstream.

Note that resetting the file also dropped `a4b934809 fix(scripts): pass seq_lens from mini_moe's
verify`. That was never DeepSeek-V4-specific (`verify()` on `main` raises `TypeError` for every
preset) and `origin/fix/mini-moe-script` owns the generic repair.

## Two DeepSeek V4 shims to delete on the next transformers bump

Transformers added V4's `compressed_sparse_attention` / `heavily_compressed_attention` to the
vocabulary `PretrainedConfig.validate_layer_type` checks in 5.15. Below that, every DeepSeek V4
config raises on construction, so `utils/transformers_compat.py` extends the tuple by hand and
both halves of the repo call it: the trainer at config-module import, and the inference side
through `monkey_patch_deepseek_v4_allowed_layer_types` in `inference/patches.py` (vLLM's own
`DeepseekV4Config` has no override of its own, and vLLM declares only `transformers>=5.5.3`).
Once the pin reaches 5.15 or later, both call sites go away; `DEEPSEEK_V4_LAYER_TYPES` moves
back next to the validator that reads it.

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
(`tests/unit/train/models/test_deepseek_v4_hf.py`, which runs only under the transformers 5.15
override). What follows is what it still takes to call it
production ready.

Fixed during review: `init_buffers_post_meta` unconditionally zeroed `MoE`'s persistent
`expert_bias` buffer, which by that point already holds the real value `dcp_load` loaded from a
checkpoint (`to_empty` -> `dcp_load` -> `init_buffers_post_meta`, per `trainer/model.py`), so it
silently discarded a checkpoint's load-balancing bias on every load. `tokens_per_expert` (a
non-persistent buffer, never in a checkpoint) is still correctly reset. The same bug exists in
`laguna/modeling_laguna.py` (copied from there) and is tracked/fixed on its own branch
(`fix/laguna-expert-biases`), independent of this port.

End-to-end verified via a real `uv run sft` run against a tiny local `DeepseekV4Config`-only
checkpoint (no weights, `model.debug.random_init=true`, all scratch under `/tmp`, nothing
committed): FSDP, activation checkpointing, the fused chunked LM head, hash-routed MoE, standard
MoE, and all three attention layer types all run cleanly with finite, varying loss and nonzero,
varying gradient norms. Along the way, found (not DeepSeek-V4-specific): `debug.random_init=True`
never actually randomizes weights, it only fixes a few named buffers via `init_buffers_post_meta`
and then skips checkpoint loading (`trainer/model.py::load_dcp_from_hf`); parameters are left as
whatever `model.to_empty(device=...)` happens to leave in memory, which was all-zero for this tiny
model in a fresh CUDA process and produced exactly-zero gradients and constant loss end to end.
Confirmed by direct diagnostic: manually re-randomizing weights after `load_dcp_from_hf` restores
correct gradients (128/140 nonzero; the 12 dead ones are the Lightning Indexer's, expected). Being
fixed separately (shared `trainer/model.py` code, affects every model's random-init debug path),
to land in this worktree once ready.

Open items:

- **No MTP.** `num_nextn_predict_layers` is not carried over and no multi-token-prediction head is
  built (HF does not build one either). The conversion chain drops `mtp.*` keys at either nesting
  depth, mirroring HF's `_keys_to_ignore_on_load_unexpected`.
- **Per-document compression makes HCA inert on short rollouts.** The compressors now pool and
  number their entries per document, so a packed batch matches running each rollout alone, which
  is what vLLM serves. That closes the packing defects, but it also means a document shorter than
  a compress rate gets no compressed entry at all: at HCA's production `compress_rate` of 128
  against 77-token rollouts, roughly 18 of the 43 layers contribute nothing beyond their local
  sliding window. This is a property of the architecture under short-rollout RL, not a bug, and it
  is what vLLM already does. Whether it costs quality at the production rollout length is not yet
  measured.
- **The mismatch-KL residual has not been re-measured.** On the local RL smoke, 20 steps, same
  checkpoint and config either side, clipping the sliding window took `mismatch_kl/all/mean` from
  0.507 to 0.063 (worst per-token 106.15 to 3.39), so the window was the dominant term. The
  residual 0.063 was roughly 4x the 0.015 merge bar and the compressors were expected to account
  for most of it. Per-document compression has since landed but the smoke has not been re-run, and
  this sandbox's fp8 KV cache and torch-fallback output projection contribute an unapportioned
  share that will not appear on the cluster.
- **A RoPE term has been removed from the mismatch and the residual should be re-measured.**
  vLLM's `build_deepseek_v4_rope` branches the RoPE base on `compress_ratio` but not the scaling,
  so it handed YaRN to the two pure sliding-window layers (0 and 1 of 43 on the real checkpoint),
  where DeepSeek's own reference disables it. Fixed locally by
  `monkey_patch_deepseek_v4_rope_disable_yarn_on_sliding_layers` in `inference/patches.py`, which
  also flattens the nested `main`/`compress` schema that vLLM's config shim cannot read (that one
  dropped YaRN from all 43 layers on any `save_pretrained` round trip). Both are pinned against the
  reference's own `precompute_freqs_cis` by `tests/unit/inference/test_deepseek_v4_rope_patch.py`
  and written up for upstream in `ISSUES_vllm_dsv4.md`. The sliding-layer error was bounded at
  0.037 rad of phase inside the 128-token window, so it was likely a small term in the 0.063
  residual above rather than the explanation for it.
- **No context parallelism.** CP hands the model pre-shard (global) document boundaries, which
  cannot address a dense local mask built over post-shard positions. `get_model` rejects `cp > 1`
  for this model and `DeepseekV4Model.forward` rejects `seq_lens_are_pre_shard`. Lifting this
  means a flash-attention path with `window_size` instead of a dense mask, which needs a
  flash-attention equivalent for the per-head sink logit that prime-rl's vendored kernels lack.
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
- **LoRA doesn't support `DeepseekV4Experts` yet.** `lora.py` dispatches on the three known
  expert classes, so a LoRA run would silently leave the routed experts frozen.
- **Expert parallelism for `DeepseekV4Experts` works.** The reasoning changed with #3411 even
  though the conclusion did not. This note used to say EP worked because the experts held
  literal `w1`/`w2`/`w3` params matching the names torchtitan's `ExpertParallel._partition_fn`
  sharded by. That class is gone; `ExpertWeightParallel._partition_fn` now shards every
  `named_parameters(recurse=False)` on `Shard(0)` regardless of name, so the stacked
  `gate_proj`/`up_proj`/`down_proj` the experts inherit from `GroupedExperts` are covered by
  construction. Re-verified after the port with an `ep=8` SFT run on
  a mini-checkpoint config since removed from the repo: finite loss (12.66, 12.24,
  13.01), nonzero varying grad norms, no NaNs, 12.3 GiB peak. That run uses the default torch
  dispatch and sets no `[model.ac]`, so it does not speak to the DeepEP-versus-torch deadlock
  `sft.toml` documents under `ac.mode = "full"`.
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
- **`tests/unit/train/models/test_deepseek_v4_temp.py` is the per-mechanism suite.** It carries
  the only coverage of several internals (compressor window structure, indexer selection,
  grouped-mm experts) plus the packed-batch invariants; `test_deepseek_v4.py` covers the
  assembled model. Its HF-oracle half lives in `test_deepseek_v4_temp_hf.py`.

Investigated whether the eager-only attention above (`modeling_deepseek_v4.py:128-133`: head_dim
512 exceeds FlashAttention's 256 cap, no SDPA equivalent for the per-head sink logit,
FlexAttention's `BlockMask` can't cover the compressed-KV entries CSA/HCA concatenate onto the KV
axis) has a real fix available. Short answer: not yet, and the gap is ecosystem-wide, not specific
to this port: DeepSeek-V4 GA'd 2026-08-13, and torchtitan's own in-flight port (PR #3634) ships
the same "small operators" attention today, for the same three reasons. DeepSeek's own FlashMLA
(`deepseek-ai/FlashMLA` @ `15f13e5`) is a dead end for training this: `ValueError("SM100 bwd
doesn't support GQA now")` still blocks any backward at `num_kv_heads=1`, and no kernel there
exposes a differentiable `attn_sink` (only the forward-only decode/sparse-prefill paths have one).
Two live OSS candidates exist, neither drop-in: `meta-pytorch/attention-gym`'s `selected_attention`
(Triton backend) has a correct, hand-verified `attn_sink` backward and native shared-KV support,
but explicitly raises `NotImplementedError` for backward at `head_dim=512` (this port's exact head
dim) as a real Blackwell shared-memory/register wall, not unwritten code (PR #240: "existing
backward shared-memory OOR"), so forward-only there today, with no committed timeline to close it
despite an explicit "DSV4-like" benchmark in the same PR. NVIDIA-NeMo/Automodel's TileLang
sparse-MLA kernel (`nemo_automodel/components/models/deepseek_v4/kernels/tilelang_sparse_mla_
{fwd,bwd}.py`) does have a correct sink backward and reuses the same TileLang toolchain already
vendored here for `glm_moe_dsa`, but it's a distinct sibling kernel (not a patch to the existing
vendored file), needs a real mask-to-top-k-indices conversion to replace the current concat-and-mask
design, and has its own already-hit Blackwell TileLang codegen bug distinct from the NaN bug the
existing vendored kernel already works around.

Even a perfect kernel swap for the main attention op would not remove every `O(seqlen²)` term.
Per-layer naive quadratic-memory components (`B`=batch, `H`=64=`num_attention_heads`,
`H_idx`=64=`index_n_heads`, `r_csa`=4, `r_hca`=128):

| Component | Layers | Scaling | Notes |
|---|---|---|---|
| Main local attention scores | all | `B·H·S²` | dominant term; hits even sliding-window layers since eager code never physically truncates K |
| Main compressed/remote scores | CSA, HCA | `B·H·S²/r` | CSA: `B·16·S²`; HCA: `B·0.5·S²` |
| Lightning Indexer scoring | CSA only | `B·H_idx·S²/r_csa` | `B·16·S²`, the same order as CSA's compressed-attention term, since `index_n_heads`=`num_attention_heads`=64 by default; only FLOPs are cheaper (narrower `index_head_dim`), not memory |
| CSA/HCA `block_bias` construction | CSA, HCA | `B·S²/r` | no head multiplier; eliminated entirely by passing indices directly to a fused kernel instead |
| Sliding-window mask construction | all | `S²` | no `B`/`H` factor; smallest term |

Confirmed directly against `tilelang_indexer_fwd.py`: it tiles away the per-head intermediate (a
real ~64x memory win over the naive PyTorch scoring path) but still materializes the final dense
`[seq_len, seq_len_kv]` score tensor before `torch.topk` (`layers.py:839`). Exact top-k selection
fundamentally requires scoring every candidate, so the indexer's forward memory stays genuinely
quadratic in `seq_len` no matter how the kernel is engineered; only its backward is linear (touches
only the selected top-k entries). Getting the indexer to true linear memory would need an
approximate or hierarchical top-k, which nothing surveyed implements. Fine at `seq_len=2048` (this
port's current validation target), but will matter at the million-token context
lengths DeepSeek-V4's own paper targets. Revisit attention-gym's Triton backend if/when D=512
backward lands upstream (no tracking issue exists there to watch instead); the NeMo-Automodel
TileLang path is usable sooner at the cost of writing the mask-to-indices conversion.

State-dict deltas, all forced by prime-rl's own `MoE`/router naming and all implemented in
`converting_deepseek_v4.py`. prime-rl owns the router one level above HF, so everything HF keeps
under `gate` moves onto `router`: `mlp.gate.weight` -> `mlp.router.gate.weight`,
`mlp.gate.e_score_correction_bias` -> `mlp.router.selection_bias`, and, on the hash layers only,
`mlp.gate.tid2eid` -> `mlp.router.tid2eid`. The shared expert is named in the singular
(`mlp.shared_experts.*` -> `mlp.shared_expert.*`). The routed experts need **no** conversion:
`mlp.experts.gate_up_proj`/`down_proj` already match HF's own names and shapes, unlike every other
prime-rl MoE. The two MoE layer types have different key sets: a hash layer has
`mlp.router.tid2eid` and no `mlp.router.selection_bias`, a standard one the other way round.

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

## DeepSeek V4: blockers for the real cluster run

Found while getting the local 4-GPU validation working against a mini checkpoint with real
config values. None of these are visible from the test suite; all of them are on the path of
`examples/advanced/deepseek-v4-flash/kl-check.toml`.

- **The trainer cannot load the real checkpoint.** `deepseek-ai/DeepSeek-V4-Flash-0731` ships
  `quantization_config: {quant_method: "fp8", fmt: "e4m3", weight_block_size: [128, 128],
  scale_fmt: "ue8m0", activation_scheme: "dynamic"}`, i.e. FP8 weights plus `*.weight_scale_inv`
  tensors on disk (verified by fetching the config directly from the Hub). `load_dcp_from_hf`
  (`trainer/model.py:963`) has no dequantization step anywhere in the conversion chain, and
  `DeepseekV4MoE` rejects `fp8` outright. Both `rl.toml` and `kl-check.toml` name that
  checkpoint, so they would fail at weight load. Needs an fp8-plus-block-scales to bf16
  dequantizing load path; `trainer/models/glm_moe_dsa/converting_glm_moe_dsa.py` is the nearest
  precedent but goes the other direction (bf16 to fp8, for weight transfer).
- **`--use-deep-gemm` is mandatory, not an optimization, at least on SM120.**
  `InferenceConfig.use_deep_gemm` defaults to False, which sets `VLLM_USE_DEEP_GEMM=0`, which
  makes `is_deep_gemm_supported()` False, which makes `mhc_pre_tilelang` take
  `_tilelang_hc_prenorm_gemm`. That function's `x.shape[0] >= 1024` branch,
  `hc_prenorm_gemm_block_m_tilelang`, is numerically wrong on SM120: measured against a torch
  reference at `hidden_size=4096` it agrees to a relative 3e-7 for every token count below 1024
  and returns garbage (relative error about 1.0, `sqrsum` left at zero) from exactly 1024 up. So
  without the flag, any forward pass with 1024 or more tokens gets silently corrupt mHC
  activations.

  Checked on Hopper (H200): a 2015-token prompt against the real checkpoint with
  `--use-deep-gemm` produced coherent, non-garbled output, no NaNs. But on Hopper the flag turns
  out to be required for an unrelated, more basic reason too: `deepseek-ai/DeepSeek-V4-Flash-0731`
  quantizes with `scale_fmt: "ue8m0"`, and vLLM's CUTLASS-backed dense FP8 linear kernel
  (`Fp8BlockScaledMMLinearKernel` in `vllm/model_executor/kernels/linear/scaled_mm/
  BlockScaledMMLinearKernel.py`) hardcodes `use_ue8m0=False` for its activation quantizer,
  regardless of the checkpoint's actual weight-scale format. With deep_gemm off, dense FP8
  linears route to that CUTLASS kernel and crash outright during the memory-profiling forward
  pass: `RuntimeError: dispatch_scaled_mm, .../scaled_mm_helper.hpp:17`. DeepGEMM's kernel already
  handles UE8M0 (logged at startup: "Detected quantization_config.scale_fmt=ue8m0; enabling
  UE8M0 for DeepGEMM"), so `--use-deep-gemm` sidesteps this entirely. Net effect: on this
  checkpoint the flag isn't just about avoiding silent corruption above 1024 tokens, it's
  required to boot at all. `examples/advanced/deepseek-v4-flash/{inference,kl-check,sft}.toml`
  all set `use_deep_gemm = true`.
- **Filesystem weight broadcast silently downcast everything to bf16** (fixed here, but worth
  knowing it affected six models). `gather_weights_parallel` cast every DTensor to bf16 and the
  filesystem sender passed no exceptions, so `keep_in_fp32_for_weight_transfer`, which only the
  NIXL transport honored, did nothing. For DeepSeek V4 that meant vLLM received bf16
  hyper-connection parameters where its kernels assert fp32; the loader casts them back up
  instead of failing, so the only symptom was NaN logprobs out of the Sinkhorn normalization
  (surfacing as `400 - Out of range float values are not JSON compliant: nan` on every rollout).
  `glm4_moe`, `glm_moe_dsa`, `nemotron_h`, `qwen3_5`, and `qwen3_5_moe` all declare such lists
  too and were equally affected; whether they show quality loss or nothing visible depends on
  their kernels.
- **Filesystem weight broadcast cannot feed an FP8-served vLLM.**
  `gather_weights_parallel` (`utils/weights.py:153`) casts every DTensor parameter to bf16, and
  no `config.json` is written into the broadcast directory, so a broadcast contains no
  `*.weight_scale_inv` at all. Pushing that into a checkpoint-quantized FP8 vLLM model has
  nothing to load. The compatible combination is vLLM-side online quantization
  (`[inference.vllm] quantization`), which `finalize_layerwise_reload` re-applies on every
  broadcast; the `quantize_in_weight_transfer` route needs `convert_layer_to_vllm_kernel`, which
  DeepSeek V4 does not implement.
- **The checkpoint ships no chat template, and does not need one.** `deepseek-ai/DeepSeek-V4-Flash-0731`
  has no `chat_template.jinja` and no `chat_template` key in its `tokenizer_config.json`, but vLLM
  does not render this model through Jinja at all. `ModelConfig` auto-selects
  `tokenizer_mode = "deepseek_v4"` from the `DeepseekV4ForCausalLM` architecture
  (`vllm/config/model.py:625-634`) and dispatches to `DeepseekV4Renderer`
  (`vllm/renderers/registry.py:24`), which wraps DeepSeek's own reference encoder. The vLLM recipe
  documents the same thing. No template needs supplying, inventing or vendoring.
  Two claims previously recorded here were wrong and are corrected for the record: the `vllm-router`
  does **not** render chat messages (`vllm_router` 0.2.0 contains no template code at all and
  `mini_lb.py:359-361` forwards `/v1/chat/completions` straight to the backend), and the missing
  template was therefore never the reason a real-checkpoint run could not be rendered.
  What was genuinely missing on the prime-rl side was the parser mapping: `utils/parsers.py` had
  patterns for DeepSeek V3.1/V3.2 but none for V4, so `tool_call_parser` and `reasoning_parser`
  resolved `"auto" -> None` and were dropped before reaching vLLM, leaving `<think>` content
  unsplit from `content`. Fixed. The mini checkpoint needs no template either, for the same reason
  the real one does not: it saves `architectures: ["DeepseekV4ForCausalLM"]`, so vLLM picks the
  same renderer. `scripts/mini_moe.py` no longer writes one. The one component that did read a
  Jinja template is prime-rl's own client-side renderer, since `train_client_type="renderer"`
  means rollouts are tokenized locally and `[orchestrator.renderer] name = "default"` resolves to
  `DefaultRenderer`, which calls `tokenizer.apply_chat_template`. No config knob can fix that: the
  renderer pool builds its tokenizer with `load_tokenizer(path)`, which takes no template
  argument. That renderer now exists: `renderers` 0.1.11 (which `main` pins) ships
  `DeepSeekV4Renderer`, a native implementation rather than a wrapper around DeepSeek's
  `encode_messages`, and registers it in `MODEL_RENDERER_MAP` for
  `deepseek-ai/DeepSeek-V4-Flash-0731`. `sft.toml` picks it up through auto-resolution and needs
  no `[renderer]` section. `kl-check.toml` pins `name = "deepseek-v4"` explicitly, which is what
  auto-resolution would pick anyway. Verified against the real checkpoint's tokenizer:
  `name = "default"` raises `Cannot use chat template functions because tokenizer.chat_template
  is not set`, while `deepseek-v4` renders. Any config naming a local checkpoint path instead of
  the hub id needs the explicit pin, since the path is not a key in `MODEL_RENDERER_MAP` and
  auto-resolution would fall through to `DefaultRenderer`.
- **NCCL weight broadcast breaks with `data_parallel_size > 1`.** With two DP replicas the
  orchestrator logs `inference_world_size=1, gpus_per_server=1` and only DP rank 0 gets a
  receiver installed, so the collective RPC reaches DP1 and fails with `'Worker' object has no
  attribute 'nccl_broadcast_receiver'` while the API servers keep answering `/health`. Not root
  caused. Not on the DeepSeek V4 critical path, since this port has to use filesystem broadcast
  anyway (no `convert_layer_to_vllm_kernel`), but it is not DeepSeek-V4-specific either.
- **The real checkpoint's `config.json` is in the legacy flat format.** It carries a
  46-element `compress_ratios` list and no `layer_types` / `mlp_layer_types` / `rope_parameters`,
  while anything written by `save_pretrained` (including every filesystem broadcast directory) is
  in the new format. This asymmetry used to matter: `vllm/models/deepseek_v4/attention.py` reads
  `config.compress_ratios[layer_id]` directly, which vLLM's own shadow `DeepseekV4Config`
  (`vllm/transformers_utils/configs/deepseek_v4.py`) never derives from `compress_rates`/
  `layer_types` on its own. Resolved by writing `compress_ratios` directly into this repo's own
  `DeepseekV4Config` (`trainer/models/deepseek_v4/configuration_deepseek_v4.py`, right after
  `layer_types` is finalized) so every checkpoint this repo ever saves already carries the field,
  the same way the real checkpoint does — no vLLM-side runtime patch needed anymore
  (`monkey_patch_deepseek_v4_compress_ratios` in `inference/patches.py` has been removed).
  Confirmed against a real vLLM boot on both the real checkpoint and a freshly-rebuilt mini one.
- **The trainer's own `DeepseekV4Config` never read the legacy schema either, and this one
  blocked training, not just inference.** Separately from the vLLM-side fix above, the
  trainer's `DeepseekV4Config` (`trainer/models/deepseek_v4/configuration_deepseek_v4.py`) is
  registered via `AutoConfig.register("deepseek_v4", ...)` and used for every real-checkpoint
  trainer load, but its `__init__` never translated `compress_ratios`/`num_hash_layers` into
  `layer_types`/`mlp_layer_types` (unlike upstream `transformers`' own `DeepseekV4Config`,
  which does). Loading the real checkpoint built the wrong per-layer attention schedule
  outright: layers 0/1/41/42 should be `sliding_attention` but came out
  `heavily_compressed_attention`, and the rest of the stack was phase-shifted between
  `compressed_sparse_attention`/`heavily_compressed_attention`. This blocked any
  real-checkpoint SFT/RL trainer run, truncated or not. Fixed: `layer_types` now derives from
  `compress_ratios` via a reverse lookup over `compress_rates` when the modern fields aren't
  given, generalizing upstream's hardcoded translation to this repo's configurable
  `compress_rates` dict. Regression test:
  `test_deepseek_v4_config_translates_legacy_compress_ratios` in
  `tests/unit/train/models/test_deepseek_v4.py`. Verified directly against
  `deepseek-ai/DeepSeek-V4-Flash-0731`'s real `config.json` that `layer_types` now matches the
  transformers-native reference exactly.
- **The full 43-layer real checkpoint does not fit for training on a single 8xH200 node.**
  Confirmed by direct CUDA OOM under both CPU-offload strategies this repo supports, once the
  compress_ratios fix above unblocked the load in the first place. This is a hardware ceiling,
  not a bug: `examples/advanced/deepseek-v4-flash/{sft,kl-check}.toml` already spec
  `num_train_nodes = 4` (32 GPUs) for this exact checkpoint.
  - Plain FSDP+EP (`optimization_dtype = "float32"`) keeps the entire fp32 parameter shard
    resident on GPU: `284.6e9 params * 4 bytes / 8 GPUs ≈ 132.5 GiB/GPU` (total params derived
    from the real config's per-layer shapes: ~278B in routed/shared experts across 43 MoE
    layers, ~5.4B in attention incl. compressor/indexer, ~1.1B in the untied embedding/lm_head).
    Confirmed via a direct OOM at ~135 GiB allocated out of 143.8 GiB -- no room for even the
    first layer's activations.
  - `full_offload = true` + `optim.type = "sign_sgd"` (the lowest-memory optimizer full offload
    supports, since it carries no momentum/variance) move the fp32 masters/gradients/optimizer
    step to CPU RAM (confirmed: "132.40 GiB of pageable FP32 masters" logged per rank, matching
    the math above), but this does not reduce peak GPU memory here: the persistent bf16
    parameter shard (~66 GiB) plus the bf16 gradient buffer FSDP2 must hold before handing it to
    the CPU optimizer (~66 GiB) sum back to ~132 GiB. Confirmed via a second OOM (3 GiB
    allocation failure, ~132 GiB already allocated).
  - `debug.num_layers=4` was verified clean on the real checkpoint with `full_offload`: finite
    loss, `nan_count=0`, peak memory 31/139.8 GiB. This confirms dequantization, the
    compress_ratios fix, and `ep=8` are all correct together against real weights -- the ceiling
    above is specifically about the full 43-layer parameter count, not a correctness gap.
  - Separately, `load_dcp_from_hf`'s one-time HF -> prime-rl format conversion
    (`trainer/model.py`) reads and dequantizes the checkpoint's *entire* on-disk state dict on a
    single rank before any GPU work, regardless of `debug.num_layers` -- for this checkpoint that
    took ~90 minutes (156GB fp8/MXFP4 -> ~530GB dequantized), blowing past the default 3600s
    `dist_timeout_seconds`. Needs `--dist-timeout-seconds 14400`-ish on the first run only; the
    converted result is cached to `<snapshot>/prime/` and reused (with the default timeout) by
    every subsequent run regardless of truncation. Not fixed: the conversion could skip
    non-truncated-model keys to make truncated runs meaningfully cheap to load, but as a
    one-time, cached cost it wasn't worth the (model-agnostic, cross-architecture-risk) code
    change for this session's purposes.
  See the header comment in `examples/advanced/deepseek-v4-flash/sft.toml` for the single-node
  pre-flight command and the memory derivation.

## DeepSeek V4's routed-expert selection flips under small numerical perturbation

`GroupedExperts.forward` runs the routed experts through `torch._grouped_mm` in bfloat16 whatever
dtype the model runs in. PR #3411 removed the alternative: `layers/moe.py` used to pick between
`_run_experts_for_loop_impl`, per-expert `torch.matmul` with no dtype cast and so float32 when its
inputs were, and `_run_experts_grouped_mm_impl`, which cast to bfloat16. `use_grouped_mm=False`
selected the former and is what the V4 tests ran on. There is now one path, and all three
`model.moe.compute` options (`bf16`, `deepgemm_fp8`, `mxfp8`) are below float32.

Bfloat16 experts perturb the residual stream by well under 1%, which is enough to flip near-tied
top-k routing decisions in the score-routed layers, and a token routed to a different expert
changes its block output entirely.

The sensitivity is not V4-specific. Top-k selection over perturbed scores flips near-ties in any
routed MoE, and every model on main now runs its experts in bfloat16. What is V4-specific is that
anything noticed: V4's tests were written to float32 exactness (`rtol=1e-5, atol=1e-8`, and
`mini_moe`'s absolute `0.1`), while the other models' HF-parity tests already ran in bfloat16 with
bounds wide enough to swallow it. `test_qwen3_moe.py` asserts `atol=1e-0` on logits and `atol=2048`
on gradients, which is why #3411 could delete its `use_grouped_mm=False` and change nothing else.
Even after being recalibrated, V4's bounds stay far tighter than any sibling's. Note that the flip
counts below were measured on V4 only; that the mechanism generalises is inference from the shared
top-k code path, not a measurement.

Measured on the `scripts/mini_moe.py --arch deepseek_v4` checkpoint (float32 model, bfloat16
experts against float32 per-expert matmuls, 64 tokens, 4 of 16 experts each):

| layer | routing | selections flipped | block output deviation |
| --- | --- | --- | --- |
| 0, 1 | hash | 0 / 256 | 0.45%, 0.85% |
| 2 | score | 4 / 256 | 33% |
| 3 | score | 3 / 256 | 28% |
| 4 | score | 12 / 256 | 40% |

Final logits deviate by 1.19 on a scale of 6.7, against 3.1e-5 with float32 experts. The hash
layers are the control: their routing is a frozen table, so they show textbook bfloat16 error and
no flips at all.

Two consequences. `scripts/mini_moe.py --arch deepseek_v4` fails its `assert max_diff < 0.1` for
this reason and is left failing deliberately: the threshold is a precision bound, what breaks it
here is discrete, and raising it to fit would only hide the next real regression. Note the on-disk
to prime to on-disk weight roundtrip further down `verify()`, which is the check that actually
covers the conversion chain, never runs while that assert fires.

The one that matters more: the trainer and vLLM run different MoE kernels, so the same mechanism
should make them disagree on a comparable fraction of expert selections, and that lands directly on
the mismatch-KL this port is judged by. Nothing here measures it; that needs the real checkpoint
and a `kl-check.toml` run, and is worth doing before trusting any KL number. If it does bite, the
lever already exists: `MoE.forward` accepts `routed_experts`, so a served rollout's routing
decisions can be replayed rather than recomputed.

## The pinned `deep_gemm` wheel now loads, so its missing SM120 kernels are live

This entry used to say the pinned `deep_gemm-2.5.0+891d57b` wheel could not import: its `_C`
extension declares `NEEDED libcudart.so.13` and `libnvrtc.so.13` while the stack was CUDA 12, so
`uv sync --all-extras` produced a `deep_gemm` that raised
`ImportError: libcudart.so.13: cannot open shared object file`. vLLM noticed and silently fell back
to its vendored `vllm.third_party.deep_gemm`, which made the pin dead weight.

PR #3425 moved the whole stack to CUDA 13, and the conclusion inverts rather than going away.
Verified on this branch: `import deep_gemm` now succeeds from site-packages, version 2.5.0. So the
pinned wheel is what every FP8 kernel path actually runs on now, not the vendored copy, which
affects `InferenceConfig.use_deep_gemm`, the `examples/advanced/glm-5.2/` GLM-5-FP8 configs, and
`[trainer.model.quantization] type = "fp8"`.

That makes the warning this entry used to bury the live concern. The pinned build has no SM120
kernels for the block-scaled FP8 GEMMs, the UE8M0 scale-layout transform, the hyper-connection
GEMM, or the paged MQA logits, all of which the vendored copy handles. It was previously
unreachable so it did not matter; now, on Blackwell consumer parts, the pin is the thing that gets
loaded. Irrelevant on H100/H200 (SM90), where these tests ran. Worth deciding deliberately before
anyone runs this on SM120: either build the wheel with SM120 kernels, or drop the pin and let vLLM
use its vendored copy.

## `convert_rope_params_to_dict` overrides are dead code

`LagunaConfig.convert_rope_params_to_dict` and `DeepseekV4Config.convert_rope_params_to_dict`
are both no-ops (`return kwargs`) in practice: `rope_theta`/`rope_scaling` are consumed as named
`__init__` params and never reach `super().__init__()`'s `**kwargs`, and `self.rope_parameters`
is unconditionally overwritten by each config's own normalization method right after
`super().__init__()` returns. Verified empirically for both (deleting the method changes
nothing). DeepSeek V4's copy was inherited from Laguna's precedent. Decision: keep both for now;
remove together in one follow-up commit, after also checking `from_pretrained`/`from_dict`
checkpoint-loading paths (not covered by the empirical check above).
