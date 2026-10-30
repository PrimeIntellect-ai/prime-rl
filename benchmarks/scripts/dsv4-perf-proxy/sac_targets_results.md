# WS5: per-call-site selective activation checkpointing for DeepSeek V4

## Summary

- `model.ac.targets` now accepts `<operator or namespace>@<module path glob>` entries, for example `prime_rl::fp8_blockwise_mm@self_attn.o_b_proj`, resolved from torch.compile's per-node module metadata. It raises in eager mode or on a graph break inside the block, and on patterns that match no module.
- Mechanism verified on the proxy by logging what the min-cut partitioner saves: the fitting subset (b) saves exactly mHC pre (attention and FFN), o_a_proj, o_b_proj, q_a_proj, kv_proj and the shared expert, 1.86 GiB per layer per micro batch at 32768 tokens per rank. Variant (c) swaps q_a_proj for q_b_proj, 3.79 GiB per layer. No new recompiles or graph breaks.
- H200 proxy, paired on one node, steps 4 to 20 (also the post-last-compile window): fit is 3.1% faster than full AC at 4 layers (10.30 vs 10.63 s, noise floor 0.2%) and 3.1% at 6 layers (14.48 vs 14.94 s, noise floor 0.3%). The 43-layer projection is 90.8 vs 93.7 s per step (-3.1%). qb is about 0.6% faster than fit at 4 and 6 layers.
- Memory: at 4 layers the peak sits in a layer's backward, where fit's smaller recompute working set (-9.1 GiB) outweighs its extra saves (+6.9 GiB). At 43 layers the saves dominate: fit projects to about +69 to +72 GiB over full AC (207 to 210 GiB in production against about 268 GiB usable on B300), qb to about +150 GiB, which does not fit.

## What changed

Branch `exp/dsv4-sac-targets` (worktree `/home/garrett/github/PrimeIntellect-ai/prime-rl-exp-dsv4-sac-targets`), on top of `exp/dsv4-perf-proxy-main`:

| commit | subject |
|---|---|
| db72bd0cb | feat(trainer): let selective AC targets name a module path |
| f753ccb61 | chore(benchmarks): record the tensors saved for backward per compiled graph in the DSv4 proxy |
| f7d930952 | chore(benchmarks): add the per-call-site selective AC arms driver for the DSv4 proxy |
| fa86eed5c | chore(benchmarks): count only tensor bytes in the saved-activations hook |
| e5761f2b1 | style(benchmarks): ruff-format the saved-activations summarizer |

All timing and mechanism runs used fa86eed5c (e5761f2b1 only reformats an analysis script). This report is also committed as `benchmarks/scripts/dsv4-perf-proxy/sac_targets_results.md`. Every command that produced a number here is in `~/tmp/profiling/dsv4-sac-targets/commands.sh`, run directories are under `/home/garrett/prl_output_dir/dsv4-perf-proxy/` (`M4-ws5b-*`, `A4-ws5-*`, `A6-ws5-*`, `M4-ws5-mem-*`). `M4-ws5-full` crashed in an early version of the hook and `M4-ws5-fit` was stopped, both are unused.

### Target syntax

`model.ac.targets` entries are either plain (unchanged meaning: an operator name such as `aten::mm` or a namespace such as `prime_rl_collectives`) or module-qualified: `<operator or namespace>@<module path pattern>`, for example `prime_rl::fp8_blockwise_mm@self_attn.o_b_proj`.

- The pattern is an `fnmatch` glob over module paths relative to the checkpointed decoder block. It matches an operator that runs inside a matching module or any of its descendants, so `prime_rl::fp8_blockwise_mm@mlp.shared_expert` keeps the shared expert's gate, up and down GEMMs.
- `@` was chosen over the plan's `module:operator` example because operator names already contain `::`, and `self_attn.o_b_proj:prime_rl::fp8_blockwise_mm` both reads and parses ambiguously. `@` appears in neither operator names nor module paths, and pydantic-config only treats arguments that start with `@` as config files.
- On the CLI: `--model.ac.targets '["prime_rl::dsv4_mhc_pre", "prime_rl::fp8_blockwise_mm@self_attn.o_b_proj"]'`. In TOML: a list of strings.

### How it works and where it raises

- Under `torch.compile`, Dynamo records each traced operation's `nn_module_stack`. AOTAutograd retraces the checkpoint body through an FX `Interpreter` with node metadata preserved, so the SAC policy reads the current node's module stack from `torch.fx.traceback.get_current_meta()` while it tags nodes for the min-cut partitioner. The policy strips the Dynamo source prefix up to `_checkpoint_wrapped_module.` to get the block-relative path. Verified first on a tiny `compile(checkpoint_wrapper(block))`, then on the proxy.
- Eager mode has no module metadata when the policy runs (`has_preserved_node_meta()` is False), so module-qualified targets raise there instead of being ignored:
  - config time: `model.ac.targets` with `@` and `model.compile = None` is rejected,
  - run time: the policy raises a `RuntimeError` if it runs outside compile tracing (covers a graph break inside the block, which makes the checkpoint fall back to eager),
  - `apply_ac` raises if a module pattern matches no submodule of any checkpointed block (typo guard),
  - the config rejects malformed entries (`@x`, `op@`, `op@a@b`).
- Mandatory saves (DeepEP, `aten::topk`, the indexer ops, CUDA-to-CPU copies, MoE routing statistics) still run first.
- PR #3954 (DeepEP, open) also edits `activation_checkpointing.py`. Whichever lands second resolves a small conflict. Not acted on.

### Tests

- `tests/unit/train/models/test_checkpointing.py::test_module_qualified_targets_save_only_matching_call_sites` (4 cases): compiles a two-projection block with `aot_autograd` and the min-cut partitioner and asserts which `aten::mm` outputs are saved for backward (both for a plain target, only the named site for a qualified one, glob included).
- `test_module_qualified_targets_raise_without_compile`: eager raises.
- `tests/unit/test_configs.py::test_invalid_ac_targets_fail` (4 cases): malformed entries and the missing-compile case.
- Ran only these plus the existing tests in `test_checkpointing.py`, selected by `-k`, on the compute node: 13 passed.

## Mechanism evidence: what each arm saves

Measured with `benchmarks/scripts/dsv4-perf-proxy/saved_activations/sitecustomize.py`, which wraps Inductor's min-cut partitioner (`custom_partitioner_fn`) and logs every value the forward graph hands to the backward, with its op, checkpoint tag, module and bytes. Runs `M4-ws5b-{full,fit,qb}` (4 layers, 2 steps, commit fa86eed5c), summarized by `saved_activations.py`. Tables: `~/tmp/profiling/dsv4-sac-targets/derived/M4-ws5b-saved-rank{0,3,7}.md` (identical across the three ranks checked).

Arms (all `mode = "selective"` except full, all with balanced routing):
- (a) full: full AC.
- (b) fit: `prime_rl::dsv4_mhc_pre`, `aten::bmm@self_attn.o_a_proj`, and `prime_rl::fp8_blockwise_mm@` each of `self_attn.o_b_proj`, `self_attn.q_a_proj`, `self_attn.kv_proj`, `mlp.shared_expert`.
- (c) qb: fit with `self_attn.q_b_proj` in place of `self_attn.q_a_proj`. Reading of the plan: (c) compares a q_b_proj save (does not fit at 43 layers) against the q_a_proj-only save in (b), with everything else equal.

Each compiled layer graph (one per layer structure: sliding, CSA, HCA, each recompiled once in step 1 on a guard on the FSDP input view's base stride, in every arm) saved exactly the targeted sites and nothing else beyond full AC. Bytes per token per layer follow from the shapes, so the table scales them to 32768 tokens per rank (production's 262144-token micro batch over cp 8):

```
saved for backward per layer per micro batch, measured, 32768 tokens per rank
site (module, op)                     full AC    (b) fit     (c) qb
attn_hc   dsv4_mhc_pre outputs        -          0.26 GiB    0.26 GiB
self_attn.q_a_proj  fp8 GEMM          -          0.06        -          <-- (c) swaps
self_attn.q_b_proj  fp8 GEMM          -          -           2.00       <-- these two
self_attn.kv_proj   fp8 GEMM          -          0.03        0.03
self_attn sparse attention            -          -           -
self_attn.o_a_proj  bmm (bf16)        -          0.50        0.50
self_attn.o_b_proj  fp8 GEMM          -          0.25        0.25
ffn_hc    dsv4_mhc_pre outputs        -          0.26        0.26
mlp.shared_expert   3 fp8 GEMMs       -          0.50        0.50
total                                 0          1.86 GiB    3.79 GiB
```

The fit total is 60.8 KB per token per layer (1.83 GiB at the 32256-token trace hint of the first micro batch). Full AC saves only the layer input and the mandatory saves (the CSA indexer output, 2 KB per token), which all arms share. The indexer's own `q_b_proj` (`self_attn.compressor.indexer.q_b_proj`) is not matched by `self_attn.q_b_proj`, as intended. Recompiles were identical in all three arms (20 per rank: 5 layer forwards plus Muon), with no graph breaks.

## Paired A/B, 4 layers

One exclusive H200 node (job 7462, prime-nebius-puku-h200-gpu-041), all runs at commit fa86eed5c (code identical across arms, only `model.ac` differs), fresh compile caches per run, 20 steps, balanced routing, order full, fit, qb, fit, full. Proxy: DeepSeek V4 Flash checkpoint truncated to 4 layers (sliding, sliding, CSA, HCA), cp 8, ep 8, FP8 blockwise via DeepGEMM, Muon, fullgraph, 8 micro batches per step of one 229k to 262k-token fake document each, so about 247k tokens per GPU per step (about 30.9k tokens per GPU per micro batch).

Step time is `time/step`, median over steps 4 to 20, lower is better. "vs full" is the arm's median divided by the mean of the two full-AC medians, minus 1. The post-last-compile window is also steps 4 to 20 (see the step 3 note below), so the two requested views coincide. Throughput is `perf/throughput_per_gpu` (higher is better), `perf/mfu` uses the dense BF16 peak, peak memory is `perf/peak_memory` (max over ranks of reserved memory plus memory outside the caching allocator, lower is better).

| arm (run) | step ms | min to max ms | vs full | tok/s/GPU | perf/mfu % | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| (a) full (r1) | 10636 | 10377 to 10861 | | 23577 | 29.40 | 87.5 |
| (b) fit (r1) | 10295 | 10044 to 10529 | -3.1% | 24270 | 30.28 | 85.3 |
| (c) qb (r1) | 10242 | 10006 to 10465 | -3.6% | 24461 | 30.51 | 93.2 |
| (b) fit (r2) | 10309 | 10067 to 10538 | -3.0% | 24280 | 30.28 | 85.3 |
| (a) full (r2) | 10619 | 10377 to 10843 | | 23576 | 29.40 | 87.5 |

- Noise floor from same-code repeats: 17 ms (full) and 14 ms (fit), both 0.2%.
- fit saves 326 ms per step at 4 layers, which is 82 ms per layer per step, or about 10 ms per layer per micro batch.
- qb is 60 ms per step (0.6%) faster than fit at 4 layers, for 2 GiB more saved per layer per micro batch.
- Steps 2 to 20 (after the last Dynamo and DeepGEMM compile, which is step 1 in every run) give the same medians within 20 ms. That window still contains step 3 (about 18 s in every arm).

Step 3 note: step 3 is a compile that `compile_events.py` does not see. Every run writes about 15 non-Inductor Triton kernels to its Triton cache during step 3 only: the mHC kernels (`_triton_sinkhorn_fwd/bwd_kernel`, `_triton_hpb_*`), `dsv4_mhc_pre`'s `_fwd/_bwd/_bwd_weight_kernel` and `fp8_utils._per_token_fp8_kernel`. The cached TTIR shows why: the step 3 variant of `_triton_sinkhorn_fwd_kernel` has `N_batch: i32 {tt.divisibility = 16}`, the step 1 variants do not. Step 3 holds the first micro batch whose per-rank token count is a multiple of 16, and Triton re-specializes each custom kernel that takes an unannotated integer length. Nothing compiles after step 3 (Triton, TileLang, DeepGEMM or Dynamo), so steps 4 to 20 are the post-last-compile window. This most likely does not happen in production DeepSeek V4 MXFP8 runs. Assume production does not set `pad_to_multiple_of` (Garrett is not sure whether it does). Even so, per Garrett, MXFP8 pads to 32 tokens, which with cp 8 makes every per-rank token count a multiple of 32 (the multiple of 256 in an earlier planning note comes from 8 x 32), so production sees the divisible-by-16 case from the first micro batch. It does happen in the proxy (blockwise FP8, no 32-token padding, random fake lengths) and in any unpadded setup, where it is a bounded one-time cost: Triton specializes an integer only on divisibility by 16 (and equality to 1), so each such kernel compiles at most one extra variant. `compile_events.py` could read Triton cache listings the way it reads DeepGEMM's.

## 6 layers and the 43-layer time projection

Same node and commit, `A6-ws5-r1-full`, `A6-ws5-r1-fit`, then (after the memory snapshot runs) `A6-ws5-r1-qb` and the repeat `A6-ws5-r2-full` (6 layers: sliding, sliding, CSA, HCA, CSA, HCA), steps 4 to 20 (again the post-last-compile window: last Dynamo and DeepGEMM compile in step 1, 15 Triton respecializations in step 3, nothing after).

| arm (run) | step ms | min to max ms | vs full | tok/s/GPU | perf/mfu % | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| (a) full (r1) | 14918 | 14609 to 15287 | | 16737 | 30.39 | 107.6 |
| (b) fit (r1) | 14485 | 14162 to 14844 | -3.1% | 17258 | 31.35 | 109.6 |
| (c) qb (r1) | 14400 | 14071 to 14722 | -3.6% | 17404 | 31.65 | 121.7 |
| (a) full (r2) | 14963 | 14611 to 15559 | | 16741 | 30.43 | 107.6 |

"vs full" uses the mean of the two full-AC medians. Noise floor from the full repeat: 45 ms (0.3%). Steps 2 to 20 give medians within 60 ms of these.

`extrapolate.py` pairs each arm's 4-layer and 6-layer runs step by step (steps 4 to 20): the per-layer cost is the step time difference over the 2 added layers, and the 43-layer step is the 6-layer step plus 37 more layers.

| arm | s per added layer per step | 43-layer step s | 43-layer tok/s/GPU | 43-layer perf/mfu % |
|---|---:|---:|---:|---:|
| (a) full | 2.135 (2.069 to 2.187) | 93.6 (91.0 to 96.0) | 2630 | 32.5 |
| (a) full, repeat (r2 pair) | 2.138 (2.082 to 2.548) | 93.8 (91.5 to 109.6) | 2635 | 32.5 |
| (b) fit | 2.069 (2.003 to 2.126) | 90.8 (88.1 to 93.3) | 2716 | 33.6 |
| (c) qb | 2.040 (1.993 to 2.096) | 89.7 (87.6 to 92.0) | 2747 | 33.9 |

Projected at 43 layers on H200: fit is 2.9 s (3.1%) faster per 8-micro-batch step than full AC (mean of the two full projections, 93.7 s), about 0.36 s per micro batch. qb projects to 4.0 s (4.3%), but it does not fit in memory at 43 layers (next section). qb's 6-layer peak is 12.1 GiB above fit's, which is 6 layers times q_b_proj's 2.0 GiB output, so its saves show up one for one. Assumptions:
- The 4 to 6 layer slope (one CSA and one HCA layer) stands for each of the 41 compressed layers of the full model. The two sliding layers are in the 4-layer intercept.
- Same tokens per GPU as production (about 31k per micro batch), balanced routing, FP8 blockwise GEMMs on H200, FSDP over 8 GPUs on NVLink. Production runs MXFP8 on B300 with FSDP across nodes. SAC changes none of the communication, but faster B300 GEMMs make each saved GEMM recompute worth less, while the mHC pre-pass part (memory-bound) should carry over. The B300 gain needs its own A/B.
- `perf/mfu` uses the H200 dense BF16 peak (989 TFLOP/s) and forced balanced routing, so it is not comparable with production's B300 MFU.

## Where the peak sits, and the 43-layer memory projection

Why fit saves about 7.4 GiB more at 4 layers yet peaks 2.2 GiB lower. Memory snapshots `M4-ws5-mem-{full,fit}` (4 layers, 3 steps, `--memory-profiler-path`), step 3, analyzed with `memsnap_peak.py`. All 8 ranks peak within 0.1 GiB of each other, rank 4 is detailed (`~/tmp/profiling/dsv4-sac-targets/derived/M4-ws5-mem-{full,fit}.step3.*`).

`perf/peak_memory` is reserved memory: the max over ranks of `torch.cuda.max_memory_reserved()` plus memory outside the caching allocator (5.5 GiB here). The snapshot peaks below are allocated memory. Reserved minus allocated at the peak is about 10.6 GiB in both arms (full 81.9 vs 71.2, fit 79.5 vs 68.9), so the 2.2 GiB drop shows in both.

| live at the peak, GiB (allocated) | (a) full | (b) fit | fit minus full |
|---|---:|---:|---:|
| parameters, Muon state, gradients (static) | 39.3 | 39.3 | 0 |
| saved for backward (layer inputs plus SAC saves) | 3.1 | 10.0 | +6.9 |
| backward/recompute temporaries | 15.6 | 14.0 | -1.5 |
| MoE dispatch/combine buffers | 7.6 | 1.5 | -6.1 |
| FSDP all-gather | 5.5 | 4.0 | -1.5 |
| other | 0.2 | 0.2 | 0 |
| total | 71.2 | 68.9 | -2.2 |

- Neither peak is in the lm_head/logprob chunk or in Muon. Both are in a layer's backward.
- Full AC peaks while recomputing layer 2 (the second layer backpropagated, a CSA layer: its sparse indices are 128 window plus 512 top-k wide). The recompute materializes the whole layer forward, so the MoE half's dispatch and grouped-GEMM buffers are live together with the recomputed attention internals (RoPE outputs 4.0 GiB, sparse attention output 2.0 GiB).
- Fit peaks in layer 3's sparse-attention backward (the first layer backpropagated, HCA: its indices are 128 plus a dynamic entry count wide). The saved o_b, o_a, mHC and shared-expert outputs cut the recompute, so the MoE buffers are already gone when attention's backward runs.
- So at 4 layers the stored activations do not set the difference. Fit's extra saves (+6.9 GiB: layers 0 to 2 at 1.86 GiB each, layer 3's input at 1.0 GiB, live because fit peaks one layer earlier in the backward, and 0.35 GiB of layer 3's saves not yet consumed) are outweighed by a 9.1 GiB smaller working set in the layer being backpropagated.

43-layer projection, built from saved bytes per layer plus where the peak sits, not from the 4-layer peak delta:
- At 43 layers the first layer backpropagated is layer 42, a CSA layer, and both arms peak in its backward (full AC: in its recompute, fit: in its attention backward), with the saves of the other 42 layers live. Both arms hold the same layer inputs (about 1 GiB per layer), so they cancel in the difference, as do the static state and fragmentation.
- Fit minus full at 43 layers = 42 x 1.86 GiB (other layers' SAC saves at 32768 tokens per rank) + 0.35 GiB (layer 42's saves not yet consumed) + the working-set change. The working-set change is -9.1 GiB measured (fit's HCA attention backward against full's CSA recompute). Fit's CSA attention backward was not the proxy's peak, so its working set is at most 2.9 GiB larger than the HCA one, which bounds the change at -6.3 GiB. That gives +69 to +72 GiB, or +79 GiB if the working set did not shrink at all.
- Check against the 6-layer runs: the same accounting predicts fit minus full = +1.5 GiB at 6 layers, and the runs measured +2.1 GiB (reserved), so the per-layer slope holds within 0.6 GiB.
- Production: 135 to 139 GiB `perf/peak_memory` with full AC on B300 (138 GiB on rank 0 in the MEGA_MOE benchmark), against about 268 GiB usable per GPU, from the B300 planning note. Projected fit peak 207 to 210 GiB (217 GiB without the working-set gain), which leaves about 50 to 60 GiB. Projected qb peak: 42 x 3.79 GiB = 159 GiB of saves, so about 288 to 297 GiB, which does not fit.
- Unknowns: production routing is not balanced, so the busiest EP rank's MoE buffers are larger. That only enlarges full AC's working set and therefore the fit gain, but the production peak rank may differ. Production micro batches are packed up to 262144 tokens, and the 1.86 GiB figure assumes all 32768 tokens per rank.

## Open questions

- B300 op names. Production quantizes dense linears with MXFP8 (torchao `_to_mxfp8_then_scaled_mm`, an autograd function around quantize kernels and a scaled GEMM), not `prime_rl::fp8_blockwise_mm`. The module paths carry over unchanged, but the operator half of each qualified target has to name the op that MXFP8 dispatches. The saved-activations hook answers that in one short B300 run. o_a_proj stays bf16 in production too (`extra_ignore_patterns` includes it), so `aten::bmm@self_attn.o_a_proj` carries over.
- Production compiles with `fullgraph = false`. Module-qualified targets need each block traced without a graph break, otherwise the policy raises (by design, rather than silently keeping nothing). The B300 MEGA_MOE benchmark ran with `fullgraph = true`, which suggests there is no break in the block, but it is worth one confirming run.
- The B300 time gain is unmeasured. Faster MXFP8 GEMMs make each skipped GEMM recompute worth less, while the mHC pre-pass save (memory-bound) should carry over.
- qb against fit: q_b_proj's save buys 60 ms per step at 4 layers (0.6%) for 2 GiB more per layer per micro batch, and it does not fit at 43 layers, so fit is the candidate.
- PR #3954 (DeepEP, open) also edits `activation_checkpointing.py`. Whichever lands second resolves a small conflict.

## Left out

- No comments, TODOs or docstrings were omitted. The two short comments added in `activation_checkpointing.py` follow that file's existing style (one on the target syntax next to the existing target comment, one showing the Dynamo source-name form the regex parses).
- `apply_ac`'s unmatched-pattern check has no unit test (repo policy is conservative). It was checked by hand on the node: `unmatched_module_patterns` returned exactly the two misspelled patterns out of five.
- `compile_events.py` (profiling skill) does not see non-Inductor Triton JIT compiles. I did not extend it. The step 3 finding came from the Triton cache mtimes and TTIR (commands in `~/tmp/profiling/dsv4-sac-targets/commands.sh`).
- The saved-activations hook lives in the branch (`benchmarks/scripts/dsv4-perf-proxy/saved_activations/`). It is general and could move into the profiling skill.
