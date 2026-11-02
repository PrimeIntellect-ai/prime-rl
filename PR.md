This PR extends selective activation checkpointing's `model.ac.targets` with `<operator>@<module path glob>` entries, so a checkpoint target can name a custom op plus the module path it must appear under instead of only a module type. DeepSeek V4 then saves the activation-checkpointing fit subset (the mHC pre-pass, `o_a_proj`, `o_b_proj`, `q_a_proj`, `kv_proj`, the shared expert, and the fp8 blockwise GEMMs at those sites) instead of checkpointing everything, the same mechanism as PR #4339.

The target syntax is `<op or namespace>@<module path glob relative to the decoder block>`, for example `prime_rl::fp8_blockwise_mm@self_attn.o_b_proj`. Under `compile(checkpoint_wrapper(block))` the policy reads the fx node's `nn_module_stack`, whose paths traverse the checkpoint wrapper (`L['self'].__dict__['_modules']['_checkpoint_wrapped_module'].inner.b`), and in eager mode no such metadata exists, so eager raises rather than silently checkpointing the wrong thing. A partitioner-hook mechanism run verified what each arm saves: the fit config saves exactly the listed operator outputs, 60.8 KB per token per layer, which is 1.86 GiB per layer per 32768-token micro batch, identical across ranks, with no sparse-attention or `q_b_proj` outputs saved.

The numbers come from two campaigns. The fit-subset A/B (this branch's own campaign, full AC versus fit AC, 4 and 6 layers, the same proxy otherwise, medians of `time/step` over steps 4 to 20, a negative delta is faster, full numbers in `~/tmp/dsv4-sac-targets-report.md` and `benchmarks/scripts/dsv4-perf-proxy/sac_targets_results.md`) isolates the policy change: fit saves -3.2% at 4 layers (10.64 to 10.29 s per step, 0.02 s same-code floor) and -2.9% at 6 layers (14.92 to 14.48 s, 0.3% floor), and the 43-layer projection puts a step at 90.8 s versus full AC's 93.6 s (-3.0%), with throughput 2630 to 2716 tokens/s per GPU. The stack campaign (recomputed from the raw metrics in `~/tmp/dsv4-stack-attribution.md`, a DeepSeek V4 Flash checkpoint at 4 and 6 layers, 262144-token rows, cp 8, ep 8, 20 steps of 8 micro batches, `fb` is `time/forward_backward`, medians over steps 4 to 20, the post-compile window, with the all-20-steps median as the second view) measures the arm that also carries the earlier optimization levels of this series, with the halo flag off and the tf32 backward env set:

| run | layers | fb med s (steps 4-20) | fb med s (all steps) | MFU % | tok/s per GPU | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| main-L4-D | 4 | 10.340 | 10.361 | 29.68 | 23804 | 87.71 |
| l6-L4-D (this change) | 4 | 9.650 | 9.683 | 31.74 | 25486 | 91.84 |
| main-L6-D | 6 | 14.577 | 14.623 | 30.64 | 16880 | 107.74 |
| l6-L6-D (this change) | 6 | 13.561 | 13.607 | 32.93 | 18144 | 115.64 |

The stack arm's cumulative delta is -6.67% at 4 layers and -6.97% at 6 layers, about 47x the campaign's 0.15% same-code repeat floor, and the chain marginal against the previous arm is -3.26% and -3.32%, consistent with the fit-subset campaign's -3% class. The 43-layer projection puts a step at 85.92 s versus main's 93.38 s, about -8%. Compile bar: identical Dynamo counts (160 at 4 layers, 176 at 6, all step 1), zero graph breaks, and +1 one-time DeepGEMM variant shared with the arm below.

Numerics: the AC policy touches no math, so no numerics A/B was needed. Memory is the trade: the saved fit-target activations raise the peak by +4.49 GiB at 6 layers in the stack frame (115.64 versus 111.15 GiB) and +1.19 GiB at 4 layers, where the peak sits in a layer's backward and the smaller recompute working set partly pays for the extra saved activations. In the fit-versus-full frame the peak moves from 107.6 to 109.6 GiB at 6 layers and falls from 87.5 to 85.3 GiB at 4 layers: under full AC the recompute materializes the MoE buffers alongside the recomputed attention internals, while the saved outputs cut that recompute, so the 4-layer peak drops even though more is saved.

Scale caveat, measured: at 43 layers the saved fit-target activations dominate, about +69 to +72 GiB over full AC in production, about 207 to 210 GiB against about 268 GiB usable on B300, which fits. The qb variant (saving `q_b_proj` instead, 3.79 GiB per layer) would need about +150 GiB more and does not fit B300, which is why the fit subset is the shipped choice.

### Verification

- Trainer unit tests for the `<operator>@<module path>` target parsing and matching (9 selected, passing on H200).
- Partitioner-hook mechanism runs (full, fit, qb) verifying the saved-operator sets byte for byte across ranks.
- Fit-subset A/B at 4 and 6 layers with same-code repeats for the noise floor, plus memory-snapshot runs locating both arms' peaks inside the first backpropagated layer.

### Commits

2d7da5d58 feat(trainer): let selective AC targets name a module path

## B300 rerun (2026-11-02)

The same proxy rerun on one 8x B300 SXM6 node (SM103), 20 steps, balanced routing, 262144-token rows, cp 8, ep 8, full activation checkpointing, medians of `time/forward_backward` over steps 4 to 20, paired against a `main` arm from the same worktree series on the same node. The B300 campaign runs MXFP8 (`quantization.type=mxfp8`, `recipe=mxfp8_rceil_wgrad_with_hp`, `moe.compute.type=mxfp8`) with a DeepSeek V4 Flash checkpoint, the production-representative low precision path on this hardware, so cross-campaign comparability is the per-level delta, not the absolute step time (the H200 campaign ran FP8 blockwise and MFU here divides FP8 work by the dense BF16 peak 2.25e15 FLOP/s). The arm ran with `PRIME_RL_DSV4_MHC_BWD_PRECISION=tf32` and selective AC whose FP8 target names were rewritten for MXFP8 (`aten::mm@` entries for the four dense GEMM sites, `prime_rl::dsv4_mhc_pre` and `aten::bmm@self_attn.o_a_proj` unchanged). Noise floors from same-protocol repeats: main 0.04 to 0.05 percent, tip 0.12 percent.

| run | layers | fb med s (steps 4-20) | fb med s (all steps) | MFU % | tok/s per GPU | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| b300-main-L4-mxfp8 | 4 | 5.896 | 5.902 | 22.90 | 41782 | 88.26 |
| b300-main-L6-mxfp8 | 6 | 8.227 | 8.252 | 23.89 | 29937 | 109.17 |
| b300-l6-L4-mxfp8 | 4 | 5.543 | 5.554 | 24.34 | 44418 | 93.47 |
| b300-l6-L6-mxfp8 | 6 | 7.666 | 7.690 | 25.63 | 32119 | 117.44 |
The arm delta is -5.99 percent at 4 layers and -6.81 percent at 6 layers against the B300 main (H200 measured -6.67 and -6.97 percent). The memory signature reproduces: peak rises 5.2 GiB at 4 layers and 8.3 GiB at 6 layers over the full-AC main (H200 stack frame: +4.1 and +7.9 GiB).

Compile bar: zero Dynamo recompiles and zero graph breaks in every B300 run, all compilation confined to step 1, Triton cubin count at or below the main arm (the stack removes kernels), DeepGEMM variants 3 to 4, the same one-time hc_head DeepGEMM variant the arm below carries, TileLang cache files unchanged at 5. Mechanism note for MXFP8, settled by a three-arm differential accounting run (4 layers, 2 steps, arms differing only in the GEMM-site target op name): the `aten::mm@` targets used here DO match at CheckpointPolicy time and save exactly the fit subset, adding 851 MiB per graph at the four GEMM sites (o_b_proj 252, shared expert 504, q_a_proj 63, kv_proj 32 MiB) and taking per-graph saved op outputs from about 0.97 GiB (min-cut alone, saved by both the no-match control and an `aten::_scaled_mm@` variant) to 1.84 to 1.90 GiB. The `aten::_scaled_mm@` variant matches nothing at policy time even though the partitioner records the saved tensors at `aten::_scaled_mm` nodes, so the trace-time op name is the one the targets must use. The min-cut alone saves the o_a_proj bmm and the mHC pre-pass outputs but not the GEMM fit subset.
