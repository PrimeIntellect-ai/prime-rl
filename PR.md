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
