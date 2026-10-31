# B300 vs H200 MoE throughput repro: results

Question (PLAN.md, execution scope): does the roofline prediction that B300's MFU is about 0.83x H200's for DSv4 with the same software hold, and do the model-specific roofline predictions hold for GLM-4.5-Air and GLM-5.3? Answer: no. Measured B300/H200 MFU ratios land 0.73 to 0.78 for the two DSA-sparse-attention models (DSv4, GLM-5.3) against roofline predictions of 0.89 to 0.95, while dense-attention GLM-4.5-Air matches its prediction almost exactly (0.98 to 1.00 measured vs 0.99 predicted). The sparse-attention stack runs memory-bound on B300: DSv4's wall speedup is 1.66 to 1.67x, the HBM bandwidth ratio, and GLM-5.3's is a flat 1.78x, both against a 2.27x compute peak ratio.

## Setup

- B300: one 8x B300 SXM6 node (SM103), SLURM exclusive, branch `exp/dsv4-b300-repro` at `8123bea94`, `uv run --no-sync`, FP8 blockwise and MXFP8 arms as in PLAN.md.
- H200: one 8x H200 node, same branch code state, FP8 blockwise (proxy.toml default), references from the earlier H200 pass plus fresh GLM-5.3 long runs (provenance below).
- Both: `run_arm.sh` harness, balanced routing (`debug.force_balanced_routing`), fake data, 20 steps, medians over steps 4 to 20 from `monitors/file/metrics.jsonl`. `perf/mfu` divides FP8 work by the device's dense BF16 peak (H200 9.89e14, B300 2.25e15 FLOP/s), so ratios are comparable across definitions; `mfu_6dn` ratios track `perf/mfu` ratios within about one point everywhere.
- Settings: short is seq 32768, cp 1, 64 documents of 4096 to 8192 tokens; long is seq 262144, cp 8, 8 documents of 229376 to 262144 tokens; both are 32768 tokens per GPU per step.
- Roofline predictions: per-op FLOPs and compulsory HBM bytes per micro batch, per-device peaks (B300 from the HGX B300 table, H200 dense BF16 9.89e14, FP8 1.979e15, HBM 4.8 TB/s from nvidia.com/en-us/data-center/h200/), computed with the profiling skill's `roofline.py`; H200 copies of the ops files differ from the B300 files only in the device section. Inputs and derived reports live in `~/tmp/profiling/dsv4-perf-proxy/` on the B300 cluster; commands are in `commands.sh` there.

## Long context, 262144 seq, FP8 blockwise (same software both sides)

| model | layers | s/step H200/B300 | speedup | raw mfu H200/B300 | measured ratio | predicted ratio |
|---|---|---|---|---|---|---|
| DSv4 Flash | 4 | 10.3 / 6.2 | 1.67x | 29.8 / 21.9% | 0.73 | 0.89 |
| DSv4 Flash | 6 | 14.5 / 8.7 | 1.66x | 30.8 / 22.5% | 0.73 | 0.89 |
| GLM-4.5-Air | 6 | 68.2 / 30.7 | 2.22x | 84.2 / 82.7% | 0.98 | 0.99 |
| GLM-4.5-Air | 12 | 135 / 59.7 | 2.26x | 84.7 / 84.5% | 1.00 | 0.99 |
| GLM-5.3 | 4 | 13.1 / 7.4 | 1.77x | 44.9 / 35.1% | 0.78 | 0.93 |
| GLM-5.3 | 6 | 18.6 / 10.5 | 1.78x | 40.8 / 32.0% | 0.78 | 0.93 |
| GLM-5.3 | 8 | 24.5 / 13.7 | 1.79x | 40.1 / 31.4% | 0.78 | 0.92 |

GLM-5.3 long H200 provenance: fresh runs on 8x H200 (node gpu-056, branch code identical, FP8 blockwise, balanced routing, 20/20 steps, recompiles confined to step 1); the fresh 4L and 8L arms match two equivalent runs from the same day exactly (13.10 s and 24.51 s), so the numbers are reproducible. Peak memory matches B300 to within 0.2 GiB at every depth (71.8/102.2/130.9 vs 71.7/102.2/131.1 GiB), confirming shape-for-shape comparison.

## Short documents, 32768 seq, FP8 blockwise

| model | layers | s/step H200/B300 | speedup | raw mfu H200/B300 | measured ratio | predicted ratio |
|---|---|---|---|---|---|---|
| DSv4 Flash | 4 | 9.6 / 5.7 | 1.69x | 26.3 / 19.7% | 0.75 | 0.89 |
| DSv4 Flash | 6 | 12.9 / 7.6 | 1.70x | 26.8 / 20.1% | 0.75 | 0.88 |
| GLM-4.5-Air | 6 | 11.4 / 6.2 | 1.84x | 39.3 / 31.8% | 0.81 | 0.93 |
| GLM-4.5-Air | 12 | 19.8 / 10.6 | 1.88x | 40.9 / 33.8% | 0.83 | 0.92 |
| GLM-5.3 | 4 | 12.0 / 6.9 | 1.75x | 35.4 / 27.2% | 0.77 | 0.95 |
| GLM-5.3 | 8 | 12.9 / 12.9* | | no H200 ref | | 0.93 |

The GLM-5.3 short 8L B300 run measured 12.9 s/step, 18.3k tok/s/GPU, 25.7% mfu; the H200 side never ran this arm, so no measured ratio. The predicted ratio at this config is 0.934 and the short-setting pattern from 4L suggests the measured ratio would land near 0.77 to 0.80.

## Full-depth extrapolation (FP8, extrapolate.py, B300 low/high pairs, medians over steps 4 to 20)

| model | setting | B300 full-depth s/step | tok/s/GPU B300/H200 | mfu B300/H200 | ratio |
|---|---|---|---|---|---|
| DSv4 Flash (43L) | short | 43.2 | 5.5k / 3.18k | 21.2 / 27.9% | 0.76 |
| DSv4 Flash (43L) | long | 56.3 | 4.4k / 2.67k | 23.8 / 32.9% | 0.72 |
| GLM-4.5-Air (46L) | short | 35.2 | 6.7k / 3.50k | 35.7 / 42.4% | 0.84 |
| GLM-4.5-Air (46L) | long | 223.8 | 1.1k / 476 | 86.0 / 85.2% | 1.01 |
| GLM-5.3 (78L) | short | 119.4 | 2.0k / n/a | 24.2 / 30.7% | 0.79 |
| GLM-5.3 (78L) | long | 125.0 | 2.0k / 1.10k | 27.7 / 35.1% | 0.79 |

The GLM-4.5-Air long extrapolated ratio of 1.01 slightly exceeds 1; both full-depth extrapolations carry intercept noise at these shallow truncation depths (per-layer medians: DSv4 short 0.96 s, long 1.29 s; Air short 0.72 s, long 4.8 s; GLM-5.3 short 1.52 s, long 1.59 s).

## MXFP8 vs FP8 blockwise on B300 (same configs, s/step medians)

| config | fp8 | mxfp8 | mxfp8 gain |
|---|---|---|---|
| DSv4 short 4L / 6L | 5.68 / 7.61 | 5.66 / 7.52 | 0.5 / 1.2% |
| DSv4 long 4L / 6L | 6.16 / 8.72 | 5.87 / 8.27 | 4.9 / 5.5% |
| Air short 6L / 12L | 6.20 / 10.55 | 6.06 / 10.25 | 2.4 / 2.9% |
| Air long 6L / 12L | 30.7 / 59.7 | 30.9 / 59.7 | 0.0 / 0.0% |
| GLM-5.3 short 4L / 8L | 6.86 / 12.93 | 6.79 / 12.71 | 1.0 / 1.7% |
| GLM-5.3 long 4L / 8L | 7.38 / 13.73 | 7.27 / 13.50 | 1.4 / 1.7% |

MXFP8 is never slower than FP8 blockwise on B300 and helps most on DSv4 long context (about 5%). The precision recipe is not the source of the B300/H200 gap: both precisions show the same ratios.

## Interpretation

The roofline predictions already encode B300's worse compute-to-bandwidth ratio (FP8 compute 2.27x, HBM bandwidth 1.67x), so the predicted ratio drop from 0.99 (Air, long) to 0.89 to 0.93 (DSv4, GLM-5.3) is the expected cost of carrying more memory-bound work. The measured shortfall below prediction is a separate effect: DSv4 achieves 82% of its predicted ratio, GLM-5.3 84%, while Air achieves 99%. Since the roofline assumes compulsory HBM bytes at full bandwidth, the residual is either unmodeled traffic (activation-recompute, quantize and dequantize passes, unfused elementwise, index rereads) or kernels achieving a lower fraction of their per-op roofline on SM103 than their SM90 counterparts, or both. Two further observations: Air's short setting also undershoots (0.81 to 0.83 vs 0.92 to 0.93 predicted, about 89% of prediction), so bandwidth-bound MoE work on B300 also runs below its roofline, but the DSA models pay roughly twice that penalty; and DSv4's long-context wall speedup is exactly the bandwidth ratio (1.66x) while GLM-5.3 sits slightly above (1.78x), the one place DSv4 looks specifically worse beyond the shared DSA-stack effect. Per-kernel traces were collected for all three models (short setting, 4 steps each, `~/prl_output_dir/dsv4-perf-proxy/traces/` on the B300 cluster); the kernel-by-kernel attribution against the H200 traces is pending analysis.

## Provenance and caveats

- All B300 runs: 20/20 steps, medians over steps 4 to 20, verified by step count in each run's metrics, run dirs under `~/prl_output_dir/dsv4-perf-proxy/` on the B300 cluster (b300-*).
- Recompiles: all Dynamo recompiles in step 1 in every run. Five FP8 runs show a single late DeepGEMM JIT compile at step 3 (dsv4-long-4L, dsv4-long-6L, glm53-long-4L/6L/8L), so their steady window from step 4 is correct; no MXFP8 run showed a late compile.
- Measurements use truncated layer counts, fake data, and forced balanced routing on one node per architecture; absolute numbers are not production predictions, ratios are the point.
- B300 node primeintellect-tur-r2n1 hit SLURM NODE_FAIL once mid-pass (one run lost and rerun elsewhere); the replacement allocation was cancelled by uid 0 (cluster-side) before the extra arms ran.
- Not run: the two padding-check arms and the ten deeper-layer arms (12L for DSv4, 12L/16L for GLM-5.3, both precisions) from the plan's optional extras; the scaling ladder above uses 4L/6L, 6L/12L, and 4L/6L/8L respectively instead.
- H200 GLM-5.3 long runs were measured on demand for this repro after the gap was found; the other H200 references predate it.
