# DeepSeek V4 stack on B200: correctness and step time

On one 8x B200 node, the five-PR stack (#4354 to #4358) passes every correctness check that touches it. On a 6-layer DeepSeek V4 Flash proxy it cuts `time/forward_backward` by about 4.5% with default flags, about 6% with the halo exchange on, and about 9.5% with halo plus the selective AC module targets below, which cost about 7 GiB more peak memory.

All numbers come from 8x B200 (SM100) nodes, torch 2.14.0+cu130, driver 580.196.07. The stack was measured after reverting f2b06271e (the SM90 admission for the cuDNN compressor kernels).

## Setup

- Proxy: `proxy.toml`, run by `run_arm.sh` (both from `benchmarks/scripts/dsv4-perf-proxy/` on `exp/dsv4-stack-run-main`, with the model and precision switched to MXFP8 as in `configs/advanced/deepseek-v4-flash/swe_mxfp8.toml`). 262144-token rows, cp 8, ep 8, full activation checkpointing, `compile.fullgraph = true`, `matmul_precision = "highest"`, Muon, 8 micro batches per step, `data.fake.doc_len_range = [229376, 262144]`, 20 steps.
- Every run adds `--model.debug.force-balanced-routing --model.debug.num-layers 6 --no-monitors.prime`, and loads real weights from a local DeepSeek V4 Flash checkpoint (`--model.name $DSV4_CKPT --model.conversion-dir $DSV4_CONV`).
- Arms are frozen detached worktrees at each PR branch tip, plus ee35cdf13 (`doc_len_range` for fake data) and the proxy harness, committed locally. Base is main at b7770c930, the stack's base.
- Each node runs main and its arm interleaved (main, arm, main, arm), so every delta is measured on one node. `ab_node.sh` drives one node, `launch_ab.sh` lists the sequences, and `commands.sh` records every command in order (paths rewritten to environment variables).
- Fresh per-run Inductor, Triton, TileLang and DeepGEMM caches. quack's CuTe DSL cache was shared across runs on a node, which affects step 1 only.
- Every node passed `profiling/node_check.py` (GEMM, HBM, NVLink and collective checks against the fleet median) before use.
- Metric: median `time/forward_backward` over steps 4 to 20, from `monitors/file/metrics.jsonl`, via `profiling/run_metrics.py`. No compile event after step 1 on any rank in any run (`profiling/compile_events.py`). Busy SM clock medians match within each node, about 1950 to 1965 MHz with no power-brake throttling (`profiling/gpu_clocks.py`).

## Step time

Each arm includes every PR below it in the stack. Values are the mean of two runs, each run's median over steps 4 to 20. Lower is faster.

| node | arm | main (s) | arm (s) | vs main | main r1 vs r2 | peak GiB |
|---|---|---|---|---|---|---|
| n4354 | #4354 | 9.00 | 8.92 | -0.8% | 0.0% | 109.0 |
| n4355 | #4355, halo | 9.02 | 8.78 | -2.6% | 0.5% | 109.1 |
| n4356 | #4356, halo | 9.00 | 8.61 | -4.3% | 0.3% | 112.2 |
| n4357 | #4357, halo | 9.01 | 8.46 | -6.1% | 0.5% | 112.2 |
| n4358 | tip, default flags | 9.03 | 8.59 | -4.9% | 0.1% | 114.0 |
| n4358 | tip, halo + SAC targets | 9.03 | 8.17 | -9.5% | 0.1% | 118.9 |
| naux | tip, halo | 9.00 | 8.44 | -6.2% | 0.4% | 112.2 |
| naux | tip, default flags | 9.00 | 8.61 | -4.4% | 0.4% | 114.0 |

Main peaks at 110.8 GiB. `perf/peak_memory` is the maximum over ranks of reserved plus non-torch memory, not allocated bytes.

- Per PR, from the differences between nodes: about -0.8% for #4354, -1.8% for #4355 with halo on, -1.7% for #4356, -1.8% for #4357, and no measurable change for #4358 without targets. These cross-node increments carry about 0.5% of noise.
- Halo on against halo off, at the tip on one node: about -2.0% in time and -1.8 GiB in peak memory.
- SAC targets on top of halo: about -3.3% more, for about +6.7 GiB.
- The proxy rows hold about one document each, the case the cuDNN compressor targets.

Per-run tables: `results/`.

## SAC targets for MXFP8

Under MXFP8 the linears show the policy `aten::mm` with MXTensor operands, so the H200 `prime_rl::fp8_blockwise_mm` entries become `aten::mm` (`sac_targets.json`):

```
["prime_rl::dsv4_mhc_pre", "aten::bmm@self_attn.o_a_proj", "aten::mm@self_attn.o_b_proj",
 "aten::mm@self_attn.kv_proj", "aten::mm@mlp.shared_expert", "aten::mm@self_attn.q_a_proj"]
```

`sac_probe/sitecustomize.py` logged the operators the policy sees on one compiled step. `sac_probe/count_scaled_mm.sh` confirmed that the compiled backward graphs then recompute 1 instead of 7 forward `_scaled_mm` GEMMs per layer (only `q_b_proj`, which is not targeted).

## Correctness on B200, at the stack tip

| check | result |
|---|---|
| cuDNN compressor pooling vs fp64 (csa, indexer, hca) | 3 passed, cuDNN kernels confirmed launched |
| compressor packing, model backward, halo shards | 4 passed |
| halo compressor vs one GPU, torchrun 2 ranks | 2 passed on each rank |
| CP parity, `../dsv4-stack-checks/cp_halo_parity.py`, cp 2, 4, 8 | 972 of 972 cases at each cp |
| RoPE, mHC, post-BDA, sinkhorn kernels | 53 passed |
| fp8 indexer | 8 passed, 1 failed identically on main |
| checkpointing, AC target configs | 23 and 5 passed |
| recompiles at cp 1 and cp 2 on short packed documents | same as main: none after step 2 |

- The CP parity cases needed `torch._inductor.config.emulate_precision_casts = True`. Without it, one cp 4 case of the torch compressor backend under compile failed: Inductor keeps the pooling's bf16 softmax weights unrounded inside fused kernels, and the halo and gather graphs fuse differently.
- The fp8 indexer failure (`test_selection_matches_float64_on_quantized_inputs`) is a test tolerance on SM100: DeepGEMM's `fp8_fp4_mqa_logits` returns bf16 logits there, which the test's fp32 threshold does not allow for.
