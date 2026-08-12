# Native optimizer-offload handover

## What is implemented

PR #3234 overlaps native CPU AdamW with backward and transfers gradients and refreshed compute weights over PCIe in BF16 while retaining FP32 masters, moments, accumulation, and optimizer arithmetic.

The latest hardening replaces model-sized pinned slabs with bounded, size-classed D2H/H2D rings. Autograd hooks only validate the generation, record a preallocated CUDA event, enqueue the gradient, and clear `param.grad`. Pageable CPU slabs hold persistent state. Duplicate/stale generations fail explicitly, host waits have diagnostic timeouts, and validation and checkpoint-load paths drain the rings correctly. No Gloo process group or CPU collective was added. Fused `torch.optim.AdamW` remains available through `cpu_optimizer_backend = "torch"` for debugging.

The bounded design used 0.83 GiB pinned RAM for Qwen3-0.6B and survived accumulation 8, synchronous validation, and checkpoint save/resume. Its remaining known cost is the pinned-BF16 to pageable-FP32 CPU materialization: at sequence length 2K, steady steps are about 498 ms versus 343 ms for the earlier model-sized persistent-pinning design. This cost should become increasingly hidden by longer forward/backward compute.

## Production baseline order

### 1. Qwen3-30B-A3B on one 8-GPU node

Start with `Qwen/Qwen3-30B-A3B` or the exact production 30B-A3B variant. Use the custom implementation, fake fixed-length data, random weights, forced balanced routing, EP8, AdamW, no clipping, and native full offload. Keep compile and training quantization disabled for the first comparison, then enable the intended production settings after the baseline is stable.

Compare, where memory permits:

1. no optimizer offload;
2. optimizer-state-only offload;
3. full offload with `cpu_optimizer_backend = "torch"` for correctness;
4. full offload with `cpu_optimizer_backend = "native"`.

The relevant native configuration is:

```toml
[trainer.model]
impl = "custom"
ep = 8
optim_cpu_offload = { full = true, cpu_optimizer_backend = "native", transfer_buffer_count = 4, max_inflight_backwards = 16, timeout_seconds = 600 }

[trainer.model.debug]
random_init = true
force_balanced_routing = true

[trainer.optim]
type = "adamw"
max_norm = "None"
```

Do not use SignSGD for this baseline: the existing GLM fake-data debug config uses it, but the bounded pipeline in this PR is selected only for native full-offload AdamW.

### 2. GLM-5 four-node proxy on one 8-GPU node

GLM-5 has 78 layers: 3 dense layers followed by 75 MoE layers. A one-node truncation to 22 layers retains the 3 dense layers plus 19 MoE layers, closely matching one quarter of the full model's expert state. This is more representative for optimizer memory and bandwidth than a naive 19- or 20-layer truncation.

For the primary four-node proxy use:

```toml
[trainer.model]
name = "zai-org/GLM-5"
impl = "custom"
cp = 2
ep = 8
optim_cpu_offload = { full = true, cpu_optimizer_backend = "native", transfer_buffer_count = 4, max_inflight_backwards = 16, timeout_seconds = 600 }

[trainer.model.debug]
num_layers = 22
random_init = true
force_balanced_routing = true

[trainer.optim]
type = "adamw"
max_norm = "None"
```

Why CP2: on eight GPUs it leaves four-way DP/FSDP sharding, while the intended 32-GPU job with CP8 also has four-way DP/FSDP sharding. The 22-layer/CP2 run is therefore close to one target node's optimizer-state volume and slightly conservative for per-GPU compute. It cannot reproduce inter-node communication or aggregate four-node CPU/PCIe bandwidth, so a final 4-node confirmation remains mandatory.

Before launching, check host-RAM feasibility. Native full offload needs approximately 16 bytes per local trainable parameter for FP32 master, first moment, second moment, and accumulated gradient, plus ring, framework, checkpoint, and dataloader overhead. The 22-layer proxy intentionally approximates one quarter of the full model and may require roughly the same aggregate CPU state as one target node. Use measured parameter counts rather than assuming a 2 TiB node is sufficient.

Measured counts (meta-device instantiation of the custom `GlmMoeDsaForCausalLM` from the `zai-org/GLM-5` HF config): the 22-layer proxy has 190.8B trainable parameters, or 2.78 TiB of CPU state at 16 B/param — it does not fit a 2 TiB node and needs a ≥3 TiB host. A 14-layer truncation (3 dense + 11 MoE, 111.8B parameters, 1.63 TiB) fits a 2 TiB node with headroom. The full 743.9B model on four 2 TiB nodes needs 2.71 TiB/node, so the intended four-node full-offload job is itself RAM-infeasible at 16 B/param; it requires six or more such nodes or a per-parameter state reduction (dropping the FP32 gradient slab via ring-direct native Adam at accumulation 1 gives 12 B/param, 2.03 TiB/node — still too tight for overhead). Ready-to-launch configs for both baselines live in `benchmarks/offload/`.

Use fake fixed-length data and forced balanced routing throughout. Random initialization avoids weight-download and load-time noise and prevents an untrained router from concentrating tokens on a few experts.

## Sequence length and CP matrix

For every point, compare the same model/CP configuration with and without native full offload. Report median and p95 after at least two warm-up steps and five measured steps.

- Sweep sequence lengths 2K, 8K, 16K, 32K, 64K, and 128K.
- Keep CP fixed when possible so sequence scaling is not confused with a topology change.
- Separately sweep CP1, CP2, CP4, and CP8 at the longest lengths that fit.
- The required long-context endpoint is 128K with CP8.
- Test gradient accumulation 1, 2, and 8. Include a longer accumulation-8 soak after the short sweep.
- Repeat the final candidate with compile, activation offload, FP8 training, and other intended production settings enabled one at a time.

Capture step time, TPS, MFU, peak HBM, process RSS, pinned bytes and allocation time, CPU Adam time, BF16-to-FP32 materialization time, achieved D2H/H2D bandwidth, CPU utilization, NUMA locality, and ring occupancy/stalls. Record the exact GPU, CPU, RAM, PCIe/NVLink topology, thread count, affinity, and environment variables with every result.

## One-node 8xH200 baseline results (2026-08-12)

Qwen3-30B-A3B, custom impl, EP8, fake fixed data, random init, balanced routing, AdamW, no clipping, compile/quantization off. Node: 8xH200 SXM, 2x Xeon Platinum (240 hw threads, 2 NUMA nodes), 2.95 TiB RAM. Median over steps 3-7; configs in `benchmarks/offload/`.

Four variants at seq 8192, accumulation 1, `OMP_NUM_THREADS=28`:

| Variant | Step med | TPS | Peak HBM |
| --- | ---: | ---: | ---: |
| no offload | 4.57 s | 14.1k | 35.3 GiB |
| optimizer-state-only offload | 4.41 s | 13.2k | 35.3 GiB |
| full offload, torch backend | 6.93 s | 9.4k | 14.5 GiB |
| full offload, native backend | 6.75 s | 9.9k | 19.4 GiB |

Losses were bit-identical across all variants at every step. Sequence sweep, no offload vs native full offload (64K adds activation recompute to both):

| Seq | No offload | Full native | TPS retained | Peak HBM |
| ---: | ---: | ---: | ---: | ---: |
| 2K | 3.48 s | 6.77 s | 51% | 32.9 → 16.5 GiB |
| 8K | 4.57 s | 6.75 s | 68% | 35.3 → 19.4 GiB |
| 16K | 9.35 s | 10.28 s | 91% | 39.3 → 22.6 GiB |
| 32K | 19.84 s | 18.45 s | 108% | 46.9 → 29.9 GiB |
| 64K | 35.29 s | 34.62 s | 102% | 58.7 → 45.6 GiB |

Findings from the pipeline timing diagnostics (now built in, debug level):

- The launcher defaults trainer processes to `OMP_NUM_THREADS=1`, which serializes the CPU optimizer (15.7 s/step at 8K). The benchmark configs set 28 threads/rank; anything ≥14 is equivalent (below).
- The short-sequence floor (~6.1-6.8 s for 30.5B params) is host-DRAM-bandwidth-bound, not core-bound: the native Adam call measures ~2.9-3.4 s/step against a ~0.4 s isolated-compute estimate, is invariant to OMP 14/28/40, and ring depth 8 is slightly worse than 4. Eight ranks jointly move roughly 1 TB of host DRAM traffic per step (Adam state + materialization); PCIe waits are negligible (d2h_wait ≈ 0.05 s).
- `optim_cpu_offload.numa_bind = true` pins each rank to its GPU's socket before state allocation (first-touch locality). At 8K it trims the floor 6.39 → 6.13 s; at 16K it moves full offload from 10.28 s to 7.89 s — faster than the 9.35 s no-offload baseline. With binding, full offload is a net throughput win from 16K per-GPU sequence upward while roughly halving peak HBM.

Given production per-GPU sequence lengths sit at or above the crossover, the measured data supports the existing guidance: keep the bounded implementation and use `numa_bind`; the chunk-owned-D2H redesign (ring-direct BF16 consume, ~29% less DRAM traffic) is only warranted if short-sequence full offload becomes a production requirement.

GLM-5 20-layer proxy (171B params, CP2, EP8, native full offload, numa_bind, recompute, accumulation 1) on the same node — 22 layers (2.78 TiB CPU state) does not fit this node's 2.95 TiB, 20 layers does. Median of steps 2-5, loss constant at 11.9504 across all lengths:

| Seq | Step med | TPS | MFU | Peak HBM | Exposed drain | Host RAM free |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 25.6 s | 2.6k | 2.7% | 102 GiB | 17.4 s | 76 GiB |
| 32K | 23.1 s | 5.8k | 8.0% | 115 GiB | 11.8 s | 61 GiB |
| 64K | 37.2 s | 7.3k | 14.5% | 134 GiB | 6.9 s | 16 GiB |

At this parameter scale the CPU pipeline costs ~18 s of Adam per step (21.4B params/rank); 32K is faster end-to-end than 16K because the pipeline unhides. No no-offload reference exists — optimizer states (257 GiB/GPU) cannot fit on H200s, so full offload is what makes this model trainable on one node at all. The per-rank load matches the four-node full-model target (23.2B params/rank), so expect a similar ~7 s exposed drain at 64K there, shrinking further at the 128K endpoint. 64K runs at 95% HBM and 16 GiB free host RAM: 20 layers is the ceiling for this node class, and 128K will need activation offloading.

Note for fake-data debug runs: the entrypoints now skip the weight pre-download under `debug.random_init` (previously a GLM-5 run tried to pull ~1.4 TB of weights into the shared HF cache).

## Robustness gates

Before optimizing further:

- run at least 50 steps with accumulation 8;
- save a DCP checkpoint, resume it, and verify the next loss against an uninterrupted run;
- exercise synchronous validation followed by the optimizer update;
- verify pinned memory and process RSS remain stable across steps;
- confirm all ranks terminate cleanly after an injected timeout/error;
- run the winning configuration on the actual four-node topology, since a one-node proxy cannot expose inter-node FSDP/CP interactions.

## Next optimization pass

Only optimize after the baseline identifies exposed time. The main architectural candidate is chunk-owned D2H storage consumed directly by native Adam, which removes the extra CPU gradient materialization while keeping pinned memory bounded. Production-specific follow-ups are NUMA placement, CPU-thread affinity/count, CPU chunk size, ring depth, and transfer-stream scheduling. Ring depth should not be increased blindly: on the development host, increasing normal slots from 4 to 8 raised pinned memory and reduced performance.

If the materialization and Adam work are already hidden at production sequence lengths, keep the simpler bounded implementation rather than adding another pipeline layer.
