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
