# Optimizer-offload baseline matrix

Ready-to-launch configs for the production baseline described in the PR.
All runs use the SFT entrypoint with fake fixed-length data, random weights,
forced balanced routing, AdamW, no gradient clipping, and compile/quantization
disabled. Requires an 8-GPU node.

## Qwen3-30B-A3B (step 1)

Three variants, in comparison order:

```bash
# 1. no optimizer offload (the default)
uv run sft @ benchmarks/offload/qwen3-30b-base.toml
# 2. full offload, fused PyTorch AdamW (correctness reference)
uv run sft @ benchmarks/offload/qwen3-30b-base.toml @ benchmarks/offload/full-torch.toml
# 3. full offload, native CPU AdamW (production candidate)
uv run sft @ benchmarks/offload/qwen3-30b-base.toml @ benchmarks/offload/full-native.toml
```

Sweeps compose on top:

```bash
--data.seq-len 32768                 # sequence lengths 2K/8K/16K/32K/64K/128K
--data.batch-size 64                 # accumulation 8 (global batch / dp ranks)
```

The launcher defaults every trainer process to `OMP_NUM_THREADS=1`, which
serializes the CPU optimizer and roughly doubles full-offload step time
(measured 15.7 s → 6.5 s at 28 threads/rank on 8×H200). The full-offload
configs here set `env_vars.OMP_NUM_THREADS = "28"`; budget roughly
cores/ranks for other hosts.

Report median and p95 step time over steps 3–7 (two warm-up steps). Capture
step time, TPS, MFU, peak HBM, process RSS, pinned bytes, CPU Adam time,
BF16-to-FP32 materialization time, D2H/H2D bandwidth, CPU utilization, and
ring occupancy, plus the exact hardware/topology/env for every point.

## GLM-5 proxy (step 2)

```bash
uv run sft @ benchmarks/offload/glm5-proxy.toml
```

### Host-RAM feasibility (measured)

Trainable parameters measured by instantiating the custom `GlmMoeDsaForCausalLM`
on the meta device from the `zai-org/GLM-5` HF config. Native full offload holds
16 B/param of CPU state (FP32 master + both Adam moments + accumulated
gradient), before ring/framework/dataloader overhead.

| Truncation | Params | CPU state @16 B | Fits 2 TiB node? |
| --- | ---: | ---: | --- |
| 22 layers (3 dense + 19 MoE) | 190.8B | 2.78 TiB | no — needs ≥3 TiB host |
| 14 layers (3 dense + 11 MoE) | 111.8B | 1.63 TiB | yes, ~370 GiB headroom |
| 78 layers (full) on 4 nodes | 743.9B | 2.71 TiB/node | no |

The last row is the important one: the intended four-node full-offload job
exceeds 2 TiB/node by itself. At 16 B/param it needs six or more 2 TiB nodes,
or a per-param state reduction (e.g. consuming ring gradients directly in the
native kernel at accumulation 1 removes the 4 B/param FP32 gradient slab,
bringing four nodes to 2.03 TiB/node — still too tight for overhead).
