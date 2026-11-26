# ModelExpress

`[weight_broadcast] type = "modelexpress"` sends full FSDP weight updates from the trainer to vLLM through [ModelExpress](https://github.com/ai-dynamo/modelexpress) (MX). Inference GPUs pull their shards over RDMA (NIXL) from a host snapshot of the trainer shards, so the trainer does not run a collective with inference.

## Setup

The `inference` extra installs the MX client (`modelexpress==0.8.0.dev20261022`) and the `modelexpress-server` package, which puts the matching server and Valkey (the Redis-compatible store it keeps its state in) into the environment's `bin/`. Nothing else needs to be built.

`uv run rl` starts a run-scoped server and Valkey (logs under `logs/modelexpress/`). Multi-node SLURM jobs start them on the trainer head node. To use an existing server instead, set `slurm.launch_modelexpress = false` and configure `weight_broadcast.host` and `weight_broadcast.port`.

## Config

```toml
[weight_broadcast]
type = "modelexpress"
staging_buffer_bytes = 4294967296  # bounded staging; omit to stage the full update
staging_buffers_count = 2          # overlap the next read with the current install
```

- `staging_mode`: `COPY_TO_HOST` (default) snapshots trainer shards into host memory. `COPY_TO_DEVICE` uses extra GPU memory. `IN_PLACE` copies nothing, but requires the trainer tensors to stay unchanged and already have the transfer dtype until inference has installed them.
- `metadata_port` (default 5655): base port of the trainer's NIXL metadata listener. Each trainer GPU listens on this port plus its local device id.
- `staging_buffer_bytes` × `staging_buffers_count` bounds the extra receiver memory per inference GPU. Without it, each GPU stages a full copy of its shard, which can OOM at high `gpu_memory_utilization`.

## Behavior

- The first update is slower while MX plans the transfer; later updates reuse the plan.
- Inference is paused only after the trainer has published the version. A failed update leaves inference paused; restart the trainer and inference together, since there is no rollback.
- Trainer rank 0 and the orchestrator must share the broadcast directory, which holds only rendezvous markers, not weights.
- MX transfers floating-point tensors only. Integer tensors are lookup tables that training never changes, such as DeepSeek-V4's hash-routing table `tid2eid`: inference keeps the values it loaded from the checkpoint.

## Limitations

LoRA, speculative decoding, Dynamo discovery and SFT online evals are rejected at config validation. P/D disaggregated inference is not supported, and pipeline parallelism is untested.
