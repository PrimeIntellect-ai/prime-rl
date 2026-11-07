# ModelExpress

`[weight_broadcast] type = "modelexpress"` sends full FSDP weight updates from the trainer to vLLM through [ModelExpress](https://github.com/ai-dynamo/modelexpress) (MX). Inference GPUs pull their shards over RDMA (NIXL) from a host snapshot of the trainer shards, so the trainer does not run a collective with inference.

## Setup

The GPU extra pins the MX client. Build the matching server and its Redis backend once on the shared filesystem:

```bash
bash scripts/install_modelexpress.sh
```

`uv run rl` starts a run-scoped server and Redis (logs under `logs/modelexpress/`). Multi-node SLURM jobs start them on the trainer head node. To use an existing server instead, set `slurm.launch_modelexpress = false` and configure `weight_broadcast.host` and `weight_broadcast.port`.

## Config

```toml
[weight_broadcast]
type = "modelexpress"
staging_buffer_bytes = 4294967296  # bounded staging; omit to stage the full update
staging_buffers_count = 2          # overlap the next read with the current install
```

- `staging_mode`: `COPY_TO_HOST` (default) snapshots trainer shards into host memory. `COPY_TO_DEVICE` uses extra GPU memory. `IN_PLACE` copies nothing, but requires the trainer tensors to stay unchanged and already have the transfer dtype until inference has installed them.
- `staging_buffer_bytes` × `staging_buffers_count` bounds the extra receiver memory per inference GPU. Without it, each GPU stages a full copy of its shard, which can OOM at high `gpu_memory_utilization`.

## Behavior

- The first update is slower while MX plans the transfer; later updates reuse the plan.
- Inference is paused only after the trainer has published the version. A failed update leaves inference paused; restart the trainer and inference together, since there is no rollback.
- Trainer rank 0 and the orchestrator must share the broadcast directory, which holds only rendezvous markers, not weights.

## Pull

With `mode = "pull"`, each inference server keeps itself on the newest published version instead of the orchestrator pushing every update. The trainer publishes a version and continues, keeping the two newest and retiring older ones once no engine still reads them. An inference server installs the newest version before it answers its first request and then follows every newer one, so a starting or restarting engine catches up on its own. Every token records the version that sampled it, so the orchestrator enforces `max_staleness` per token instead of waiting for engines.

Every engine runs the same configuration. A new version reaches the fleet in waves: one engine at a time reads it from the trainer, and every other engine copies it from a peer that already installed it, each peer feeding one engine at a time. A peer copy goes straight into the live weights, so only that engine pauses while the rest keep serving, and peers already hold the converted inference layout. The inference server sets `MX_MAX_TRAINER_READERS` to the tensor-parallel size and `MX_MAX_TENSOR_READERS` to 1 to get this, and loads weights through ModelExpress so each engine can serve its peers; override any of them in the inference environment or with `vllm.load_format`.

Pull needs `staging_buffer_bytes`: a copy from the trainer streams through those buffers into the live weights, so no engine sets aside memory for a second copy and `gpu_memory_utilization` can stay high. A standalone inference server sets `weight_broadcast.pull.server = "host:port"` and the staging fields itself.

## Limitations

LoRA, speculative decoding, Dynamo discovery and SFT online evals are rejected at config validation. P/D disaggregated inference is not supported, and pipeline parallelism is untested.
