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

With `mode = "pull"`, each inference server keeps itself on the newest published version instead of the orchestrator pushing every update. The trainer publishes a version and continues. It holds one copy and publishes a version only once it has freed the previous one; while an engine is still reading that from the trainer, it skips the publication and tries again next step. Older versions stay available from peers that installed them until they are three versions old. An inference server installs the newest version before it answers its first request and then follows every newer one, so a starting or restarting engine catches up on its own. Every token records the version that sampled it, so the orchestrator enforces `max_staleness` per token instead of waiting for engines. It also masks tokens of a version newer than the trainer's, which an engine that outlived a trainer restart from an earlier checkpoint samples until it installs the new trainer's weights. An engine drops its KV cache when it installs a version, since the cache salt follows the orchestrator's policy version rather than the weights each engine holds; requests running at that moment recompute their prefix on the new weights.

Every engine runs the same configuration. A new version reaches the fleet in waves: one engine at a time reads it from the trainer, and every other engine copies it from a peer that already installed it, each peer feeding one engine at a time. A peer copy goes straight into the live weights, so only that engine pauses while the rest keep serving, and peers already hold the converted inference layout. The inference server sets `MX_MAX_TRAINER_READERS` and `MX_MAX_TENSOR_READERS` to 1 to get this, and `MX_TRANSFER_TIMEOUT` to 120s so an install whose source stops answering shuts the engine down instead of holding it paused, and loads weights through ModelExpress so each engine can serve its peers; override any of them in the inference environment or with `vllm.load_format`.

Pull needs `staging_buffer_bytes`: a copy from the trainer streams through those buffers into the live weights, so no engine sets aside memory for a second copy and `gpu_memory_utilization` can stay high. A standalone inference server sets `weight_broadcast.pull.server = "host:port"` and the staging fields itself. Pull needs one ModelExpress server per run, since engines follow every version published for their model.

## Limitations

LoRA, speculative decoding, Dynamo discovery and SFT online evals are rejected at config validation. P/D disaggregated inference is not supported, and pipeline parallelism is untested. Pull installs per inference server, so expert parallelism across separately launched data parallel ranks (vLLM's external or hybrid load balancing, as the multi-node SLURM templates run it) can mix versions within one forward pass; run each engine behind one API server instead.
