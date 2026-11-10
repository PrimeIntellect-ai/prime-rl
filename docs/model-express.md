# ModelExpress

`[weight_broadcast] type = "modelexpress"` sends full FSDP weight updates from the trainer to vLLM through [ModelExpress](https://github.com/ai-dynamo/modelexpress) (MX). Inference GPUs pull their shards over RDMA (NIXL) from a host snapshot of the trainer shards, so the trainer does not run a collective with inference.

## Setup

The `inference` extra installs the MX client and the `modelexpress-server` package, which puts the server and Valkey (the Redis-compatible store it keeps its state in) into the environment's `bin/`. Nothing else needs to be built.

`uv run rl`, and `uv run sft` with online evals, start a run-scoped server and Valkey (logs under `logs/modelexpress/`). Multi-node SLURM jobs start them on the trainer head node. To use an existing server instead, set `slurm.launch_modelexpress = false` and configure `weight_broadcast.host` and `weight_broadcast.port`.

## Config

```toml
[weight_broadcast]
type = "modelexpress"
staging_buffer_bytes = 4294967296  # bounded staging; omit to stage the full update
staging_buffers_count = 2          # overlap the next read with the current install
```

- `staging_mode`: `COPY_TO_HOST` (default) snapshots trainer shards into host memory. `COPY_TO_DEVICE` uses extra GPU memory. `IN_PLACE` copies nothing, but requires the trainer tensors to stay unchanged and already have the transfer dtype until inference has installed them.
- `metadata_port` (default 5700): base port of the trainer's NIXL metadata listener. Each trainer process offsets this by its first physical GPU and each local rank listens on that plus its device id.
- `staging_buffer_bytes` × `staging_buffers_count` bounds the extra receiver memory per inference GPU. Without it, each GPU stages a full copy of its shard, which can OOM at high `gpu_memory_utilization`.

## Behavior

- The first update is slower while MX plans the transfer; later updates reuse the plan.
- Inference is paused only after the trainer has published the version. A failed update leaves inference paused; restart the trainer and inference together, since there is no rollback.
- Trainer rank 0 and the orchestrator (in SFT, the online-eval process) must share the broadcast directory, which holds only rendezvous markers, not weights.
- MX transfers floating-point tensors only. Integer tensors are lookup tables that training never changes, such as DeepSeek-V4's hash-routing table `tid2eid`: inference keeps the values it loaded from the checkpoint.
- P/D deployments update like the other transports: prefill and decode GPUs form one transfer group. Each update pauses the prefill engines first, since a prefill engine's pause waits until decode has pulled the KV it holds, then the decode engines, installs the version on all of them and resumes them.
- With Dynamo (`[orchestrator.model.client.dynamo]`), the orchestrator discovers the Dynamo workers before every update instead of using a fixed endpoint list. Each update initializes ModelExpress on every worker that does not run it yet, such as a worker that joined or whose engine restarted, and drops workers that left. A worker that joins between updates serves its launch weights until the next update. Each worker must load the ModelExpress extension, `--worker-extension-cls prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker`.
- With speculative decoding, MX updates the target model and the draft keeps its own weights; a drafter that shares the target's embedding or LM head reads the updated ones. A draft swap (`[orchestrator.speculative_decoding] draft`, changed live) loads from disk at the next update, after the target is installed and before inference resumes.

## Pull

With `mode = "pull"`, each inference server installs published versions on its own instead of the orchestrator pushing every update. The trainer publishes a version and continues. It holds one copy, so before publishing it frees the previous one, waiting for any engine still reading that from the trainer. Older versions stay available from peers that installed them until they are three versions old. An inference server installs the newest version some source can serve before it answers its first request, which can be an older one while the newest is still spreading, and then follows newer ones, so a starting or restarting engine catches up on its own. Every token records the version that sampled it, so the orchestrator enforces `max_staleness` per token instead of waiting for engines. It also masks tokens of a version newer than the trainer's, which an engine that outlived a trainer restart from an earlier checkpoint samples until it installs the new trainer's weights. An engine drops its KV cache when it installs a version, since the cache salt follows the orchestrator's policy version rather than the weights each engine holds; requests running at that moment recompute their prefix on the new weights.

Every engine runs the same configuration. A new version reaches the fleet in waves: one engine at a time reads it from the trainer, and every other engine copies it from a peer that already installed it, each peer feeding one engine at a time. A peer copy goes straight into the live weights, so only that engine pauses while the rest keep serving, and peers already hold the converted inference layout. The trainer creates each version with one trainer reader and the inference server sets `MX_MAX_TENSOR_READERS` to 1 to get this, and `MX_TRANSFER_TIMEOUT` to 120s so an install whose source stops answering shuts the engine down instead of holding it paused. Staging retries only while no source can serve a version yet or its source is busy; any other error shuts the engine down. The server also boots each engine through ModelExpress without reading any weights, since it installs a version before serving, so each engine can serve its peers whether the trainer starts from HF weights, a checkpoint, or scratch; override any of them in the inference environment or with `vllm.load_format`.

Pull needs `staging_buffer_bytes`: a copy from the trainer streams through those host buffers into vLLM's own weight loader, as a checkpoint reload does, so any model vLLM can reload installs, no engine sets aside GPU memory for a second copy and `gpu_memory_utilization` can stay high. Each GPU reads only the rows of the trainer's tensors its loader keeps, so an expert parallel engine reads only its own experts; dense tensors are read whole on every tensor parallel rank. Push installs as before. A standalone inference server sets `[weight_broadcast]` with `type = "modelexpress"`, `mode = "pull"`, `host`, `port` and the staging fields itself. Evals run when the trainer publishes a version, not when engines install it, so an eval can sample from engines still on an older version. Pull needs one ModelExpress server per run, since engines follow every version published for their model.

## Limitations

LoRA is rejected at config validation. Pipeline parallelism is untested. Pull rejects disaggregated P/D and an expert parallel data parallel group spread over several servers, which the multi-node launcher starts for expert parallelism with more than one data parallel rank per replica; give each replica one data parallel rank.
