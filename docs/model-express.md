# ModelExpress

`[weight_broadcast] type = "modelexpress"` sends full FSDP weight updates from the trainer to vLLM through [ModelExpress](https://github.com/ai-dynamo/modelexpress) (MX). Inference GPUs pull their shards over RDMA (NIXL) from a host snapshot of the trainer shards, so the trainer does not run a collective with inference.

## Setup

The `inference` extra installs the MX client (`modelexpress==0.8.0`) and the `modelexpress-server` package, which puts the matching server and Valkey (the Redis-compatible store it keeps its state in) into the environment's `bin/`. Nothing else needs to be built.

`uv run rl`, and `uv run sft` with online evals, start a run-scoped server and Valkey (logs under `logs/modelexpress/`). Multi-node SLURM jobs start them on the trainer head node. To use an existing server instead, set `slurm.launch_modelexpress = false` and configure `weight_broadcast.host` and `weight_broadcast.port`.

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

- The trainer and inference layouts need not match. Each trainer rank publishes the region of each tensor it holds and each inference worker reads the slices its weight loaders take, so trainer FSDP, HSDP (`dp_replicate`), expert parallelism (`ep`, also below the FSDP size and with uneven shards) and context parallelism (`cp`) combine with tensor, data and expert parallelism in vLLM.
- Each inference worker registers with MX under an id it derives itself, from the run's session, its GPU's UUID and its process id. The orchestrator sends every engine the same initialization and needs no knowledge of the inference layout: data-parallel engines, external-LB servers, P/D roles and replicas of any size each get their own ids.
- The first update is slower while MX plans the transfer; later updates reuse the plan.
- Inference is paused only after the trainer has published the version. A failed update leaves inference paused; restart the trainer and inference together, since there is no rollback.
- Trainer rank 0 and the orchestrator (in SFT, the online-eval process) must share the broadcast directory, which holds only rendezvous markers, not weights.
- MX transfers floating-point tensors only. Integer tensors are lookup tables that training never changes, such as DeepSeek-V4's hash-routing table `tid2eid`: inference keeps the values it loaded from the checkpoint.
- P/D deployments update like the other transports: prefill and decode GPUs form one transfer group. Each update pauses the prefill engines first, since a prefill engine's pause waits until decode has pulled the KV it holds, then the decode engines, installs the version on all of them and resumes them.
- With Dynamo (`[orchestrator.model.client.dynamo]`), the orchestrator discovers the Dynamo workers before every update instead of using a fixed endpoint list. Each update initializes ModelExpress on every worker that does not run it yet, such as a worker that joined or whose engine restarted, and drops workers that left. A worker that joins between updates serves its launch weights until the next update. Each worker must load the ModelExpress extension, `--worker-extension-cls prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker`.
- With speculative decoding, MX updates the target model and the draft keeps its own weights; a drafter that shares the target's embedding or LM head reads the updated ones. A draft swap (`[orchestrator.speculative_decoding] draft`, changed live) loads from disk at the next update, after the target is installed and before inference resumes.

## Engines that pull their weights

With `pull = true`, the engines move themselves onto each version the trainer publishes, instead of the orchestrator pausing every engine and pushing each update:

```toml
[weight_broadcast]
type = "modelexpress"
pull = true
```

- Each inference server polls the MX server for the newest version published after the one it serves, and skips the versions in between. While it serves, every worker stages that version on a background thread. Once all have, the server pauses its engines at the next step with the running requests kept, installs the version, labels the tokens sampled from then on with its step (`weight_versions` on every response) and resumes. A request that runs across a switch keeps its KV cache and samples its later tokens with the new version, which its labels record.
- The first version of a trainer stages while the engines are paused: that stage records on the live model how the trainer's tensors map onto the engine's. Later versions of the same trainer stage while serving.
- Staging while serving needs memory for it. Without `staging_buffer_bytes`, each GPU holds one more copy of its weights, and the reads overlap serving, so the pause only installs. With bounded staging, staging while serving only plans the transfer and the bytes move during the pause, as with push.
- On several nodes every DP rank is its own server. The ranks of an expert-parallel DP group share every forward pass, so the group's first rank pulls for all of them, through their admin routes (`/modelexpress/stage`, `/modelexpress/apply`, `/pause`, `/resume`), and gets the admin token for it. vLLM's data-parallel pause stops every rank of the group at the same step, so no forward pass mixes versions. Groups switch independently, each as soon as it is ready.
- The trainer publishes each version and goes on. It holds one version: before it publishes the next, it stops new reads of the previous one and waits only for the reads already in flight. Trainer host memory stays at one copy of the weights. At the end of training it keeps the last version until the engines serve it.
- The orchestrator only reads the version each engine serves (`GET /weight_version`), for its dispatch gate, the startup sync and the final evals. It keeps dispatching across a switch.
- A failed stage leaves the live weights untouched: the engines keep serving and retry with the newest version. A failed install may leave them partly written, so the engines stay paused and install the newest version from scratch until one succeeds; past `weight_broadcast.timeout` the server exits.
- Not supported yet: draft swaps (speculative decoding itself works, with the draft the engines launch with), rollout checkpoints, Dynamo and SFT online evals.

## Limitations

LoRA is rejected at config validation. Pipeline parallelism is untested.
