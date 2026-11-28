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

## Swapping checkpoints on a running deployment

A deployment that serves exported checkpoints (offline evals of an RL run) can move to another checkpoint without a restart: a checkpoint publisher offers the export as an MX version, and the engines install it through `POST /swap-checkpoint`.

```toml
[weight_broadcast]
type = "modelexpress"
host = "<MX server host>"
port = 8001
model_name = "my-run"   # the MX model the publisher publishes under; defaults to vllm.model
```

**Publisher.** `tools/publish_checkpoint.py` reads an HF safetensors export (bf16, fp16 or fp32 tensors) into page-locked host memory and serves it to the engines over RDMA until it is released:

```bash
uv run python tools/publish_checkpoint.py /exports/step-300 300 --model-name my-run --mx-host <host> --ranks 8
curl -X POST http://<host>:8090/publish -d '{"model_dir": "/exports/step-400", "step": 400}'   # roll to the next one
curl -X POST http://<host>:8090/release                                                     # or SIGTERM
```

- Its ranks (`--ranks`, one process each) split the checkpoint's tensors, read their share with `--read-threads` threads each and register it with NIXL. MX reads those buffers in place, so the publisher holds one copy of the checkpoint in host memory. Each rank needs a GPU visible (one is enough for all of them): MX's NIXL agent and the pinning need CUDA.
- With at least as many GPUs as ranks (and `CUDA_VISIBLE_DEVICES` unset), rank r uses GPU r, the RDMA port nearest it and that port's NUMA node (`--rails`): each rank registers its tensors with one port, which on a 530 GiB checkpoint over 8 ranks cuts the publish from ~100 s to ~20 s and an engine's connection to a new publisher from ~34 s to ~0.1 s.
- Reads bypass the page cache (O_DIRECT): on Lustre, the client cache of a checkpoint read through it can grow to most of the host's memory beside the pinned copy. The publisher refuses to start, and a publish fails, rather than leave less host memory available than the larger of `--min-available-gib` (200) and `--min-available-fraction` (0.15) of the total.
- The tensors keep the checkpoint's names and layout, which vLLM's own loader reads, so the engines install them without a conversion. An engine converts a version only when its tensor names are PrimeRL's (a trainer's), as the trainer does when it loads a checkpoint.
- A version is numbered with its step (`version_number`), and engines label the tokens they sample with it.
- The publisher holds one version at a time and keeps one trainer mesh for its lifetime. To publish the next checkpoint it marks the current version RELEASING, so no new reads start, waits until the reads in flight end, reads the new checkpoint into the same buffers and publishes it. A checkpoint whose tensors differ in names, shapes or dtypes is refused before the current version is retired: start a new publisher for it.
- MX publishes floating-point tensors only. The engines keep the integer tensors they loaded at startup (DeepSeek-V4's hash-routing tables), which is right only while the new checkpoint holds the same ones. The version's uid carries a digest of the checkpoint's integer tensors, and a swap to a version whose digest differs from the engines' is refused (409): serving it needs a restart on that checkpoint.

**Swap.** `POST /swap-checkpoint {"step": 400, "mode": "drain", "drain_timeout_s": 300}` (or `"version_uid"`), behind the admin token, on the first rank of a DP group (`weight_broadcast.group`, which the multi-node templates pass to it) or on a server that holds all of its engines:

1. It resolves the target: the newest READY version of the step under `weight_broadcast.model_name`. A version the engines already serve returns `{"unchanged": true}`.
2. Every worker stages it while the engines keep serving. A missing or unreadable version fails here (404, 502) and nothing changes.
3. New requests stop entering the engines: they wait at the API servers, they are not rejected. Requests in flight decide the `mode`: `fail` (default) answers 409 if there are any; `drain` waits for them up to `drain_timeout_s`, else 409; `abort` aborts them, and they finish with the `abort` finish reason. A 409 changes nothing.
4. The group pauses (vLLM's data-parallel pause stops every rank at the same step), installs the version, drops the prefix cache (its blocks hold KV of the old weights), labels the tokens sampled from then on with the step and resumes. The waiting requests then run on the new weights.
5. It answers `{"step", "version_uid", "unchanged": false, "cold", "timings": {"stage", "drain", "pause", "apply", "total"}}`. `GET /weight_version?uid=true` returns `{"version", "uid"}` on every rank, and every completion can return its per-token labels (`return_weight_versions`).

- The first version of a mesh that a group installs (the first swap to a publisher process) stages while paused, since that stage records on the live model how the mesh's tensors map onto the engine's: `cold: true`, and its `stage_paused` time counts toward `pause`. Later versions of the same publisher stage while serving. A worker's first stage also builds MX's receive-buffer allocator, a JIT-compiled extension cached under `$TORCH_EXTENSIONS_DIR` (by default `~/.cache/torch_extensions`): tens of seconds of pause whenever that cache is cold, such as after another environment rebuilt it. Give each environment its own `TORCH_EXTENSIONS_DIR` to keep it warm.
- With bounded staging (`staging_buffer_bytes`), each buffer must hold the largest tensor a worker receives (the embedding or LM head), and the bytes move during the install, inside the pause.
- One swap runs at a time per group. A second request for the same version waits for the first one's result; one for another version gets 409. `GET /swap-checkpoint` reports the group, the version it serves and where a running swap is (`phase`: staging, draining, paused).
- A failed stage, also a paused one, leaves the weights as they were, and the group serves on. A failed install may leave them partly written: the group stays paused, every rank answers `/health` with 503 so the router drops it, the request gets 500, requests wait at the API servers, and the group retries the install while paused until it succeeds or `weight_broadcast.timeout` passes. It never resumes on mixed weights, also when a later stage fails. A new swap request, to any version, takes over from a failed one. An install that failed while moving bytes under bounded staging can leave MX unable to stage again in that process ("streaming cleanup is unproven"); the group then stays out of service until it restarts.
- A server that pulls from a trainer (`pull = true`) does not swap checkpoints (409), and swaps need one API server per engine (`api_server_count = 1`, as every external-LB rank has).
- NIXL ships its own UCX. Engines that swap must not point `LD_LIBRARY_PATH` or `UCX_MODULE_DIR` at another UCX build: mixing the two aborts every worker as soon as its NIXL agent starts (`undefined symbol: uct_ib_mlx5_ext_register`).
- With speculative decoding, the draft keeps its weights unless the request names a matching one (`draft_dir`: a directory or a Hub id in the local cache, with the served draft's architecture), which loads in the same pause after the version. Either way decoding stays lossless, and a draft that shares the target's embedding or LM head reads the new ones.

**Whole deployment.** `tools/swap_checkpoint.py --router <url> --step 400 --mode drain` asks every engine behind the router (or `--servers`) for its DP group and swaps every group at once, so the deployment pauses together; it exits 0 only if all of them serve the step. `--rolling` swaps one group at a time and takes it out of the router's rotation meanwhile, so the others keep serving. Guarded engines take the admin token from `$PRIME_RL_ADMIN_TOKEN`; a DP group's first rank needs it too, to call its ranks' admin routes (the multi-node inference template passes it from `$ADMIN_TOKEN_SEND_ENV`, `NAME=token`). Serve under a stable model name (`vllm.served_model_name`): the step is in the labels, not the name.

## Limitations

LoRA is rejected at config validation. Pipeline parallelism is untested.
