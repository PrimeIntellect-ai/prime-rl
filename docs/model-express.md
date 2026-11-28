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

- The first update is slower while MX plans the transfer; later updates reuse the plan.
- Inference is paused only after the trainer has published the version. A failed update leaves inference paused; restart the trainer and inference together, since there is no rollback.
- Trainer rank 0 and the orchestrator (in SFT, the online-eval process) must share the broadcast directory, which holds only rendezvous markers, not weights.
- MX transfers floating-point tensors only. Integer tensors are lookup tables that training never changes, such as DeepSeek-V4's hash-routing table `tid2eid`: inference keeps the values it loaded from the checkpoint.
- P/D deployments update like the other transports: prefill and decode GPUs form one transfer group. Each update pauses the prefill engines first, since a prefill engine's pause waits until decode has pulled the KV it holds, then the decode engines, installs the version on all of them and resumes them.
- With Dynamo (`[orchestrator.model.client.dynamo]`), the orchestrator discovers the Dynamo workers before every update instead of using a fixed endpoint list. Each update initializes ModelExpress on every worker that does not run it yet, such as a worker that joined or whose engine restarted, and drops workers that left. A worker that joins between updates serves its launch weights until the next update. Each worker must load the ModelExpress extension, `--worker-extension-cls prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker`.
- With speculative decoding, MX updates the target model and the draft keeps its own weights; a drafter that shares the target's embedding or LM head reads the updated ones. A draft swap (`[orchestrator.speculative_decoding] draft`, changed live) loads from disk at the next update, after the target is installed and before inference resumes.

## Verifying updates

`tools/check_weight_update_bit_exact.py` takes a local HF safetensors checkpoint, W0, and writes two deterministic perturbations of its floating-point tensors, W1 and W2. vLLM cold-loads each version. The trainer publishes W0, W1, W0, W2 and W0 through the real weight transport. Each live install must match its own cold load on every worker for tensor inventory, bytes, dtype, shape, strides, and storage aliasing. The check also requires the live parameters to keep their addresses across updates.

For ModelExpress, the checker verifies each published wire manifest for tensor names, dtypes, shapes, and shard geometry. `--wire-fp32` accepts a regex over native trainer names to specify the expected FP32 tensors independently. The default uses the model's dtype policy. `--forward` additionally compares greedy generations and sampled logprobs against the cold references.

```bash
uv run python tools/check_weight_update_bit_exact.py MODEL_DIR OUTPUT_DIR \
  --trainer-gpus 0,1 --inference-gpus 2,3 --vllm '{"max_model_len": 4096}' \
  --staging-buffer-bytes 4294967296 --staging-buffers-count 2 --forward
```

`--trainer-gpus` selects the FSDP ranks. `--inference-gpus` selects the engine devices, and `--vllm` can set the TP and DP layout explicitly. `--transport filesystem` exercises filesystem installs. Use an empty output directory for `report.json`, per-worker fingerprints, and logs. The temporary checkpoints need disk space for two additional copies of the model.

In GitHub Actions, dispatch **GPU Tests** with `mx-architecture` for the two-GPU sweep, `mx-distributed` for the four-Blackwell DeepSeek V4 MXFP8 check, or `mx-all` for both. These jobs run every selected case as a required pass and upload reports and logs. They require a native Linux runner with CUDA, a UCX-visible network interface and the requested GPUs. Choose its label with `mx_runner` or the repository variable `PRIME_RL_MX_GPU_RUNNER`. The fallback label is `mx-gpu`.

## Limitations

LoRA is rejected at config validation. Pipeline parallelism is untested.
