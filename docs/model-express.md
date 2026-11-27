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
- With speculative decoding, MX updates the target model and the draft keeps its own weights; a drafter that shares the target's embedding or LM head reads the updated ones. A draft swap (`[orchestrator.speculative_decoding] draft`, changed live) loads from disk at the next update, after the target is installed and before inference resumes.

## Verifying updates

`tools/check_weight_update_bit_exact.py` checks that updates install exactly what vLLM's own loader produces:

1. An engine cold-loads `MODEL_DIR`, a checkpoint holding the trainer's initial weights W0. The per-tensor digests it then reports (`GET /weight_fingerprint/model?buffers=true`: parameters, buffers and bare tensors such as fused or post-processed copies) are the reference.
2. The trainer loads the same checkpoint and checks that its weights, converted back, equal it bit for bit.
3. Instead of optimizer steps, the trainer publishes W0, W0 + D1, W0, W0 + D2 and W0 through the real transport, where D1 and D2 are seeded perturbations of every floating-point tensor it publishes. After each install the engine's digests are compared with the reference: every W0 install must restore every engine tensor bit for bit, and every perturbed install must change every floating-point engine parameter.

A pass shows that each update is installed, and that the install reproduces the loader's result for the same trainer weights, including fused, permuted and post-processed tensors (such as Mamba's `A = -exp(A_log)`) and dtypes. It runs no forward pass, and integer tensors are not perturbed, so they are only checked for equality.

```bash
uv run python tools/check_weight_update_bit_exact.py MODEL_DIR OUTPUT_DIR \
  --trainer-gpus 0,1 --inference-gpus 2,3 --vllm '{"max_model_len": 4096}'
```

`--trainer-gpus` sets the FSDP ranks, `--inference-gpus` the engine's tensor parallelism, `--transport filesystem` checks the filesystem transport instead, and `OUTPUT_DIR/report.json` lists every mismatching tensor by version. The integration test `tests/integration/test_modelexpress_bit_exact.py` runs the check on a tiny checkpoint of every architecture in the vLLM parity test, on two GPUs; cases that fail today are strict xfails that name the fix they wait for.

## Limitations

LoRA and Dynamo discovery are rejected at config validation. Pipeline parallelism is untested.
