# GLM-5.2 FP8 R2E with external Dynamo inference

This one-step smoke recipe runs a distributed Prime trainer and orchestrator
against a separately managed Dynamo deployment serving
`zai-org/GLM-5.2-FP8`. Dynamo owns the frontend, vLLM engines, and one native
gRPC sidecar per engine group; Prime discovers the mutable engine control
surface through `/v1/rl/workers`.

The recipe is split into trainer and orchestrator files because the external
Dynamo DGD and the multi-node trainer have independent lifecycles. It does not
add a Prime inference configuration or require Prime's launcher to manage the
DGD.

## Validated topology

| Component | Shape | GPUs |
|---|---|---:|
| Dynamo prefill | 2 nodes, DP4 x TP2 x PP1 x EP8 | 8 |
| Dynamo decode | 2 nodes, DP4 x TP2 x PP1 x EP8 | 8 |
| Prime trainer | 4 nodes, FSDP16 x CP4 x EP8 | 16 |

The two discovery records must report `prefill` and `backend` components with
a combined `world_size` of 16. If the DGD topology changes, update
`weight_broadcast.inference_world_size` in both TOML files to the atomic sum
returned by the same `/v1/rl/workers` response.

The inference engines must load
`prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker`, expose vLLM's
admin routes, and run the version-matched `vllm-rs` and
`dynamo-vllm-sidecar` binaries described in [`../README.md`](../README.md).
For mutable GLM weight reloads, launch every vLLM rank with `--enforce-eager`.

## Configure

Replace the checked-in service names when the DGD and trainer use different
DNS names:

- `model.client.base_url`: Dynamo OpenAI frontend on port 8000;
- `model.client.dynamo_discovery_url`: Dynamo RL discovery on port 8001;
- `weight_broadcast.host`: trainer rank zero, reachable from every inference
  engine.

The R2E harness uses Prime sandboxes. Configure the normal Prime credentials,
or replace the runtime with the sandbox backend used by your cluster. Model
and dataset caches should be shared across the trainer and inference nodes.

Verify Dynamo before starting Prime:

```bash
curl --fail http://dynamo-frontend:8000/v1/models
curl --fail http://dynamo-frontend:8001/v1/rl/workers
```

## Run

Launch the trainer on four 4-GPU nodes with the cluster's distributed runner.
For example, rank zero's rendezvous address can be passed to `torchrun` while
all ranks consume the same trainer file:

```bash
uv run torchrun \
  --nnodes=4 --nproc-per-node=4 \
  --rdzv-backend=c10d --rdzv-endpoint="$TRAINER_RANK_ZERO:29501" \
  --node-rank="$NODE_RANK" \
  -m prime_rl.trainer.rl.train \
  @ examples/dynamo/glm52_fp8_r2e/trainer.toml \
  --output-dir /shared/glm52-dynamo-r2e/train
```

After trainer rank zero opens port 29500, launch the orchestrator once:

```bash
uv run orchestrator \
  @ examples/dynamo/glm52_fp8_r2e/orchestrator.toml \
  --output-dir /shared/glm52-dynamo-r2e/train/run_0
```

The gate succeeds when one optimizer step completes, the NCCL update settles
on all 16 inference ranks, and a post-update multi-turn rollout completes
without changing its Dynamo session assignment.
