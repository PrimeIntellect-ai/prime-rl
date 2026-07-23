# GLM-5.2 FP8 R2E with external Dynamo inference

This three-step smoke recipe runs a distributed Prime trainer and orchestrator
against a separately managed Dynamo deployment serving
`zai-org/GLM-5.2-FP8`. Dynamo owns the frontend, vLLM engines, and one native
gRPC sidecar per engine group; Prime discovers the mutable engine control
surface through `/v1/rl/workers`.

The recipe is split into trainer and orchestrator files because the external
Dynamo DGD and the multi-node trainer have independent lifecycles. It does not
add a Prime inference configuration or require Prime's launcher to manage the
DGD.

## Reference topology

| Component | Shape | GPUs |
|---|---|---:|
| Dynamo prefill | 2 nodes, DP4 x TP2 x PP1 x EP8 | 8 |
| Dynamo decode | 2 nodes, DP4 x TP2 x PP1 x EP8 | 8 |
| Prime trainer | 4 nodes, FSDP16 x CP4 x EP8 | 16 |

The checked-in configuration targets this topology; it is not a claim that
every cluster can use these parallelism dimensions unchanged. The two
discovery records must report `prefill` and `backend` components with a
combined `world_size` of 16. If the DGD topology changes, update
`weight_broadcast.inference_world_size` in both TOML files to the atomic sum
returned by the same `/v1/rl/workers` response.

The inference engines must load
`prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker`, expose vLLM's
admin routes, and run the version-matched `vllm-rs` and
`dynamo-vllm-sidecar` binaries described in [`../README.md`](../README.md).
For mutable GLM weight reloads, launch every vLLM rank with `--enforce-eager`.
When serving a snapshot path, set `--served-model-name
zai-org/GLM-5.2-FP8`. The tested GLM entrypoint also uses the `glm47` tool
parser, `glm45` reasoning parser, the model chat template, and complementary
NIXL `kv_producer`/`kv_consumer` roles for prefill/decode.
The trainer and every inference engine must also use a compatible NCCL
transport. Apply cluster-specific settings such as `NCCL_IB_DISABLE`,
`NCCL_SOCKET_IFNAME`, and the NCCL network plugin consistently on both sides;
do not force Socket on the trainer while allowing inference to select IB.
The filesystem rollout transport requires the orchestrator and all trainer
nodes to mount the same read-write shared output root.

## Configure

Initialize the R2E environment submodule and install its workspace package:

```bash
git submodule update --init -- deps/research-environments
uv sync --package prime-rl --package r2e-gym-v1
```

The taskset intentionally uses the current `r2e-gym-v1` default,
`PrimeIntellect/R2E-Gym-Subset-Verified`, instead of pinning an older dataset
override in this recipe.

Replace the checked-in service names when the DGD and trainer use different
DNS names:

- `model.client.base_url`: Dynamo OpenAI frontend on port 8000;
- `model.client.dynamo_discovery_url`: Dynamo RL discovery on port 8001;
- `weight_broadcast.host`: trainer rank zero, reachable from every inference
  engine.

The R2E harness uses Prime sandboxes. Configure the normal Prime credentials,
or replace the runtime with the sandbox backend used by your cluster. Model
and dataset caches should be shared across the trainer and inference nodes.

Multi-turn affinity is not enabled merely by sending a session header. Start
the Dynamo frontend with `--router-session-affinity-ttl-secs <seconds>` (or
`DYN_ROUTER_SESSION_AFFINITY_TTL_SECS`) and choose an idle TTL longer than the
longest expected R2E turn gap. Prime maps each trajectory ID to the canonical
`X-Dynamo-Session-ID` header in `orchestrator.toml`.

Verify both the model and the complete atomic worker snapshot before starting
Prime. A successful HTTP status alone is insufficient:

```bash
MODEL=zai-org/GLM-5.2-FP8
curl -fsS http://dynamo-frontend:8000/v1/models |
  jq -e --arg model "$MODEL" '.data | any(.id == $model)'
curl -fsS http://dynamo-frontend:8001/v1/rl/workers |
  jq -e --arg model "$MODEL" '
    .protocol_version == 1 and
    (.workers | length == 2) and
    (all(.workers[]; .model == $model and
      ((.error // "") == "") and
      (.instance_id != null) and
      ((.admin_base_url // "") != ""))) and
    ([.workers[].instance_id] | unique | length == 2) and
    ([.workers[].admin_base_url] | unique | length == 2) and
    ([.workers[] | select(.model == $model) | .component] | sort == ["backend", "prefill"]) and
    ([.workers[] | select(.model == $model) | .world_size] | add == 16)
  '
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

The gate succeeds when the first optimizer step completes, policy version 1
settles on all 16 inference ranks through NCCL, and a later multi-turn rollout
completes without changing its Dynamo session assignment. Three steps are
required because finite NCCL runs skip broadcasts once
`step >= max_steps - 1`; this leaves step 1 as the first non-final broadcast
slot. The disabled post-batch zero-advantage filter keeps this small smoke run
from stalling on a homogeneous batch; enable it for a production training run.
