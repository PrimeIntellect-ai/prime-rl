# Dynamo on SLURM

This recipe gives Prime-RL one trainer node and one Dynamo inference node. The
SLURM launcher starts, supervises, and cleans up:

- job-local etcd;
- one Dynamo frontend with the RL discovery listener;
- one native-gRPC vLLM engine per local DP rank;
- one `dynamo-vllm-sidecar` per engine;
- the Prime trainer, orchestrator, and environment server.

## Prepare artifacts once on the shared checkout

```bash
git submodule update --init --recursive
uv sync --all-extras --all-packages
examples/dynamo/scripts/build_vllm_wheel.sh
examples/dynamo/scripts/build_dynamo_artifacts.sh
examples/dynamo/scripts/install_artifacts.sh
```

The inference environment lives at `.venv-dynamo`; Prime-RL continues to use
`.venv`. Both paths and `dist/dynamo/dynamo-vllm-sidecar` must be visible on all
allocated nodes.

## Configure and submit

Edit `partition`, `account`, and any cluster-specific Slurm settings in
`rl.toml`, then run:

```bash
uv run rl @ examples/dynamo/slurm-managed/rl.toml   --output-dir outputs/dynamo-slurm --clean-output-dir
```

Preview the rendered job without submitting:

```bash
uv run rl @ examples/dynamo/slurm-managed/rl.toml   --output-dir /tmp/dynamo-slurm --clean-output-dir --dry-run
bash -n /tmp/dynamo-slurm/rl.sbatch
```

`[dynamo] enabled = true` replaces the normal global `vllm-router` launch. The
orchestrator uses the Dynamo frontend for generation and receives
`dynamo_discovery_url` from the generated job script. Direct admin endpoints and
world sizes come from `/v1/rl/workers`.


## Configuration reference

The `[dynamo]` block supports:

```toml
[dynamo]
enabled = true
env_path = ".venv-dynamo"
sidecar_path = "dist/dynamo/dynamo-vllm-sidecar"
etcd_path = "etcd"
discovery_port = 8001
etcd_port = 2379
etcd_peer_port = 2380
engine_port = 8100
grpc_port = 50051
system_port = 9000
namespace = "prime-rl"
```

The launcher appends the Slurm job ID to the namespace, starts job-local etcd on
inference node zero, and advertises each engine's node-reachable HTTP admin URL.
The native gRPC and engine HTTP base ports are reused on different nodes; local
DP ranks use consecutive ports. The custom `vllm-rs` splitter keeps frontend
flags on the Rust process and forwards vLLM engine flags to its managed Python
engine. Override these values when your cluster reserves
one of the defaults.

## Scale inference

Each inference node starts `gpus_per_node / tensor_parallel_size` independent
engines. Therefore:

```text
inference_world_size = num_infer_nodes * gpus_per_node
```

for the aggregated topology. Keep `weight_broadcast.inference_world_size` equal
to the sum of worker `world_size` values returned by one discovery snapshot.

This initial Slurm path supports aggregated dense inference. Disaggregated
prefill/decode and cross-engine expert-parallel deployment remain unsupported by
the launcher.

## Logs

```text
outputs/dynamo-slurm/logs/inference/dynamo-etcd.log
outputs/dynamo-slurm/logs/inference/dynamo-frontend.log
outputs/dynamo-slurm/logs/inference/node_0.log
outputs/dynamo-slurm/logs/orchestrator.log
outputs/dynamo-slurm/logs/trainer/node_0.log
```
