# Deployment

You can deploy PRIME-RL on a single GPU and larger multi-node clusters.

## SFT

### Single-GPU

For training on a single GPU, no communication orchestration is required and you can choose whether to start your trainer using our trainer entrypoint or using `torchrun`.

To start with our `sft` entrypoint

```bash
uv run sft ...
```

To do the same thing, but using `torchrun`

```bash
uv run torchrun src/prime_rl/trainer/sft/train.py ...
```

### Multi-GPU

For training on multiple GPUs, use `torchrun` with the `--nproc-per-node` flag.

```bash
uv run torchrun \
  --local-rank-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py ...
```

*The `--local-rank-filter` flag is used to only log the logs from the master rank, as detailed in [logs](logs.md).*

### Multi-Node

For training on multiple nodes, use `torchrun` with the `--nnodes`, `--node-rank`, and `--rdzv-endpoint` flags.

First, decide which node will be your head node and find a reachable private IP address for it. If your nodes are not colocated, you will likely need to setup VPN (e.g. [Tailscale](https://tailscale.com)) for the nodes to reach each other. 

(*Skip this step if the default network interface is sufficient.*) Make sure to set the network interface for GLOO and NCCL to one that allows all nodes to reach each other.

```bash
# On both nodes
export GLOO_SOCKET_IFNAME=...
export NCCL_SOCKET_IFNAME=...
```
 
Then, configure the rendezvous endpoint to allow the nodes to find each other. Here, `MASTER_ADDR` is the private IP address of the head node and `MASTER_PORT` is a free port on the head node, typically port 29500 for `torchrun`.

```bash
# On both nodes
export MASTER_ADDR=...
export MASTER_PORT=...
```

Then, on the head node, run

```bash
# On node 0
uv run torchrun \
  --nnodes 2 \
  --node-rank 0 \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local-rank-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py ...
```

And on the second node, run

```bash
# On node 1
uv run torchrun \
  --nnodes 2 \
  --node-rank 1 \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local-rank-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py ...
```

### SLURM

TBD.

## Inference

We rely on vLLMs multi-node deployment primitives and load balancing for multi-node deployments. Currently, vLLM supports multi-node data parallel deployment ([docs](https://docs.vllm.ai/en/v0.10.0/serving/data_parallel_deployment.html)).

First, decide which node will be your head node and find a reachable private IP address for it. If your nodes are not colocated, you will likely need to setup VPN (e.g. [Tailscale](https://tailscale.com)) for the nodes to reach each other. 

(*Skip this step if the default network interface is sufficient.*) Make sure to set the network interface for GLOO and NCCL to one that allows all nodes to reach each other.

```bash
# On both nodes
export GLOO_SOCKET_IFNAME=...
export NCCL_SOCKET_IFNAME=...
```
 
Then, configure the data parallel address as the private IP address of the head node.

```bash
# On both nodes
export DATA_PARALLEL_ADDRESS=...
export DATA_PARALLEL_RPC_PORT=...
```

To run TP=4 and DP=4 with DP ranks 0 and 1 on the head node and DP ranks 2 and 3 on the second node, run

```bash
# On node 0
uv run inference \
	--data-parallel-size 4 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT
```

```bash
# On node 1
uv run inference \
	--data-parallel-size 4 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT \
	--data-parallel-start-rank 2 \
	--headless
```

## RL

### Single-GPU Training

If you only have access to a single GPU, you may still be able to run small RL experiments. To do so, configure your inference server to use only a fraction of the available memory to leave some space for the trainer.

For example, to run an RL training on a single GPU while using 50% of the available memory for the inference server, run

```bash
bash scripts/tmux.sh
```

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --inference.gpu-memory-utilization 0.5
```

*Make sure to tune the `--gpu-memory-utilization` value such that you have enough GPU memory for the RL trainer.* 

You can also set this up by starting each submodule manually.

```bash
# Run this in the `Inference` pane
uv run inference @ path/to/infer.toml --gpu-memory-utilization 0.5
```

```bash
# Run this in the `Orchestrator` pane
uv run orchestrator @ path/to/orch.toml
```

```bash
# Run this in the `Trainer` pane
uv run trainer @ path/to/train.toml
```

### Multi-GPU Training

For single-node training, we recommend using the `rl` entrypoint to conveniently start all components, i.e. the inference server, the orchestrator, and the trainer. 

By default, the inference server starts on GPU ID 0 and the trainer on GPU ID 1.

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
```

You can configure to GPU IDs to use for the inference server and the trainer. For example, to run the inference server on GPUs IDs 0-5 with data parallelism and the trainer on GPUs IDs 6-7

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --inference-gpu-ids 0,1,2,3,4,5 \
  --trainer-gpu-ids 6,7 \
  --inference.parallel.dp 6
```

### Parallel Experiments

For quick ablations, it can be more efficient to parallelize experiments within a node (e.g. split your GPUs to run two experiments in parallel). For example, if you have access to 4 GPUs and your experiment fits on 2 GPUs, you can parallelize two experiments as follows:

Start the first experiment in a tmux session `exp1` with outputs directory `outputs1`. Specify it both in the tmux script, as well as in the start command (*will use the first 2 GPUs*)

```bash
bash scripts/tmux.sh -s exp1 -o outputs1
```

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --output-dir outputs1
```

For the second experiment, start a second tmux session named `exp2` with outputs directory `outputs2`. In addition, specify a new server port for the inference engine and orchestrator (*will use the last 2 GPUs*)

```bash
bash scripts/tmux.sh -s exp-2 -o outputs2
```

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --inference-gpu-ids 2 \
  --trainer-gpu-ids 3 \
  --inference.server.port 8001 \
  --orchestrator.client.base-url http://localhost:8001/v1 \
  --output-dir outputs2
```

### Multi-Node Training

> We currently require shared file system for multi-node RL training.

To faciliate multi-node RL training, ensure that all nodes have access to a shared file system and that the node that will run the inference server is reachable from the orchestrator via a private or public IP address. Then, set the following environment variables on all nodes:

```bash
# On all nodes
export OUTPUT_DIR=...               # Path to directory in shared file system
export INFERENCE_SERVER_IP=...      # Reachable IP address of the inference node
export INFERENCE_SERVER_API_KEY=... # API key for the inference server
```

Then, start the inference server on one node.

```bash
# On one node
uv run inference ... \
    --api-key $INFERENCE_SERVER_API_KEY --parallel ...
```

Then, start a single orchestrator

```bash
# On either node
uv run orchestrator ... \
    --client.base-url http://$INFERENCE_SERVER_IP:8000/v1 \
    --client.api-key-var INFERENCE_SERVER_API_KEY \
    --output-dir $OUTPUT_DIR
```

Finally, start the trainer on one as described in the [Trainer](#trainer) section.

```bash
# On other node
uv run torchrun \
    --nproc-per-node 8 \
    --local-rank-filter 0 \
    src/prime_rl/trainer/rl/train.py ... \
    --output-dir $OUTPUT_DIR
```

Of course, you can further scale up the number of nodes used by the trainer and inference server, as described in the sections above. However, make sure that there is only a single orchestrator instance.

### SLURM

TBD.

## Kubernetes

For production deployments on Kubernetes clusters, PRIME-RL provides a Helm chart that manages the entire training infrastructure. Kubernetes automatically handles pod scheduling, restarts, and resource allocation.

### Prerequisites

- Kubernetes cluster with GPU-enabled nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- [Helm 3.x](https://helm.sh/docs/intro/install/) installed
- Storage class supporting `ReadWriteMany` (NFS, CephFS, or cloud provider storage)

### Quick Start

Deploy the complete training infrastructure with auto-start enabled:

```bash
cd k8s
# IMPORTANT: Use release name "reverse-text" (hardcoded in example values)
helm install reverse-text ./prime-rl -f ./prime-rl/examples/reverse-text.yaml

# Verify deployment
kubectl get pods -l app.kubernetes.io/instance=reverse-text -w
# Press Ctrl+C when all pods show 1/1 Running

# Monitor training logs
kubectl logs -f reverse-text-trainer-0
kubectl logs -f reverse-text-inference-0
kubectl logs -f reverse-text-orchestrator-0
```

The reverse-text example is pre-configured with `autoStart: true`, so all components automatically start when pods boot. For manual control, set `autoStart: false` in your values file and exec into pods to run commands manually.

### Components

The Helm chart deploys three components (all using StatefulSets):

1. **Orchestrator** (StatefulSet, 1 replica)
   - Named: `<release-name>-orchestrator-0` (e.g., `my-exp-orchestrator-0`)
   - Coordinates training workflow
   - No GPU required
   - Communicates with trainer and inference via K8s services

2. **Inference** (StatefulSet, scalable)
   - Runs vLLM inference server
   - Pods named: `<release-name>-inference-0`, `<release-name>-inference-1`, ...
   - Configurable GPU count
   - Service exposed on port 8000

3. **Trainer** (StatefulSet, scalable)
   - Runs SFT or RL training
   - Pods named: `<release-name>-trainer-0`, `<release-name>-trainer-1`, ...
   - Configurable GPU count
   - Reads/writes checkpoints to shared storage

**StatefulSets provide:**
- Clean, predictable pod names with ordinal indices (0, 1, 2, ...)
- Stable DNS hostnames via headless services
- Environment variables for pod discovery: `$POD_NAME`, `$STATEFUL_REPLICAS`, `$HEADLESS_SERVICE`
- Required for distributed training coordination

### Shared Storage

All components mount a shared PVC at `/data` for:
- Model checkpoints (trainer writes, inference reads)
- Training data
- Experiment outputs

This **shared storage is required** for weight updates between trainer and inference.

### Scaling

Scale components by adjusting replicas in values:

```yaml
# custom-scale.yaml
orchestrator:
  replicas: 1  # Always keep at 1

trainer:
  replicas: 4  # Scale trainers for distributed training
  gpu:
    count: 2   # GPUs per trainer pod

inference:
  replicas: 2  # Load balance inference
  gpu:
    count: 4   # GPUs per inference pod
```

Deploy with custom scaling:

```bash
helm upgrade my-training ./k8s/prime-rl -f custom-scale.yaml
```

### Multi-Node Training on K8s

For distributed training across multiple nodes:

```yaml
# multi-node.yaml
trainer:
  replicas: 8  # 8 trainer pods distributed across nodes

  # Ensure pods spread across nodes
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: role
          operator: In
          values:
          - trainer
      topologyKey: kubernetes.io/hostname
```

The orchestrator coordinates all trainers via the shared storage and NCCL communication (port 29501).

### Auto-Start Configuration

By default, pods start with `sleep infinity` for manual control. To auto-start workloads, configure the `autoStart` and `command` fields:

```yaml
# example-values.yaml
orchestrator:
  autoStart: true
  command: "uv run orchestrator @ /app/examples/reverse_text/rl/orch.toml --output-dir /data/outputs --client.base-url '[\"http://my-exp-inference.default.svc.cluster.local:8000/v1\"]'"

inference:
  autoStart: true
  command: "uv run inference @ /app/examples/reverse_text/rl/infer.toml"

trainer:
  autoStart: true
  command: "uv run trainer @ /app/examples/reverse_text/rl/train.toml --output-dir /data/outputs"
```

The reverse-text example (`k8s/prime-rl/examples/reverse-text.yaml`) is pre-configured with auto-start enabled.

### Configuration

Customize resources, storage, secrets, and more:

```bash
# Override specific values
helm install my-training ./k8s/prime-rl \
  --set image.tag=v1.0.0 \
  --set trainer.gpu.count=8 \
  --set storage.size=2Ti

# Or use a custom values file
helm install my-training ./k8s/prime-rl -f my-values.yaml
```

### Monitoring

View logs and exec into pods:

```bash
# View logs
kubectl logs my-exp-trainer-0
kubectl logs -f my-exp-inference-0  # Follow logs

# Interactive shell
kubectl exec -it my-exp-trainer-0 -- bash

# List all pods for this experiment
kubectl get pods -l app.kubernetes.io/instance=my-exp
```

### Networking

The chart exposes:
- **Port 8000** - API endpoint for inference and orchestrator
- **Port 29501** - NCCL broadcast for distributed coordination

Services use `ClusterIP` by default (internal only). For external access, change to `LoadBalancer`:

```yaml
inference:
  service:
    type: LoadBalancer
```

### Advanced: Custom Images

Use your own Docker image:

```bash
helm install my-training ./k8s/prime-rl \
  --set image.repository=myregistry/prime-rl \
  --set image.tag=custom-v1
```

For a complete guide, see the [Kubernetes README](../k8s/README.md).