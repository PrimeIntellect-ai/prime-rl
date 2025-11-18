# Kubernetes Deployment with Helm

This directory contains a Helm chart for deploying PRIME-RL training infrastructure on Kubernetes clusters.

## Prerequisites

- Kubernetes cluster with GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- [Helm 3.x](https://helm.sh/docs/intro/install/) installed
- Storage class that supports `ReadWriteMany` (e.g., NFS, CephFS, or cloud provider storage)

### Verify Prerequisites

```bash
# Check Helm installation
helm version

# Check GPU operator
kubectl get pods -n gpu-operator

# Check available storage classes
kubectl get storageclass
```

## Quick Start

### 1. Deploy

```bash
# Deploy with a release name
helm install my-exp ./prime-rl -f ./prime-rl/examples/reverse-text.yaml

# Or with defaults (no example-specific config)
helm install my-exp ./prime-rl --set trainer.replicas=3 --set inference.replicas=2
```

### 2. Verify deployment

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/instance=my-exp

# Should show 3 pods:
# my-exp-orchestrator-0
# my-exp-inference-0
# my-exp-trainer-0
```

### 3. Run training

```bash
# Exec into trainer
kubectl exec -it my-exp-trainer-0 -- bash

# Inside the pod, run training
cd /data
uv run trainer @ /app/examples/reverse_text/configs/train.toml
```

### 4. Monitor progress

```bash
# Get logs
kubectl logs my-exp-trainer-0

# Follow logs in real-time
kubectl logs -f my-exp-trainer-0
```

## Available Examples

The chart includes pre-configured values for each example:

### reverse-text (Small - 1 GPU)
```bash
helm install my-exp ./prime-rl -f ./prime-rl/examples/reverse-text.yaml
```
- Model: Qwen3-0.6B
- GPUs: 1 per component
- Runs on consumer GPUs (RTX 3090/4090)

### wordle (Medium - 2-4 GPUs)
```bash
helm install my-exp ./prime-rl -f ./prime-rl/examples/wordle.yaml
```
- Model: Qwen3-1.7B
- GPUs: 2 per component
- Recommended: H100 GPUs

### alphabet-sort (Medium - 1 GPU with LoRA)
```bash
helm install my-exp ./prime-rl -f ./prime-rl/examples/alphabet-sort.yaml
```
- Model: Qwen3-4B-Instruct
- GPUs: 1 per component (uses LoRA)
- Recommended: H100 GPU

## Configuration

### Storage Configuration

By default, the chart creates a 1TB PVC with NFS storage. To customize:

```yaml
# custom-values.yaml
storage:
  storageClassName: my-storage-class
  size: 500Gi
```

Deploy with custom storage:
```bash
helm install my-release ./prime-rl -f custom-values.yaml
```

### GPU Configuration

Adjust GPU count per component:

```yaml
# custom-gpu.yaml
inference:
  gpu:
    count: 4  # Use 4 GPUs for inference

trainer:
  gpu:
    count: 2  # Use 2 GPUs for training
```

### Resource Limits

Customize memory and CPU:

```yaml
# custom-resources.yaml
trainer:
  resources:
    requests:
      memory: "64Gi"
      cpu: "16"
    limits:
      memory: "128Gi"
      cpu: "32"
```

### Secrets (Optional)

For W&B and HuggingFace authentication:

```bash
# Create secret
kubectl create secret generic prime-rl-secrets \
  --from-literal=wandb-api-key=YOUR_WANDB_KEY \
  --from-literal=hf-token=YOUR_HF_TOKEN

# Enable in values
helm install my-release ./prime-rl \
  --set config.secrets.enabled=true \
  --set config.secrets.name=prime-rl-secrets
```

## Common Operations

### Deploy a new experiment

```bash
# With example config
helm install my-exp ./prime-rl -f ./prime-rl/examples/reverse-text.yaml

# With custom settings
helm install my-exp ./prime-rl --set trainer.replicas=10 --set inference.replicas=5
```

### Exec into pods

```bash
# Exec into trainer-0
kubectl exec -it my-exp-trainer-0 -- bash

# Exec into specific trainer pod
kubectl exec -it my-exp-trainer-3 -- bash

# Exec into inference
kubectl exec -it my-exp-inference-0 -- bash
```

### View logs

```bash
# Get logs from trainer-0
kubectl logs my-exp-trainer-0

# Follow logs in real-time
kubectl logs -f my-exp-trainer-2

# Get logs from all trainers
kubectl logs -l app.kubernetes.io/instance=my-exp,role=trainer
```

### List all pods

```bash
# List pods for specific experiment
kubectl get pods -l app.kubernetes.io/instance=my-exp

# List all prime-rl pods
kubectl get pods -l app=prime-rl
```

## Architecture

### Components

The chart deploys three main components (all using StatefulSets):

1. **Orchestrator** (StatefulSet) - Coordinates training workflow
   - Always 1 replica: `prime-rl-orchestrator-0`
   - No GPU required
   - Communicates with trainer and inference

2. **Inference** (StatefulSet) - Runs vLLM inference server
   - Scalable replicas with stable pod names: `prime-rl-inference-0`, `prime-rl-inference-1`, ...
   - Each pod gets predictable DNS: `prime-rl-inference-0.prime-rl-inference-headless.default.svc.cluster.local`
   - Requires GPU(s)
   - Serves model predictions

3. **Trainer** (StatefulSet) - Runs SFT or RL training
   - Scalable replicas with stable pod names: `prime-rl-trainer-0`, `prime-rl-trainer-1`, ...
   - Each pod gets predictable DNS: `prime-rl-trainer-0.prime-rl-trainer-headless.default.svc.cluster.local`
   - Requires GPU(s)
   - Updates model weights on shared storage

**Why StatefulSets for all components?**
- **Consistent naming**: All pods have predictable names (`orchestrator-0`, `trainer-0`, `trainer-1`, ...)
- **Stable networking**: Each pod gets its own DNS hostname via headless service
- **Required for distributed training**: PyTorch/vLLM need to discover peers by stable hostname
- **Clean naming**: No random pod suffixes, easier to identify and debug

### Shared Storage

All components mount the same PVC at `/data` for:
- Model checkpoint sharing
- Training data
- Experiment outputs

This is **required** for coordinating weight updates between trainer and inference.

## Advanced Usage

### Distributed Training with StatefulSets

For distributed training across multiple trainer pods:

```yaml
# distributed-training.yaml
orchestrator:
  replicas: 1  # Always 1

trainer:
  replicas: 10  # Scale to 10 trainer pods
  gpu:
    count: 8    # 8 GPUs per pod = 80 total GPUs
```

Deploy and run distributed training:

```bash
helm upgrade --install my-training ./prime-rl -f distributed-training.yaml

# Extract pod ordinal for torchrun
# Inside prime-rl-trainer-3, $POD_NAME="prime-rl-trainer-3"
# Extract ordinal: 3

# On each trainer pod, run torchrun with:
# --nnodes=$STATEFUL_REPLICAS (e.g., 10)
# --node-rank=<extracted from $POD_NAME> (e.g., 3)
# --rdzv-endpoint=prime-rl-trainer-0.prime-rl-trainer-headless:29501
```

**Example: Running distributed training**

Exec into any trainer pod and extract the rank:

```bash
# Exec into trainer-3
kubectl exec -it my-exp-trainer-3 -- bash

# Extract rank from pod name
RANK=$(echo $POD_NAME | grep -o '[0-9]*$')  # 3
echo "My rank: $RANK, Total trainers: $STATEFUL_REPLICAS"

# Run distributed training
torchrun \
  --nnodes=$STATEFUL_REPLICAS \
  --node-rank=$RANK \
  --nproc-per-node=8 \
  --rdzv-endpoint=my-exp-trainer-0.$HEADLESS_SERVICE:29501 \
  --rdzv-backend=c10d \
  src/prime_rl/trainer/sft/train.py @ /app/examples/reverse_text/sft/train.toml
```

### Multi-Node with Pod Anti-Affinity

Ensure pods spread across nodes:

```yaml
# multi-node.yaml
trainer:
  replicas: 10

  # Force pods to run on different nodes
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

### Custom Images

Use your own Docker image:

```bash
helm install my-release ./prime-rl \
  --set image.repository=myregistry/prime-rl \
  --set image.tag=custom-v1
```

### Disable Components

Run only specific components:

```yaml
# inference-only.yaml
orchestrator:
  enabled: false

trainer:
  enabled: false

inference:
  enabled: true
  gpu:
    count: 8  # Use all GPUs for inference
```

## Troubleshooting

### Pods stuck in Pending

Check GPU availability:
```bash
kubectl describe node <node-name> | grep nvidia.com/gpu
```

Check PVC status:
```bash
kubectl get pvc
kubectl describe pvc prime-rl-shared-data
```

### Out of Memory

Increase memory limits:
```bash
helm upgrade my-release ./prime-rl \
  --set trainer.resources.limits.memory=64Gi
```

### Can't access shared storage

Verify PVC is bound:
```bash
kubectl get pvc prime-rl-shared-data
# STATUS should be "Bound"
```

Check mount inside pod:
```bash
kubectl exec -it prime-rl-trainer-xxx -- df -h /data
```

## Uninstalling

```bash
# Remove the Helm release
helm uninstall my-exp

# Delete PVC (data will be lost!)
kubectl delete pvc prime-rl-shared-data
```

## Environment Variables

Each pod has these K8s environment variables set:

- `$POD_NAME` - Full pod name (e.g., `my-exp-trainer-3`)
- `$POD_IP` - Pod IP address
- `$STATEFUL_REPLICAS` - Total number of replicas for that component
- `$HEADLESS_SERVICE` - DNS name for peer discovery (e.g., `my-exp-trainer-headless.default.svc.cluster.local`)

For distributed training, extract the rank from the pod name:

```bash
# Extract ordinal from pod name
RANK=$(echo $POD_NAME | grep -o '[0-9]*$')  # e.g., "my-exp-trainer-3" -> "3"

# Use in torchrun
torchrun \
  --nnodes=$STATEFUL_REPLICAS \
  --node-rank=$RANK \
  --nproc-per-node=8 \
  --rdzv-endpoint=my-exp-trainer-0.$HEADLESS_SERVICE:29501 \
  src/prime_rl/trainer/sft/train.py @ /app/examples/reverse_text/sft/train.toml
```

## Next Steps

- Read the [deployment guide](../docs/deployment.md) for production best practices
- Check the [reverse_text example](../examples/reverse_text/README.md) for a complete walkthrough
- See [troubleshooting guide](../docs/troubleshooting.md) for common issues
