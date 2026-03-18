# Kubernetes

This guide covers deploying PRIME-RL training infrastructure on Kubernetes clusters using the provided Helm chart.

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
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml

# Or with defaults (no example-specific config)
helm install my-exp ./k8s/prime-rl --set trainer.replicas=3 --set inference.replicas=2
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
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml
```

- Model: Qwen3-0.6B
- GPUs: 1 per component
- Runs on consumer GPUs (RTX 3090/4090)
- **Note:** You can use any release name - the chart automatically configures service URLs

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
helm install my-release ./k8s/prime-rl -f custom-values.yaml
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
helm install my-release ./k8s/prime-rl \
  --set config.secrets.enabled=true \
  --set config.secrets.name=prime-rl-secrets
```

## Common Operations

### Deploy a new experiment

```bash
# With example config
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml

# With custom settings
helm install my-exp ./k8s/prime-rl --set trainer.replicas=10 --set inference.replicas=5
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

## Environment Variables

Each pod has these K8s environment variables set:

- `$POD_NAME` - Full pod name (e.g., `my-exp-trainer-3`)
- `$POD_IP` - Pod IP address
- `$STATEFUL_REPLICAS` - Total number of replicas for that component
- `$HEADLESS_SERVICE` - DNS name for peer discovery (e.g., `my-exp-trainer-headless.default.svc.cluster.local`)
- `$INFERENCE_URL` - Full URL to the first inference pod (available in orchestrator and trainer pods)

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
  src/prime_rl/trainer/sft/train.py @ configs/train.toml
```

## Troubleshooting

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

### Pod stuck in Pending

Check if GPU resources are available:

```bash
kubectl describe pod my-exp-trainer-0
```

Look for events like `Insufficient nvidia.com/gpu`.

### Inference server not responding

Check if the inference pod is ready:

```bash
kubectl get pods -l role=inference
kubectl logs my-exp-inference-0
```

## Uninstalling

```bash
# Remove the Helm release
helm uninstall my-exp

# Delete PVC (data will be lost!)
kubectl delete pvc prime-rl-shared-data
```
