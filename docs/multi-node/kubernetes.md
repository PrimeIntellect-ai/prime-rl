# Kubernetes

This guide covers deploying PRIME-RL training infrastructure on Kubernetes clusters using the provided Helm chart.

## Prerequisites

- Kubernetes cluster with GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- [Helm 3.x](https://helm.sh/docs/intro/install/) installed
- Storage class that supports `ReadWriteMany` (e.g., NFS, CephFS, or cloud provider storage)

```bash
# Verify prerequisites
helm version
kubectl get pods -n gpu-operator
kubectl get storageclass
```

## Quick Start

### 1. Deploy

```bash
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml
```

### 2. Verify deployment

```bash
kubectl get pods -l app.kubernetes.io/instance=my-exp
```

### 3. Run training

```bash
kubectl exec -it my-exp-trainer-0 -- bash
cd /data
uv run trainer @ /app/examples/reverse_text/configs/train.toml
```

### 4. Monitor

```bash
kubectl logs -f my-exp-trainer-0
```

## Architecture

The chart deploys three main components (all using StatefulSets):

1. **Orchestrator** — always 1 replica, no GPU required
2. **Inference** — scalable replicas with GPU, runs vLLM
3. **Trainer** — scalable replicas with GPU

All components mount the same PVC at `/data` for model checkpoint sharing, training data, and experiment outputs.

## Configuration

### Storage

```yaml
storage:
  storageClassName: my-storage-class
  size: 500Gi
```

### GPU

```yaml
inference:
  gpu:
    count: 4
trainer:
  gpu:
    count: 2
```

### Resources

```yaml
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

```bash
kubectl create secret generic prime-rl-secrets \
  --from-literal=wandb-api-key=YOUR_WANDB_KEY \
  --from-literal=hf-token=YOUR_HF_TOKEN

helm install my-release ./k8s/prime-rl \
  --set config.secrets.enabled=true \
  --set config.secrets.name=prime-rl-secrets
```

## Environment Variables

Each pod has these K8s environment variables:

- `$POD_NAME` — Full pod name
- `$POD_IP` — Pod IP address
- `$STATEFUL_REPLICAS` — Total number of replicas for that component
- `$HEADLESS_SERVICE` — DNS name for peer discovery
- `$INFERENCE_URL` — Full URL to the first inference pod

For distributed training, extract the rank from the pod name:

```bash
RANK=$(echo $POD_NAME | grep -o '[0-9]*$')

torchrun \
  --nnodes=$STATEFUL_REPLICAS \
  --node-rank=$RANK \
  --nproc-per-node=8 \
  --rdzv-endpoint=my-exp-trainer-0.$HEADLESS_SERVICE:29501 \
  src/prime_rl/trainer/sft/train.py @ configs/train.toml
```

## Common Operations

```bash
# Deploy
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml

# Exec into pods
kubectl exec -it my-exp-trainer-0 -- bash

# View logs
kubectl logs -f my-exp-trainer-0

# List pods
kubectl get pods -l app.kubernetes.io/instance=my-exp

# Uninstall
helm uninstall my-exp
kubectl delete pvc prime-rl-shared-data
```

## Troubleshooting

**PVC not bound**: `kubectl get pvc prime-rl-shared-data` — STATUS should be "Bound"

**Pod stuck in Pending**: `kubectl describe pod my-exp-trainer-0` — look for `Insufficient nvidia.com/gpu`

**Inference not responding**: `kubectl logs my-exp-inference-0`
