# Kubernetes Deployment with Helm

This directory contains a Helm chart for deploying PRIME-RL training infrastructure on Kubernetes clusters.

For complete documentation, see the [Kubernetes guide](https://docs.primeintellect.ai/prime-rl/kubernetes).

## Quick Start

```bash
# Deploy with the reverse-text example
helm install my-exp ./prime-rl -f ./prime-rl/examples/reverse-text.yaml

# Verify deployment
kubectl get pods -l app.kubernetes.io/instance=my-exp

# Exec into trainer and run training
kubectl exec -it my-exp-trainer-0 -- bash
cd /data && uv run trainer @ /app/examples/reverse_text/configs/train.toml
```

## Prerequisites

- Kubernetes cluster with GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- [Helm 3.x](https://helm.sh/docs/intro/install/) installed
- Storage class that supports `ReadWriteMany` (e.g., NFS, CephFS)

## Chart Structure

```
prime-rl/
├── Chart.yaml
├── values.yaml           # Default configuration
├── examples/
│   └── reverse-text.yaml # Example values for reverse-text
└── templates/
    ├── deployment.yaml   # StatefulSets for orchestrator, inference, trainer
    ├── service.yaml      # Headless services for pod discovery
    └── pvc.yaml          # Shared storage
```

## Configuration

See [values.yaml](./prime-rl/values.yaml) for all available options. Common overrides:

```bash
# Custom GPU allocation
helm install my-exp ./prime-rl \
  --set inference.gpu.count=4 \
  --set trainer.gpu.count=2

# With secrets for W&B/HF
kubectl create secret generic prime-rl-secrets \
  --from-literal=wandb-api-key=YOUR_KEY \
  --from-literal=hf-token=YOUR_TOKEN

helm install my-exp ./prime-rl \
  --set config.secrets.enabled=true \
  --set config.secrets.name=prime-rl-secrets
```

## Uninstalling

```bash
helm uninstall my-exp
kubectl delete pvc prime-rl-shared-data  # Warning: deletes data!
```

## Learn More

- [Full Kubernetes documentation](https://docs.primeintellect.ai/prime-rl/kubernetes) - Architecture, configuration, distributed training
- [Deployment guide](https://docs.primeintellect.ai/prime-rl/deployment) - Non-Kubernetes deployments
- [Troubleshooting](https://docs.primeintellect.ai/prime-rl/troubleshooting) - Common issues
