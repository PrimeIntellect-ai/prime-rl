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

### Generated DynamoGraph deployments

`dynamo-dgd` produces internally consistent Helm values for Dynamo disaggregated inference. The source SHAs, image tag, and image digest are caller-supplied evidence: the renderer checks that they agree, but it does not authenticate source or image provenance. Verify those inputs through the release process before rendering and protect the generated values from modification.

Chart-managed mode requires explicit runnable controller commands; the renderer supplies safe execution defaults of one replica and `autoStart: true` and refuses to emit a sleeper deployment:

```bash
uv run dynamo-dgd inference.toml \
  --release-name my-exp \
  --namespace my-namespace \
  --image "$IMAGE@$IMAGE_DIGEST" \
  --image-digest "$IMAGE_DIGEST" \
  --prime-sha "$PRIME_SHA" \
  --dynamo-sha "$DYNAMO_SHA" \
  --output-dir ./artifacts \
  --gpu-architecture arm64 \
  --gpu-product NVIDIA-GB200 \
  --gpu-node-pool prime-gpu \
  --orchestrator-command 'uv run orchestrator @ /app/configs/debug/orch.toml --output-dir /data/outputs' \
  --trainer-command 'uv run trainer @ /app/configs/debug/rl/train.toml --output-dir /data/outputs'

helm install my-exp ./prime-rl \
  --namespace my-namespace \
  -f ./artifacts/dynamo-helm-values.json
```

Use `--external-controller` when another system runs orchestration and training; generated values then render no controller StatefulSets. Generated releases are limited to 41 characters so every derived Kubernetes Service name remains valid. The image tag must contain the first 12 characters of both caller-supplied source SHAs, and the digest must match `--image-digest`.

## Uninstalling

```bash
helm uninstall my-exp
kubectl delete pvc prime-rl-shared-data  # Warning: deletes data!
```

## Learn More

- [Full Kubernetes documentation](https://docs.primeintellect.ai/prime-rl/kubernetes) - Architecture, configuration, distributed training
- [Deployment guide](https://docs.primeintellect.ai/prime-rl/deployment) - Non-Kubernetes deployments
- [Troubleshooting](https://docs.primeintellect.ai/prime-rl/troubleshooting) - Common issues
