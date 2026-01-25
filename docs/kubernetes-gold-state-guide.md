# Deploying Prime-RL on Kubernetes: Gold State Environment Guide

## Overview

This guide covers creating a gold state environment where the problem given to an AI agent is to deploy prime-rl on Kubernetes. The behavioral tests verify that prime-rl is correctly deployed and running.

**The Problem Statement:**
> "Deploy prime-rl on a Kubernetes cluster to run the reverse-text training example. Verify that all components are running and training is progressing."

**The Gold State:** Prime-rl already has working Helm charts. The agent needs to understand them, configure them correctly for AWS, and verify the deployment works.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Kubernetes Resources Created](#kubernetes-resources-created)
3. [Key Concepts](#key-concepts)
4. [AWS EKS Cost Breakdown](#aws-eks-cost-breakdown)
5. [Helm Chart Structure](#helm-chart-structure)
6. [AWS-Specific Prerequisites](#aws-specific-prerequisites)
7. [Behavioral Tests](#behavioral-tests)
8. [Problem Difficulty Levels](#problem-difficulty-levels)
9. [Environment Structure](#environment-structure)

---

## Architecture Overview

Prime-rl has three components that run on Kubernetes:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AWS EKS CLUSTER                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Kubernetes Control Plane                             │   │
│  │                     (Managed by AWS - $0.10/hr)                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Worker Nodes                                      │   │
│  │                                                                          │   │
│  │   ┌─────────────────┐                                                   │   │
│  │   │  CPU Node       │  ← Orchestrator runs here (no GPU needed)         │   │
│  │   │  (t3.medium)    │                                                   │   │
│  │   └─────────────────┘                                                   │   │
│  │                                                                          │   │
│  │   ┌─────────────────┐  ┌─────────────────┐                              │   │
│  │   │  GPU Node       │  │  GPU Node       │  ← Inference + Trainer       │   │
│  │   │  (g5.xlarge)    │  │  (g5.xlarge)    │    run here                  │   │
│  │   │  1× A10G 24GB   │  │  1× A10G 24GB   │                              │   │
│  │   └─────────────────┘  └─────────────────┘                              │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Shared Storage (EFS)                                 │   │
│  │                     ReadWriteMany - All pods access                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Communication

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   INFERENCE     │     │  ORCHESTRATOR   │     │    TRAINER      │
│   SERVER        │     │                 │     │                 │
│                 │     │                 │     │                 │
│  • Runs vLLM    │────▶│  • CPU-only!    │────▶│  • GPU required │
│  • Generates    │     │  • Collects     │     │  • Updates      │
│    rollouts     │     │    rollouts     │     │    model        │
│  • GPU required │     │  • Computes     │     │    weights      │
│                 │◀────│    advantages   │◀────│                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘

        ZeroMQ / Filesystem transport between components
```

---

## Kubernetes Resources Created

When you deploy with Helm, these resources are created:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KUBERNETES RESOURCES CREATED BY HELM                         │
│                                                                                 │
│  StatefulSets (3)                    Services (6)                              │
│  ───────────────                     ────────────                              │
│  ┌─────────────────────┐             ┌─────────────────────┐                   │
│  │ my-exp-orchestrator │             │ my-exp-orchestrator │ ClusterIP        │
│  │   └─ orchestrator-0 │◀───────────▶│   (port 8000)       │                   │
│  └─────────────────────┘             │                     │                   │
│                                      │ my-exp-orchestrator │ Headless         │
│  ┌─────────────────────┐             │   -headless         │                   │
│  │ my-exp-inference    │             └─────────────────────┘                   │
│  │   └─ inference-0    │◀───────────▶┌─────────────────────┐                   │
│  └─────────────────────┘             │ my-exp-inference    │ ClusterIP        │
│                                      │   (port 8000)       │                   │
│  ┌─────────────────────┐             │                     │                   │
│  │ my-exp-trainer      │             │ my-exp-inference    │ Headless         │
│  │   └─ trainer-0      │◀───────────▶│   -headless         │                   │
│  └─────────────────────┘             └─────────────────────┘                   │
│                                      ┌─────────────────────┐                   │
│                                      │ my-exp-trainer      │ ClusterIP        │
│  PersistentVolumeClaim (1)           │   (port 8000)       │                   │
│  ─────────────────────────           │                     │                   │
│  ┌─────────────────────┐             │ my-exp-trainer      │ Headless         │
│  │ my-exp-shared-data  │             │   -headless         │                   │
│  │ 1TB, ReadWriteMany  │             └─────────────────────┘                   │
│  └─────────────────────┘                                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Why StatefulSets (Not Deployments)?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (NOT used by prime-rl)                            │
│                                                                                 │
│   Pod names are RANDOM:                                                         │
│   ┌─────────────────────────┐                                                  │
│   │ my-app-7d4b5f8c9-x2k9j │  ← Random suffix, changes on restart              │
│   │ my-app-7d4b5f8c9-m3n7p │  ← Can't predict which pod is which               │
│   │ my-app-7d4b5f8c9-q8w2e │                                                   │
│   └─────────────────────────┘                                                  │
│                                                                                 │
│   Problem: Distributed training needs to know "I am rank 2 of 3"               │
│            Random names make this impossible!                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STATEFULSET (Used by prime-rl)                               │
│                                                                                 │
│   Pod names are SEQUENTIAL:                                                     │
│   ┌─────────────────────────┐                                                  │
│   │ my-exp-trainer-0       │  ← Always "-0", stable across restarts            │
│   │ my-exp-trainer-1       │  ← Always "-1"                                    │
│   │ my-exp-trainer-2       │  ← Always "-2"                                    │
│   └─────────────────────────┘                                                  │
│                                                                                 │
│   RANK=$(echo $POD_NAME | grep -o '[0-9]*$')  # Extract: "2"                   │
│                                                                                 │
│   Now distributed training knows: "I am rank 2 of 3" ✓                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Headless Services (DNS for Pod Discovery)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REGULAR SERVICE (ClusterIP)                                  │
│                                                                                 │
│   Client ──▶ my-exp-trainer:8000 ──▶ Load Balancer ──▶ Random Pod              │
│                                                                                 │
│   DNS resolves to: 10.0.0.100 (service IP, load balanced)                      │
│                                                                                 │
│   Problem: Can't talk to a SPECIFIC pod                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HEADLESS SERVICE (clusterIP: None)                           │
│                                                                                 │
│   DNS: my-exp-trainer-0.my-exp-trainer-headless.default.svc.cluster.local      │
│         ───────────────  ───────────────────────  ───────  ─────────────       │
│              │                    │                  │          │              │
│           Pod name          Headless service     Namespace   K8s suffix        │
│                                                                                 │
│   Resolves to: 10.0.0.15 (direct pod IP!)                                      │
│                                                                                 │
│   Now trainer-0 can directly contact trainer-1, trainer-2, etc. ✓             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### ZeroMQ Communication

ZeroMQ (ZMQ) is a messaging library that lets processes communicate over TCP sockets.

**Pattern 1: PUSH/PULL (Training Batches)**
```
   Orchestrator                          Packer (in Trainer)
   ┌─────────────┐                      ┌─────────────┐
   │             │   PUSH ────────▶     │             │
   │  Produces   │   tcp://host:5555    │  Consumes   │
   │  batches    │   ────────────────▶  │  batches    │
   │             │   (one-way queue)    │             │
   └─────────────┘                      └─────────────┘
```

**Pattern 2: PUB/SUB (Micro Batches to GPU Workers)**
```
                                        ┌─────────────┐
                                        │ GPU Rank 0  │ SUB to topic "data_rank|0|"
                                        └──────▲──────┘
   Packer                                      │
   ┌─────────────┐                      ┌──────┴──────┐
   │             │   PUB ───────────▶   │ GPU Rank 1  │ SUB to topic "data_rank|1|"
   │  Broadcasts │   tcp://host:5556    └──────▲──────┘
   │  to topics  │                             │
   │             │                      ┌──────┴──────┐
   └─────────────┘                      │ GPU Rank 2  │ SUB to topic "data_rank|2|"
                                        └─────────────┘
```

**Code Example - ZMQ Sender** (from `src/prime_rl/transport/zmq.py`):
```python
class ZMQTrainingBatchSender(TrainingBatchSender):
    def __init__(self, output_dir: Path, transport: ZMQTransportConfig):
        self.context = zmq.Context.instance()

        # Create a PUSH socket (sends messages to a queue)
        self.socket: zmq.Socket = self.context.socket(zmq.PUSH)

        # High Water Mark = max messages to buffer before blocking
        self.socket.setsockopt(zmq.SNDHWM, transport.hwm)  # default: 10

        # Connect to the packer's address
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")

    def send(self, batch: TrainingBatch) -> None:
        payload = self.encoder.encode(batch)
        self.socket.send_multipart([self.sender_id, payload], copy=False)
```

**Key Difference: `bind()` vs `connect()`**
```python
# WRONG - both sides connecting (nobody listening!)
sender.socket.connect("tcp://localhost:5555")    # ❌
receiver.socket.connect("tcp://localhost:5555")  # ❌

# CORRECT - one binds (server), one connects (client)
receiver.socket.bind("tcp://localhost:5555")     # ✓ "I'm listening"
sender.socket.connect("tcp://localhost:5555")    # ✓ "I'm connecting"
```

---

## AWS EKS Cost Breakdown

### Cost Components

| Component | Cost/Hour | Notes |
|-----------|-----------|-------|
| EKS Control Plane | $0.10 | Fixed, always running |
| CPU Node (t3.medium) | $0.042 | For orchestrator |
| GPU Node (g5.xlarge) | $1.01 | 1× A10G, 24GB VRAM |
| GPU Node (g5.xlarge) | $1.01 | (need 2 for inference + trainer) |
| EFS Storage | ~$0.30/GB/mo | Shared storage for checkpoints |
| **TOTAL (On-Demand)** | **~$2.26/hr** | Minimum viable setup |
| **TOTAL (Spot)** | **~$0.80/hr** | 60-70% savings on GPU nodes |

### Configuration Options

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION OPTIONS                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OPTION A: Minimal (Testing)                          ~$2.26/hr                 │
│  ─────────────────────────────                                                  │
│  • 1× t3.medium (orchestrator)                                                  │
│  • 2× g5.xlarge (inference + trainer, separate nodes)                          │
│  • Good for: Verifying deployment works                                         │
│                                                                                 │
│  OPTION B: Single GPU Node                            ~$1.15/hr                 │
│  ─────────────────────────────                                                  │
│  • 1× g5.xlarge (all components on one node)                                   │
│  • Inference + Trainer share GPU (memory-utilization=0.5)                      │
│  • Good for: Cost-conscious testing                                            │
│                                                                                 │
│  OPTION C: Multi-GPU Training                         ~$6.20/hr                 │
│  ─────────────────────────────                                                  │
│  • 1× t3.medium (orchestrator)                                                 │
│  • 1× g5.12xlarge (4× A10G for trainer)                                        │
│  • 1× g5.xlarge (inference)                                                    │
│  • Good for: Faster training, larger models                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### GPU Memory Requirements

From `benchmarks/results/BENCHMARKS.md`:

| Model | Training Type | GPU Memory Needed |
|-------|---------------|-------------------|
| Qwen3-0.6B | RL Full | 12.5 GiB |
| Qwen3-0.6B | RL LoRA | 5.1 GiB |
| Qwen3-4B | RL LoRA | 23.3 GiB |
| Qwen3-4B | RL Full | 17.1 GiB × 8 GPUs |

---

## Helm Chart Structure

```
k8s/
├── README.md                    ← Quick start guide
└── prime-rl/                    ← The Helm chart
    ├── Chart.yaml               ← Chart metadata (name, version)
    ├── values.yaml              ← DEFAULT configuration
    ├── examples/
    │   └── reverse-text.yaml    ← Override values for reverse-text example
    └── templates/
        ├── _helpers.tpl         ← Template helper functions
        ├── deployment.yaml      ← StatefulSet definitions
        ├── service.yaml         ← Service definitions
        └── pvc.yaml             ← Storage claim
```

### Default Values (k8s/prime-rl/values.yaml)

```yaml
namespace: default

image:
  repository: primeintellect/prime-rl
  tag: main

storage:
  enabled: true
  storageClassName: nfs           # ← Must change for AWS (use efs-sc)
  size: 1Ti

orchestrator:
  enabled: true
  autoStart: false                # ← Pods start with "sleep infinity"
  command: ""

inference:
  enabled: true
  autoStart: false
  gpu:
    enabled: true
    count: 1

trainer:
  enabled: true
  autoStart: false
  gpu:
    enabled: true
    count: 1
```

### Example Override (k8s/prime-rl/examples/reverse-text.yaml)

```yaml
orchestrator:
  autoStart: true
  command: "uv run orchestrator @ /app/examples/reverse_text/rl/orch.toml ..."

inference:
  autoStart: true
  command: "uv run inference @ /app/examples/reverse_text/rl/infer.toml ..."

trainer:
  autoStart: true
  command: "uv run trainer @ /app/examples/reverse_text/rl/train.toml ..."
```

### Helm Template Syntax Explained

```
{{ .Release.Name }}
   └─ The name you pass to "helm install"
      helm install MY-EXP ./chart  →  {{ .Release.Name }} = "my-exp"

{{ .Values.trainer.replicas }}
   └─ Value from values.yaml: trainer.replicas

{{- if .Values.trainer.autoStart }}
   └─ Conditional: only include this block if autoStart is true
      The "-" removes leading whitespace

valueFrom:
  fieldRef:
    fieldPath: metadata.name
   └─ Get the pod's own name at runtime (Kubernetes downward API)
```

### Common Mistakes (Counterexamples)

```yaml
# ❌ WRONG: Hardcoded service name
env:
- name: INFERENCE_URL
  value: "http://prime-rl-inference:8000/v1"   # Won't work if release name changes!

# ✓ CORRECT: Dynamic service name
env:
- name: INFERENCE_URL
  value: "http://{{ .Release.Name }}-inference:8000/v1"


# ❌ WRONG: Using Deployment for distributed training
apiVersion: apps/v1
kind: Deployment                                # Random pod names!

# ✓ CORRECT: Using StatefulSet
apiVersion: apps/v1
kind: StatefulSet                               # Predictable pod names!


# ❌ WRONG: Missing GPU resource limits
containers:
- name: trainer
  resources: {}                                 # Kubernetes won't schedule GPU!

# ✓ CORRECT: Explicit GPU request
containers:
- name: trainer
  resources:
    limits:
      nvidia.com/gpu: 1
```

---

## AWS-Specific Prerequisites

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    AWS PREREQUISITES CHECKLIST                                  │
│                                                                                 │
│  1. EKS CLUSTER                                                                │
│     ├── Create cluster: eksctl create cluster --name prime-rl-cluster         │
│     ├── Enable GPU: Install NVIDIA GPU Operator                               │
│     └── Verify: kubectl get nodes                                              │
│                                                                                 │
│  2. GPU NODE GROUP                                                             │
│     ├── Instance type: g5.xlarge (or larger)                                   │
│     ├── AMI: Amazon Linux 2 with GPU support                                   │
│     └── Node count: 2 (one for inference, one for trainer)                     │
│                                                                                 │
│  3. EFS STORAGE (ReadWriteMany)                                                │
│     ├── Create EFS filesystem in same VPC as EKS                               │
│     ├── Install EFS CSI driver in cluster                                      │
│     ├── Create StorageClass named "efs-sc"                                     │
│     └── Verify: kubectl get storageclass efs-sc                                │
│                                                                                 │
│  4. NVIDIA GPU OPERATOR                                                        │
│     ├── Add Helm repo: helm repo add nvidia https://helm.ngc.nvidia.com/nvidia │
│     ├── Install: helm install gpu-operator nvidia/gpu-operator                 │
│     └── Verify: kubectl get pods -n gpu-operator                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### EFS Storage Class for AWS

The default `nfs` storage class won't exist on AWS. Create an EFS storage class:

```yaml
# efs-storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-0123456789abcdef0    # ← Your EFS filesystem ID
  directoryPerms: "700"
  gidRangeStart: "1000"
  gidRangeEnd: "2000"
  basePath: "/prime-rl"
```

Override in your values:

```yaml
# aws-values.yaml
storage:
  storageClassName: efs-sc
  size: 100Gi
```

### Deployment Commands

```bash
# Deploy with reverse-text example
helm install my-exp ./k8s/prime-rl \
  -f ./k8s/prime-rl/examples/reverse-text.yaml \
  --set storage.storageClassName=efs-sc

# Verify deployment
kubectl get pods -l app.kubernetes.io/instance=my-exp

# View logs
kubectl logs -f my-exp-trainer-0

# Exec into trainer
kubectl exec -it my-exp-trainer-0 -- bash
```

---

## Behavioral Tests

### Test Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL TEST STRUCTURE                                    │
│                                                                                 │
│  PHASE 1: Prerequisites                                                        │
│  ──────────────────────                                                        │
│  ├── Test: EKS cluster exists and is accessible                                │
│  ├── Test: GPU nodes are available                                             │
│  ├── Test: EFS storage class exists                                            │
│  └── Test: NVIDIA GPU Operator is running                                      │
│                                                                                 │
│  PHASE 2: Deployment                                                           │
│  ────────────────────                                                          │
│  ├── Test: Helm install succeeds                                               │
│  ├── Test: All 3 pods reach Running state                                      │
│  ├── Test: PVC is bound                                                        │
│  └── Test: Services are created                                                │
│                                                                                 │
│  PHASE 3: Component Health                                                     │
│  ─────────────────────────                                                     │
│  ├── Test: Inference server responds to health check                           │
│  ├── Test: Orchestrator can reach inference server                             │
│  └── Test: Trainer can access shared storage                                   │
│                                                                                 │
│  PHASE 4: End-to-End Training                                                  │
│  ────────────────────────────                                                  │
│  ├── Test: Training step completes without error                               │
│  ├── Test: Metrics are logged                                                  │
│  └── Test: Checkpoint is saved to shared storage                               │
│                                                                                 │
│  CLEANUP                                                                        │
│  ───────                                                                        │
│  └── helm uninstall, delete PVC                                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Example Test Implementation

```typescript
// environment/src/index.ts

const WORKSPACE_PATH = '/hyperfocal/env/workspace';

const tests: SimpleTest[] = [
  // Phase 1: Prerequisites
  {
    id: 'gpu-nodes-available',
    name: 'GPU Nodes Available',
    description: 'Verify GPU nodes exist in the cluster',
    run: async (log: Logger) => {
      const result = await executeWithExitCode(
        'kubectl get nodes -l nvidia.com/gpu.present=true -o name',
        { cwd: WORKSPACE_PATH }
      );

      if (!result.success || !result.output.includes('node/')) {
        return {
          success: false,
          error: 'No GPU nodes found. Install NVIDIA GPU Operator.'
        };
      }

      log.info(`✅ Found GPU nodes: ${result.output.trim()}`);
      return { success: true };
    }
  },

  // Phase 2: Deployment
  {
    id: 'helm-install-succeeds',
    name: 'Helm Install Succeeds',
    description: 'Deploy prime-rl using Helm',
    run: async (log: Logger) => {
      const result = await executeWithExitCode(
        `helm install test-exp ${WORKSPACE_PATH}/k8s/prime-rl \
          -f ${WORKSPACE_PATH}/k8s/prime-rl/examples/reverse-text.yaml \
          --set storage.storageClassName=efs-sc`,
        { cwd: WORKSPACE_PATH }
      );

      if (!result.success) {
        return { success: false, error: `Helm install failed: ${result.output}` };
      }

      log.info('✅ Helm install succeeded');
      return { success: true };
    }
  },

  {
    id: 'pods-running',
    name: 'All Pods Running',
    description: 'Wait for all pods to reach Running state',
    run: async (log: Logger) => {
      // Poll for up to 5 minutes
      for (let i = 0; i < 60; i++) {
        const result = await executeWithExitCode(
          'kubectl get pods -l app.kubernetes.io/instance=test-exp -o json',
          { cwd: WORKSPACE_PATH }
        );

        if (result.success) {
          const pods = JSON.parse(result.output);
          const allRunning = pods.items.every(
            (pod: any) => pod.status.phase === 'Running'
          );

          if (allRunning && pods.items.length === 3) {
            log.info('✅ All 3 pods are Running');
            return { success: true };
          }
        }

        await new Promise(r => setTimeout(r, 5000)); // Wait 5s
      }

      return { success: false, error: 'Pods did not reach Running state' };
    }
  },

  // Phase 3: Component Health
  {
    id: 'inference-responds',
    name: 'Inference Server Responds',
    description: 'Verify inference server is healthy',
    run: async (log: Logger) => {
      // Port-forward to inference service
      const portForward = spawn('kubectl', [
        'port-forward', 'svc/test-exp-inference', '8080:8000'
      ]);

      await new Promise(r => setTimeout(r, 3000)); // Wait for port-forward

      try {
        const response = await fetch('http://localhost:8080/health');
        if (response.status === 200) {
          log.info('✅ Inference server is healthy');
          return { success: true };
        }
        return { success: false, error: `Health check returned ${response.status}` };
      } finally {
        portForward.kill();
      }
    }
  },

  // Phase 4: Training Verification
  {
    id: 'training-progresses',
    name: 'Training Makes Progress',
    description: 'Verify training step completes',
    run: async (log: Logger) => {
      // Check trainer logs for training progress
      for (let i = 0; i < 60; i++) {
        const result = await executeWithExitCode(
          'kubectl logs test-exp-trainer-0 --tail=50',
          { cwd: WORKSPACE_PATH }
        );

        if (result.success && result.output.includes('step')) {
          log.info('✅ Training is progressing');
          return { success: true };
        }

        await new Promise(r => setTimeout(r, 10000)); // Wait 10s
      }

      return { success: false, error: 'No training progress detected in logs' };
    }
  }
];
```

---

## Problem Difficulty Levels

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PROBLEM DIFFICULTY LEVELS                                    │
│                                                                                 │
│  EASY: Everything pre-configured                                               │
│  ─────────────────────────────────                                             │
│  • EKS cluster exists with GPU nodes                                           │
│  • EFS storage class already created                                           │
│  • GPU Operator installed                                                      │
│  • Agent just needs to run: helm install                                       │
│                                                                                 │
│  MEDIUM: Some setup required                                                   │
│  ───────────────────────────                                                   │
│  • EKS cluster exists with GPU nodes                                           │
│  • Agent needs to create EFS storage class                                     │
│  • Agent needs to modify values.yaml for AWS                                   │
│  • Agent needs to install GPU Operator                                         │
│                                                                                 │
│  HARD: Broken configuration                                                    │
│  ───────────────────────────                                                   │
│  • Helm chart has bugs (wrong port, missing env var)                           │
│  • Agent needs to debug and fix                                                │
│  • Agent needs to understand the architecture                                  │
│                                                                                 │
│  EXPERT: Design from scratch                                                   │
│  ───────────────────────────                                                   │
│  • No Helm chart provided                                                      │
│  • Agent must create Kubernetes manifests                                      │
│  • Agent must understand all components                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Environment Structure

```
/hyperfocal/env/
├── workspace/                      # The prime-rl codebase (gold state)
│   ├── k8s/                        # Helm charts (working)
│   ├── src/                        # Source code
│   └── ...
│
├── environment/
│   ├── src/
│   │   └── index.ts                # Your behavioral tests
│   ├── .env                        # AWS credentials
│   │   ├── AWS_ACCESS_KEY_ID
│   │   ├── AWS_SECRET_ACCESS_KEY
│   │   └── AWS_REGION
│   └── problems.yaml               # Problem prompts for different difficulties
│
└── packages/
    └── env-base/                   # Shared utilities
```

---

## Quick Reference Commands

```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name prime-rl-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type g5.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3

# Install NVIDIA GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace

# Install EFS CSI Driver
helm repo add aws-efs-csi-driver https://kubernetes-sigs.github.io/aws-efs-csi-driver/
helm install aws-efs-csi-driver aws-efs-csi-driver/aws-efs-csi-driver \
  --namespace kube-system

# Deploy prime-rl
helm install my-exp ./k8s/prime-rl \
  -f ./k8s/prime-rl/examples/reverse-text.yaml \
  --set storage.storageClassName=efs-sc

# Check status
kubectl get pods -l app.kubernetes.io/instance=my-exp
kubectl logs -f my-exp-trainer-0

# Cleanup
helm uninstall my-exp
kubectl delete pvc my-exp-shared-data
```

---

## References

- [AWS EKS Pricing](https://aws.amazon.com/eks/pricing/)
- [AWS EC2 GPU Instances Guide](https://www.nops.io/blog/amazon-ec2-gpu-instances-the-complete-guide/)
- [EKS GPU Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [AWS EFS CSI Driver](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html)
