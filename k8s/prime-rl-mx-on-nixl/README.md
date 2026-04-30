# prime-rl-mx-on-nixl — GB200 Benchmark Manifests

Kubernetes manifests for running PI's NIXL weight-transfer PR #2326 on
GB200 (GKE, `kavin` namespace) with an optional ModelExpress rendezvous
overlay.

## Scenarios

| Scenario | Config | What it demonstrates |
|----------|--------|----------------------|
| A | no MX env vars | PI's code as-is — validates the environment |
| B | `PRIME_RL_MX_RENDEZVOUS` set | MX-mediated SPG host/port discovery |
| C | B + `PRIME_RL_MX_PIPELINE_REPLICATION=1` | Rollout-as-source for future pulls |

## Cluster Requirements

- GKE cluster with `customer-gpu-o7v` node pool (GB200, 4 GPU/node, ARM64).
- `kavin` namespace with:
  - `modelexpress-server.kavin.svc.cluster.local:8001` running.
  - `shared-model-cache` PVC bound.
  - `kavin-compute-domain-channel` ResourceClaimTemplate.
  - `hf-token-secret` (for gated models).
  - `nvcr-imagepullsecret` (for the overlay image).

## Image

```
nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.1
```

Build from the repo root:

```bash
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.mx-on-nixl \
  -t nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.1 \
  --push .
```

## Quick Start

```bash
# Scenario A (baseline)
./run.sh deploy A

# Scenario B (MX rendezvous)
./run.sh deploy B

# Scenario C (MX + pipeline replication)
./run.sh deploy C

# Watch logs
./run.sh logs

# Check status
./run.sh status

# Tear down
./run.sh clean
```

## Topology

```
┌───────────────────── customer-gpu-o7v (node 1) ─────────────────────┐
│  prime-rl-mx-on-nixl-trainer-0                                      │
│  ├─ 4× GB200                                                        │
│  ├─ FSDP2 across 4 ranks                                            │
│  └─ NIXLWeightBroadcast (PI) [+ MX rendezvous overlay in B/C]       │
└─────────────────────────────────────────────────────────────────────┘
                                │  NIXL RDMA (RoCE rc_mlx5)
                                ▼
┌───────────────────── customer-gpu-o7v (node 2) ─────────────────────┐
│  prime-rl-mx-on-nixl-inference-0                                    │
│  ├─ 4× GB200                                                        │
│  └─ NIXLWeightUpdateWorker (PI) [+ MX pipeline-replication in C]    │
└─────────────────────────────────────────────────────────────────────┘

prime-rl-mx-on-nixl-orchestrator (any o7v node, CPU-only)
  └─ HTTP coordinator — /pause, /resume, /init_nixl_transfer
```

`podAntiAffinity` ensures trainer and inference land on different nodes
so NIXL actually crosses the RoCE fabric.

## Known-Good Env Vars

The pod specs inherit our GB200 POC's known-good UCX tuning (`UCX_TLS`,
`UCX_IB_GID_INDEX`, `OMPI_MCA_pml`). Do not deviate without verifying
RDMA still negotiates `rc_mlx5` transport in the UCX startup log.
