# temp-experimental: K8s e2e for vLLM 0.23 P/D + mooncake + llm-d

Scratch folder for the Kubernetes/Helm path for [PR #2883](https://github.com/PrimeIntellect-ai/prime-rl/pull/2883)
(GLM-5.2 16-node disaggregated prefill/decode with mooncake KV pool + llm-d router).

This is **not** a production chart. Treat it as a vertical-slice prototype: it
gets a request flowing through `client → Envoy → EPP → vLLM prefill → NIXL →
vLLM decode → response`, with mooncake as the shared CPU KV pool. Once the
topology is proven on a small cluster, the chart graduates into
`pi-rft/k8s/rft-stack` (where it gets RBAC, KEDA, graceful shutdown, dashboards,
PR-based GitOps deploys).

## Layout

```
temp-experimental/
├── docker/
│   └── Dockerfile.llmd          # FROM prime-rl:<tag> + NIXL-from-source + llm-d bins
├── helm/
│   ├── Chart.yaml
│   ├── values.yaml              # Defaults: 1P + 1D smoke topology
│   ├── values-glm5.2-16node.yaml  # Sketch of the PR's target topology
│   └── templates/
│       ├── _helpers.tpl
│       ├── pvc.yaml             # Shared RWX volume
│       ├── configmap.yaml       # inference.toml + mooncake config.json
│       ├── mooncake.yaml        # mooncake_master Deployment + Service
│       ├── inference.yaml       # prefill + decode StatefulSets (+ pd-sidecar on decode)
│       └── router.yaml          # EPP + Envoy Deployment + Service + ConfigMap
└── README.md
```

## End-to-end flow

1. **Image.** Build the base image from prime-rl's `Dockerfile.cuda` via the
   `build-push-prime-rl` workflow in `hosted-rl`. Then layer
   `temp-experimental/docker/Dockerfile.llmd` to add NIXL-from-source (the pip
   wheel segfaults on KV transfer per the PR's own ⚠️) and the llm-d Go bins.

   ```bash
   # base image is built by pi-rft's build-push-prime-rl workflow
   BASE=ghcr.io/primeintellect-ai/hosted-rl/prime-rl:<tag>-amd64

   docker build \
     --build-arg BASE_IMAGE=$BASE \
     -f temp-experimental/docker/Dockerfile.llmd \
     -t ghcr.io/primeintellect-ai/hosted-rl/prime-rl-llmd:<tag>-amd64 \
     .
   ```

2. **Cluster prerequisites.** Audit before installing:
   - GPU operator + `nvidia` runtime class
   - NFS or other RWX `StorageClass` (override `storage.storageClassName`)
   - `k8s-rdma-shared-dev-plugin` exposing `rdma/rdma_shared_device_a` (override
     `rdma.deviceName`)
   - PSA permissive enough for `privileged: true` + `IPC_LOCK` in the workload
     namespace
   - ghcr pull secret (default name `ghcr-secret`)

3. **Smoke install** (1 prefill pod + 1 decode pod, single GPU each):
   ```bash
   helm install glm52 ./temp-experimental/helm \
     --set image.tag=<your-tag>-amd64 \
     --set rdma.enabled=true
   kubectl get pods -l app.kubernetes.io/instance=glm52
   ```

4. **Hit the router** once everything is `Ready`:
   ```bash
   kubectl port-forward svc/glm52-router 8000:8000
   curl -s http://localhost:8000/v1/chat/completions \
     -H 'content-type: application/json' \
     -d '{"model":"zai-org/GLM-5.2-FP8","messages":[{"role":"user","content":"hi"}]}'
   ```

5. **Scale up to the PR target** with `-f values-glm5.2-16node.yaml` once the
   multi-node items below land.

## What this gets you today

- vLLM 0.23 prefill engine + vLLM 0.23 decode engine, each with the PR's
  per-role env overrides (`VLLM_ENABLE_MOE_DP_CHUNK`, `deepep_high_throughput`
  vs `deepep_low_latency`, decode-only CUDA graphs).
- Mooncake distributed CPU KV pool (one master `Deployment`, one
  `mooncake_client` sidecar per inference pod).
- llm-d router stack: `epp` + `envoy` in a single `Deployment`, decode-side
  `pd-sidecar` co-located in the decode pod. Endpoints are resolved from the
  prefill/decode headless services at startup.
- A `use_pd_kv_transfer = true` override in the rendered TOML so the worker
  config still builds the NIXL transfer connector even though the `[deployment]`
  section is dropped (K8s replaces it).

## What's missing (production / 16-node)

These are gaps between this scaffold and the actual PR target. Each is a
self-contained follow-up:

1. **Multi-DP per pod.** The SLURM template starts one vLLM API server per
   local DP rank (`PORT + k`). Today the chart hardcodes DP=1. Needs a `for k
   in 0..gpusPerNode-1` loop in the entrypoint plus per-rank ports and NIXL
   side-channel ports.
2. **Cross-node TP.** For the 16-node config there are 4 inner prefill
   replicas, each spanning 2 nodes (TP across nodes). This needs
   LeaderWorkerSet, not StatefulSet — same pattern `rft-stack/lws-inference.yaml`
   already uses.
3. **Per-replica routers.** The PR config has 4 P/D outer replicas; each gets
   its own EPP+Envoy. Today the chart deploys one router for the whole pool.
4. **NCCL / GLOO socket interface.** Needs `NCCL_SOCKET_IFNAME` /
   `GLOO_SOCKET_IFNAME` set from the pod's primary interface, like
   `rft-stack/lws-inference.yaml` does via `ip -o -4 addr`.
5. **UCX device discovery.** SLURM enumerates active RDMA ports from sysfs
   (`/sys/class/infiniband/*`). The chart pins decode to `mlx5_0:1` and
   otherwise leaves UCX on auto.
6. **Liveness/readiness/graceful drain.** No probes yet. EPP and pd-sidecar
   need readiness gating so the router doesn't dispatch into a half-up pool.
   Steal the `gracefulShutdown.preStop` from rft-stack once we know vLLM
   shutdown drains cleanly through the EPP.
7. **Observability.** EPP `--metrics-port 9090` and Envoy admin are exposed
   but no `PodMonitor` yet.
8. **Auth.** Router is open. EPP needs the same JWT posture vllm-router has
   in rft-stack.

## How this maps to the eventual rft-stack PR

This scaffold is intentionally separate from `pi-rft/k8s/rft-stack` so the
existing chart keeps shipping. The migration is roughly:

| Here | Lands as |
|---|---|
| `inference.yaml` (prefill+decode StatefulSets) | `rft-stack/templates/lws-inference-disagg.yaml` (new LWS group per role) |
| `router.yaml` (EPP+Envoy Deployment) | `rft-stack/templates/router.yaml` extension: `router.type: llm-d` branch |
| `mooncake.yaml` | new `rft-stack/templates/mooncake.yaml`, gated by `kvCacheOffload.mooncake.enabled` |
| `configmap.yaml` (TOML + mooncake config) | merged into existing rft-stack config rendering |
| `Dockerfile.llmd` | sibling `pi-rft/docker/prime-rl-llmd.Dockerfile` + a `build-push-prime-rl-llmd` workflow |

## References

- PR #2883 — vLLM 0.23 bump + GLM-5.2 16-node P/D config
- `configs/glm5.2_16node_llmd/inference.toml` — golden source of truth for the
  target topology
- `src/prime_rl/templates/inference.sbatch.j2` — the SLURM equivalent of this
  chart; consult when in doubt about role wiring
- `pi-rft/k8s/rft-stack/` — the production chart this will eventually merge
  into
