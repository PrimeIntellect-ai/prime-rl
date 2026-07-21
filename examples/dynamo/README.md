# Dynamo native-gRPC deployment requirements

The recipes in this directory use Dynamo for inference and Prime only for the
trainer/orchestrator. Start from Dynamo's `sidecar_agg.yaml` or
`sidecar_disagg.yaml`, then apply the following RL overlay. The stock manifests
are serving examples and do not enable Prime worker discovery or vLLM's admin
control routes by themselves.

## Frontend

Set these variables on the Dynamo frontend and expose both container ports:

```yaml
env:
  - name: DYN_ENABLE_RL
    value: "true"
  - name: DYN_RL_PORT
    value: "8001"
ports:
  - name: http
    containerPort: 8000
  - name: rl-discovery
    containerPort: 8001
```

The frontend Kubernetes Service must also map ports 8000 and 8001. Prime's
`base_url` targets 8000; `dynamo_discovery_url` targets 8001.

## Every vLLM engine and sidecar pair

The engine HTTP address published by discovery must be reachable from the
trainer, so bind vLLM to the pod network rather than loopback:

```text
vllm-rs serve <model> --host 0.0.0.0 --port 8000 --grpc-port 50051 -- \
  --worker-extension-cls prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker \
  <other Python EngineCore arguments>
```

Install the matching Prime source in the engine image so Python can import the
worker extension. Set this environment variable on the `vllm-engine`
container to expose `/pause`, `/resume`, and `/collective_rpc`:

```yaml
env:
  - name: VLLM_SERVER_DEV_MODE
    value: "1"
```

Set the following variables on each `dynamo-vllm-sidecar` container. `POD_IP`
must appear before `VLLM_HTTP_ENDPOINT` so Kubernetes expands it:

```yaml
env:
  - name: POD_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
  - name: VLLM_HTTP_ENDPOINT
    value: "http://$(POD_IP):8000"
  - name: DYN_ENABLE_RL
    value: "true"
```

Keep the existing `--grpc-endpoint 127.0.0.1:50051`: gRPC stays pod-local,
while discovery publishes the pod-reachable HTTP admin address. No per-worker
Kubernetes Service is required when trainer-to-pod networking is routable.

After startup, `/v1/rl/workers` must return every expected worker with a
non-null `admin_base_url`, positive `world_size`, and no `error` before Prime is
launched.
