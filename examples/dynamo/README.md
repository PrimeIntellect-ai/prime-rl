# Dynamo native-gRPC integration

Prime-RL uses Dynamo's OpenAI frontend for generation and `/v1/rl/workers` to
discover the direct vLLM admin endpoints used for weight updates.

## Pinned researcher stack

The currently validated source set is:

- vLLM `biswapanda/vllm@e74fc3f`
- Dynamo `ai-dynamo/dynamo@fc556d9`
- Prime-RL changes ported from Biswa's combined integration PR #3181 onto current `main`

These features are not all present in the public vLLM 0.26 and Dynamo 1.3.0
wheels. Use the build/install scripts in [`scripts/`](scripts/) or an internally
published image containing those exact revisions. The scripts use seven-character
revision pins as the repository policy requires.

For a two-GPU researcher smoke test, follow [`local/README.md`](local/README.md).
The other recipes describe larger externally deployed topologies.

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

## Recipes

- [`local`](local): single-node 2-GPU smoke test with a real Dynamo stack
  (etcd + frontend + sidecar + vLLM engine).
- [`slurm-managed`](slurm-managed): launcher-managed aggregated Dynamo deployment on dedicated inference nodes.
- [`qwen3_06b_math`](qwen3_06b_math): single-GPU trainer and aggregate Dynamo
  inference smoke test.
- [`qwen3_30b_Thinking`](qwen3_30b_Thinking): Qwen3-30B Thinking math with an
  external prefill/decode deployment.
- [`glm52_fp8_r2e`](glm52_fp8_r2e): multi-node GLM-5.2 FP8 R2E training with a
  separately managed DGD.
