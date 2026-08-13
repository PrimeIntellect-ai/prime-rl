# Local Dynamo smoke test

This recipe runs the Prime-RL trainer on GPU 1 and a Dynamo-managed vLLM engine
on GPU 0. It uses the exact source revisions from the validated integration:

| Component | Revision |
| --- | --- |
| Prime-RL integration source | Biswa's combined PR #3181, ported onto this branch's `main` base |
| vLLM | `biswapanda/vllm@e74fc3f` |
| Dynamo | `ai-dynamo/dynamo@fc556d9` |

The public vLLM 0.26 and Dynamo 1.3.0 wheels do not contain this complete
native-gRPC/worker-discovery stack. Build the artifacts below; do not replace
them with `uv sync --extra dynamo`.

## Build and install the pinned artifacts

Building vLLM requires a CUDA development environment. Building Dynamo requires
Rust, CMake, Clang, Protobuf, and the system packages listed in Dynamo's source
build guide.

```bash
examples/dynamo/scripts/build_vllm_wheel.sh
examples/dynamo/scripts/build_dynamo_artifacts.sh
examples/dynamo/scripts/install_artifacts.sh
```

The scripts verify the seven-character revisions before building. They place the wheels and `dynamo-vllm-sidecar` executable under
`dist/dynamo/`. The installer creates a separate `.venv-dynamo` because the
custom inference stack's Torch and Pydantic constraints differ from Prime-RL's
trainer environment.

## Start the stack

Run each process in its own terminal. Stop old instances before retrying.

```bash
# 1. etcd
etcd --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  --listen-peer-urls http://127.0.0.1:2380
```

```bash
# 2. Custom vLLM engine on GPU 0
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
  PYTHONPATH="$(pwd)/src" .venv-dynamo/bin/vllm-rs serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8002 --grpc-port 50051 \
  --python "$(pwd)/.venv-dynamo/bin/python" -- \
  --worker-extension-cls prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker
```

```bash
# 3. Dynamo frontend: generation on 8000, RL discovery on 8001
DYN_ENABLE_RL=true DYN_RL_PORT=8001 \
DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE=true \
  .venv-dynamo/bin/python -m dynamo.frontend \
  --http-host 0.0.0.0 --http-port 8000 \
  --namespace dynamo --discovery-backend etcd \
  --request-plane tcp --event-plane zmq --router-min-initial-workers 1
```

```bash
# 4. Dynamo sidecar: generation uses gRPC 50051; Prime control uses HTTP 8002
DYN_NAMESPACE=dynamo DYN_DISCOVERY_BACKEND=etcd DYN_ENABLE_RL=true \
VLLM_HTTP_ENDPOINT=http://127.0.0.1:8002 \
  dist/dynamo/dynamo-vllm-sidecar \
  --vllm-endpoint 127.0.0.1:50051 \
  --admin-endpoint http://127.0.0.1:8002 \
  --model-path Qwen/Qwen3-0.6B \
  --namespace dynamo \
  --rl-discovery-model-name Qwen/Qwen3-0.6B
```

## Verify discovery before training

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq .
curl -fsS http://127.0.0.1:8001/v1/rl/workers | jq .
```

The discovery response must report protocol version 1, model
`Qwen/Qwen3-0.6B`, `admin_base_url=http://127.0.0.1:8002`, and total
`world_size=1`.

## Run Prime-RL

```bash
CUDA_VISIBLE_DEVICES=1 uv run rl @ examples/dynamo/local/rl.toml \
  --output-dir outputs/dynamo-local --clean-output-dir
```

Success means four optimizer steps complete, policy versions advance, and a
generation request succeeds after each weight update.
