# Local Dynamo smoke test

Runs Prime-RL trainer on GPU 1 with a real Dynamo inference stack on GPU 0.

## Start the Dynamo stack

```bash
# 1. etcd
etcd --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  --listen-peer-urls http://127.0.0.1:2380 &

# 2. vLLM engine (GPU 0) — use the fork from biswapanda/vllm#1
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
  vllm-rs serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8002 \
  --grpc-port 50051 --python $(which python) \
  -- --worker-extension-cls prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker &

# 3. Dynamo frontend
DYN_ENABLE_RL=true DYN_RL_PORT=8001 DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE=true \
  python -m dynamo.frontend --http-host 0.0.0.0 --http-port 8000 \
  --namespace dynamo --discovery-backend etcd --request-plane tcp \
  --event-plane zmq --router-min-initial-workers 1 &

# 4. Sidecar
DYN_NAMESPACE=dynamo DYN_DISCOVERY_BACKEND=etcd DYN_ENABLE_RL=true \
  VLLM_HTTP_ENDPOINT=http://127.0.0.1:8002 \
  dynamo-vllm-sidecar --vllm-endpoint 127.0.0.1:50051 \
  --admin-endpoint http://127.0.0.1:8002 --model-path Qwen/Qwen3-0.6B \
  --namespace dynamo --rl-discovery-model-name Qwen/Qwen3-0.6B &
```

## Run Prime-RL

```bash
CUDA_VISIBLE_DEVICES=1 uv run rl @ examples/dynamo/local/rl.toml \
  --output-dir outputs/dynamo-local --clean-output-dir
```

Success = 4 optimizer steps with math rewards and advancing weight versions.
