# Nemotron 3.5 Super VLM RL with Dynamo

This recipe runs three RL optimizer steps with `nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026`, one image per color-codeword rollout, four Prime-RL trainer ranks with expert parallelism, a TP=4 `dynamo.vllm` worker, routed-expert replay, sampling-mask replay, NCCL weight updates, and W&B metrics.

The recipe uses separate component configs because inference is an external Dynamo deployment. This keeps the discovered Dynamo worker world size explicit in both NCCL participants and avoids launching a second Prime-RL-managed vLLM server.

## Requirements

- Eight GPUs for the single-node layout below: GPUs 0-3 for Dynamo inference and GPUs 4-7 for Prime-RL training.
- A Prime-RL build with Nemotron-H Omni VLM training support and the `nemotron-3.5` renderer.
- A Dynamo build with the RL-enabled `dynamo.vllm` worker administration routes.
- vLLM 0.29 with Nemotron image dtype handling.
- A local immutable Hugging Face snapshot containing the original checkpoint and its converted `prime/` directory.
- `HF_TOKEN` for the gated checkpoint and `WANDB_API_KEY` for online W&B logging.

Install Prime-RL and its workspace environments:

```bash
git submodule update --init --recursive
uv sync --all-extras --all-packages
```

Set paths and a shared Dynamo namespace in every terminal:

```bash
export MODEL_NAME=nvidia/NVIDIA-Nemotron-3.5-Super-EA-09112026
export MODEL_DIR=/path/to/models--nvidia--NVIDIA-Nemotron-3.5-Super-EA-09112026/snapshots/<revision>
export RUN_DIR=/path/to/outputs/nemotron35-dynamo-vllm-3step
export DYN_NAMESPACE=nemotron35-rl
export DYN_ENABLE_RL=true
export DYN_RL_PORT=8001
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV=/tmp/nemotron35-dynamo-discovery
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE=true
```

The checkpoint path must resolve beneath a full 40-character immutable snapshot revision because the trusted Nemotron vision module is loaded from that snapshot. Verify the converted trainer weights before launch:

```bash
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$MODEL_DIR/prime/.prime-v1"
```

## 1. Start the Dynamo frontend

```bash
python -m dynamo.frontend \
  --namespace "$DYN_NAMESPACE" \
  --http-host 127.0.0.1 \
  --http-port 8000 \
  --router-mode round-robin
```

## 2. Start the TP=4 Dynamo vLLM worker

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DYN_SYSTEM_HOST=127.0.0.1 \
DYN_SYSTEM_PORT=8081 \
python -m dynamo.vllm \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --enable-rl \
  --enable-multimodal \
  --disaggregation-mode agg \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --moe-backend triton \
  --trust-remote-code \
  --enable-return-routed-experts \
  --return-sampling-mask \
  --logprobs-mode processed_logprobs \
  --generation-config vllm \
  --hf-overrides '{"architectures":["NemotronH_Super_Omni_Reasoning_V3"],"moe_router_dtype":"float32"}' \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --max-logprobs 128 \
  --attention-backend TRITON_ATTN \
  --mm-encoder-attn-backend TORCH_SDPA \
  --enforce-eager \
  --worker-extension-cls prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker
```

Confirm the frontend and the RL discovery contract before training. The discovery response must contain one worker with `world_size` equal to `4` and the weight-update administration routes.

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8001/v1/rl/workers
```

## 3. Start the color-codeword environment

```bash
uv run python -m prime_rl.entrypoints.env_server \
  @ examples/advanced/nemotron-3.5-super/env.toml
```

## 4. Start the trainer

The checked-in trainer config disables `torch.compile`, matching the stable smoke configuration. The shell wrapper gives every trainer rank private Triton, TorchInductor, and CUDA caches so concurrent kernel compilation cannot contend on shared cache files.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
uv run torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 4 \
  --no-python \
  bash -lc '
    export TRITON_CACHE_DIR=/tmp/nemotron35-cache/$LOCAL_RANK/triton
    export TORCHINDUCTOR_CACHE_DIR=/tmp/nemotron35-cache/$LOCAL_RANK/torchinductor
    export CUDA_CACHE_PATH=/tmp/nemotron35-cache/$LOCAL_RANK/cuda
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
    exec python -m prime_rl.trainer.rl.train "$@"
  ' bash \
  @ examples/advanced/nemotron-3.5-super/trainer.toml \
  --output-dir "$RUN_DIR" \
  --model.conversion-dir "$MODEL_DIR" \
  --model.name "$MODEL_DIR" \
  --tokenizer.name "$MODEL_DIR"
```

## 5. Start the orchestrator

```bash
uv run python -m prime_rl.entrypoints.orchestrator \
  @ examples/advanced/nemotron-3.5-super/orchestrator.toml \
  --output-dir "$RUN_DIR"
```

The run is accepted only after all three optimizer steps finish, policy versions `v0` through `v3` are applied, image tensors and routed-expert decisions reach the trainer, no rollout errors or cancellations occur, and KL, loss, reward, gradient norm, truncation, and weight-update metrics are reviewed. For a longer run, override `--max-steps 8` on both trainer and orchestrator after the three-step gate passes.

For Kubernetes, keep the same configs and override `model.client.base_url`, `model.client.dynamo.discovery_url`, weight-broadcast host, and rollout-transport host with routable Service or pod addresses. Do not use loopback across pods.
