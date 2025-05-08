# prime-rl - decentralized RL training at scale

prime-rl is a codebase for decentralized RL training at scale.



## install
quick install
```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/install.sh | bash
```


## Dev


1. Clone: 

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Set up the environment:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
uv pip install flash-attn --no-build-isolation
```

4. Precommit install

```bash
uv run pre-commit install
```

5. Test

```bash
uv run pytest
```

6. debug run 

training

```bash
uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/debug.toml
```

inference
```bash
uv run python src/zeroband/infer.py @ configs/inference/debug.toml
```


## Debug math run

on two different terminal do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/simple_math.toml
```

then start the trainer

```bash
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/simple_math.toml
```


## 2k seq length run

on two different terminal do:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/deepscaler.toml
```

then start the trainer

```bash
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/deepscaler.toml
```

if running on h100 node instead of H200 you should add ` --train.micro_bs 4`

## Distributed inference

Inference supports running in multi-node multi-GPU setups supporting DP, TP and PP, and sensible combinations of these.
Below are examples of how to run inference for different parallelization strategies.

Single Node (DP=1, TP=1, PP=1, *requires 1 GPU*)

```bash
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name Qwen/Qwen3-14B
```

Only TP (TP=2, PP=1, DP=1, *requires 2 GPUs*)

```bash
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name Qwen/Qwen3-14B \
	--tp 2
```

Only DP (DP=2, TP=1, PP=1, *requires 2 GPUs*)

```bash
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name Qwen/Qwen3-14B \
	--dp 2
```

Only PP (DP=1, TP=1, PP=2, *requires 2 GPUs*)

```bash
# Node 1
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name mikasenghaas/Qwen3-14B-0.2 \
	--pp.rank 0 \
	--pp.world-size 2 \
	--pp.iroh-seed 0 \
	--pp.iroh-peer-id ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337 \
	--seed 69
```

```bash
# Node 2
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name mikasenghaas/Qwen3-14B-1.2 \
	--pp.rank 1 \
	--pp.world-size 2 \
	--pp.iroh-seed 1 \
	--pp.iroh-peer-id ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03 \
	--seed 69
```

*Note: Setting the seed here is important to ensure model shards work on the same data shards.*

DP+TP (DP=2, TP=2, PP=1, *requires 4 GPUs*)

```bash
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name Qwen/Qwen3-14B \
	--dp 2 \
	--tp auto
```

PP+TP (DP=1, TP=2, PP=2, *requires 4 GPUs*)

```bash
# Node 1
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0,1 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name mikasenghaas/Qwen3-14B-0.2 \
	--tp auto \
	--pp.rank 0 \
	--pp.world-size 2 \
	--pp.iroh-seed 0 \
	--pp.iroh-peer-id ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337 \
	--seed 69
```

```bash
# Node 2
PRIME_LOG_LEVEL=DEBUG VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=2,3 uv run python src/zeroband/infer.py @ configs/inference/debug.toml --model-name mikasenghaas/Qwen3-14B-1.2 \
	--tp auto \
	--pp.rank 1 \
	--pp.world-size 2 \
	--pp.iroh-seed 1 \
	--pp.iroh-peer-id ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03 \
	--seed 69
```

We don't support DP+PP and so that configuration will raise an exception.