# Multi-Node RL

> We currently require a shared file system for multi-node RL training.

## Setup

Ensure that all nodes have access to a shared file system and that the node running the inference server is reachable from the orchestrator via a private or public IP address.

```bash
# On all nodes
export OUTPUT_DIR=...               # Path to directory in shared file system
export INFERENCE_SERVER_IP=...      # Reachable IP address of the inference node
export INFERENCE_SERVER_API_KEY=... # API key for the inference server
```

## Launch

Start the inference server on one node:

```bash
uv run inference ... \
    --api-key $INFERENCE_SERVER_API_KEY --parallel ...
```

Start a single orchestrator:

```bash
uv run orchestrator ... \
    --client.base-url http://$INFERENCE_SERVER_IP:8000/v1 \
    --client.api-key-var INFERENCE_SERVER_API_KEY \
    --output-dir $OUTPUT_DIR
```

Start the trainer on another node:

```bash
uv run torchrun \
    --nproc-per-node 8 \
    --local-rank-filter 0 \
    src/prime_rl/trainer/rl/train.py ... \
    --output-dir $OUTPUT_DIR
```

You can further scale up the number of nodes used by the trainer and inference server. Make sure that there is only a single orchestrator instance.

## Multi-Node Inference

For multi-node inference without SLURM, we rely on vLLM's multi-node data parallel deployment. See [Distributed Inference](../performance/distributed-inference.md) for details on TP/DP configuration.

Setup the data parallel address:

```bash
# On both nodes
export DATA_PARALLEL_ADDRESS=...
export DATA_PARALLEL_RPC_PORT=...
```

To run TP=4 and DP=4 with DP ranks 0-1 on the head node and DP ranks 2-3 on the second node:

```bash
# On node 0
uv run inference \
	--data-parallel-size 4 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT
```

```bash
# On node 1
uv run inference \
	--data-parallel-size 4 \
	--tensor-parallel-size 4 \
	--data-parallel-size-local 2 \
	--data-parallel-address $DATA_PARALLEL_ADDRESS \
	--data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT \
	--data-parallel-start-rank 2 \
	--headless
```

For SLURM-managed multi-node RL, see the [SLURM guide](slurm.md).
