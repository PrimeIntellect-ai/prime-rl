# Multi-Node SFT

For training on multiple nodes, use `torchrun` with the `--nnodes`, `--node-rank`, and `--rdzv-endpoint` flags.

## Setup

First, decide which node will be your head node and find a reachable private IP address for it. If your nodes are not colocated, you will likely need to setup VPN (e.g. [Tailscale](https://tailscale.com)) for the nodes to reach each other.

(*Skip this step if the default network interface is sufficient.*) Make sure to set the network interface for GLOO and NCCL to one that allows all nodes to reach each other.

```bash
# On both nodes
export GLOO_SOCKET_IFNAME=...
export NCCL_SOCKET_IFNAME=...
```

Then, configure the rendezvous endpoint to allow the nodes to find each other. Here, `MASTER_ADDR` is the private IP address of the head node and `MASTER_PORT` is a free port on the head node, typically port 29500 for `torchrun`.

```bash
# On both nodes
export MASTER_ADDR=...
export MASTER_PORT=...
```

## Launch

On the head node:

```bash
uv run torchrun \
  --nnodes 2 \
  --node-rank 0 \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local-rank-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ path/to/config.toml
```

On the second node:

```bash
uv run torchrun \
  --nnodes 2 \
  --node-rank 1 \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --local-rank-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ path/to/config.toml
```

For SLURM-managed multi-node SFT, see the [SLURM guide](slurm.md).
