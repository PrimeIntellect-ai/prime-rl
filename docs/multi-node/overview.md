# Multi-Node Deployment

PRIME-RL supports multi-node training for both SFT and RL. Multi-node RL requires a shared file system for coordinating weight updates between trainer and inference.

## Deployment Options

- [**SLURM**](slurm.md) — built-in SLURM support with config-based job submission
- [**Kubernetes**](kubernetes.md) — Helm chart for Kubernetes clusters
- [**SFT**](sft.md) — multi-node SFT with torchrun
- [**RL**](rl.md) — multi-node RL with separate components across nodes
