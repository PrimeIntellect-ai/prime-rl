# Docs

This directory maintains the documentation for PRIME-RL. It is organized into the following sections:

**Getting Started**
- [**Async Training**](async.md) - Understanding asynchronous off-policy training and step semantics
- [**Entrypoints**](entrypoints.md) - Overview of the main components (orchestrator, trainer, inference) and how to run SFT, RL, and evals
- [**Configs**](configs.md) - Configuration system using TOML files, CLI arguments, and environment variables
- [**Environments**](environments.md) - Installing and using verifiers environments from the Environments Hub

**Training**
- [**Trajectories**](../training/trajectories.md) - Multi-turn trajectory interleaving
- [**Checkpointing**](../training/checkpointing.md) - Saving and resuming training from checkpoints
- [**On-Policy Distillation**](../training/on-policy-distillation.md) - Using teacher models for dense token-level feedback
- [**Multimodal**](../training/multimodal.md) - Vision-Language Model support (experimental)

**Deployment**
- [**Deployment**](../deployment/deployment.md) - Training deployment on single-GPU, multi-GPU, and multi-node clusters
- [**SLURM**](../deployment/slurm.md) - SLURM cluster deployment
- [**Kubernetes**](../deployment/kubernetes.md) - Deploying PRIME-RL on Kubernetes with Helm

**Extending**
- [**Bring Your Own Algorithms**](../extending/bring-your-own-algorithms.md) - Custom loss and advantage functions
- [**Testing MoE at Small Scale**](../extending/testing-moe-at-small-scale.md) - Local MoE iteration

**Multi-Run Manager**
- [**Multi-Run Manager**](../multi-run-manager/multi-run-manager.md) - Managing concurrent LoRA training runs

**Observability**
- [**Logging**](../observability/logging.md) - Logging with loguru, torchrun, and Weights & Biases
- [**Metrics**](../observability/metrics.md) - W&B experiment tracking
- [**Benchmarking**](../observability/benchmarking.md) - Performance benchmarking and throughput measurement

**[Troubleshooting](../troubleshooting.md)** - Common issues and their solutions