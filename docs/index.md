# PRIME-RL

PRIME-RL is a high-performance reinforcement learning framework for post-training language models. It is designed for scalability — from single-GPU experiments to large-scale multi-node MoE training.

## Documentation

### [Getting Started](getting-started/installation.md)

Install PRIME-RL, run your first training, and learn the config system.

### [Architecture](architecture/overview.md)

Understand how the orchestrator, trainer, and inference service work together, and the RL algorithms that drive training.

### [Performance](performance/overview.md)

Deep dive into what makes PRIME-RL fast: MoE with expert parallelism, context parallelism, memory efficiency, and distributed inference.

### [Single Node](single-node/overview.md)

Everything you can do on a single machine: SFT, RL, LoRA, distillation, and multimodal training.

### [Multi Node](multi-node/overview.md)

Scale to multiple nodes with SLURM or Kubernetes for both SFT and RL training.

### Additional Resources

- [Supported Models](supported-models.md)
- [Custom Algorithms](custom-algorithms.md)
- [Metrics & W&B](metrics.md)
- [Logging](logging.md)
- [Troubleshooting](troubleshooting.md)
