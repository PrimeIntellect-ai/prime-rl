# Performance

PRIME-RL is built for high throughput training at scale. This section covers the techniques we use to maximize performance.

- [**MoE & Expert Parallelism**](moe.md) — training large Mixture-of-Experts models with expert parallelism
- [**Context Parallelism**](context-parallelism.md) — parallelizing over long sequences
- [**Memory Efficiency**](memory-efficiency.md) — activation checkpointing, optimizer offloading, scaledown
- [**Distributed Inference**](distributed-inference.md) — PD disaggregation, FP8
- [**Benchmarking**](benchmarking.md) — measuring throughput and MFU
