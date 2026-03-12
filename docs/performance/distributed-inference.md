# Distributed Inference

PRIME-RL leverages vLLM for inference with support for tensor parallelism and data parallelism across multiple GPUs and nodes.

## Tensor Parallelism

Split a single model across multiple GPUs:

```toml
[inference.parallel]
tp = 4
```

## Data Parallelism

Run multiple independent model replicas:

```toml
[inference.parallel]
dp = 2
```

## PD Disaggregation

*Work in progress.* Prefill-decode disaggregation for improved inference throughput.

## FP8

*Work in progress.* FP8 quantized inference for reduced memory and higher throughput.
