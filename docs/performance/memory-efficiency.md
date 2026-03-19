# Memory Efficiency

PRIME-RL provides several techniques to reduce GPU memory usage during training.

## Activation Checkpointing

Enable full activation checkpointing with `--model.ac`. You can control the frequency with `--model.ac.freq`.

```toml
[model.ac]
freq = 1
```

## Activation Offloading

Offload activations to CPU during the forward pass and reload them during backward:

```toml
[model.ac_offloading]
max_inflight_activations = 5
```

## Optimizer CPU Offload

Offload optimizer states to CPU memory:

```toml
[model]
optim_cpu_offload = true
```

<!-- TODO: add details on scaledown and other memory optimization techniques -->
