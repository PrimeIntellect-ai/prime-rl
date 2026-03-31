# Reducing memory usage

While most of our parallelism techniques in prime-rl are designed to scale training up (FSDP, EP, CP, ...), we also provide many tools to scale training down that allow training large MoE models on a limited amount of GPUs.

These techniques target the trainer part of prime-rl.


## TLDR: config to use for maximum memory usage reduction with correct throughput

```toml
[trainer.model]
impl = "custom"
attn = "flash_attention_2"
fused_lm_head_token_chunk_size = 1024
ep = 8
cp = 2
optim_cpu_offload = true

[trainer.model.compile]

[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 1
```

## Activation checkpointing

Activation checkpointing discards intermediate activations during the forward pass and recomputes them during the backward pass, trading compute for memory.

To enable it, use:

```toml
[trainer.model.ac]
freq = 1
```

`freq` controls how often layers are checkpointed: every `freq` layers. Lower values yield lower memory usage (e.g. `freq = 1` checkpoints every layer).

## Activation offloading

Activation offloading offloads the activations to CPU to reduce the memory usage of the trainer. It can be used in combination with activation checkpointing.

To enable it, use:

```toml
[trainer.model.ac]
freq = 1

[trainer.model.ac_offloading]
max_inflight_activations = 5
```

## Chunk loss

Chunk loss splits the loss computation into smaller chunks to reduce the memory usage of the trainer.

To enable it, use:

```toml
[trainer.model]
fused_lm_head_token_chunk_size = auto
```


## Expert parallelism

While expert parallelism splits the weights of the experts across all GPUs like FSDP, using EP still reduces memory usage by reducing the communication size and therefore the FSDP buffer.

EP is only available for models with MoE layers using the custom model implementation.

```
[trainer.model]
impl = "custom"
ep = 8
```

## Context parallelism

Context parallelism splits the context into smaller chunks to reduce the memory usage of the activations. We don't advise using CP across multiple nodes (i.e., increasing CP beyond 8).

CP is only available for certain models and only with the custom model implementation.

```
[trainer.model]
impl = "custom"
cp = 2
```

We recommend CP 2 or CP 4 for most 128K sequence length training runs. Can be pushed to 8.


## torch compile

Enabling torch.compile can reduce the memory usage for certain model architectures, especially MoE with the custom model implementation.

```
[trainer.model.compile]
```

## CPU Optimizer offloading

Offloading the optimizer states to CPU can reduce the memory usage of the trainer significantly, especially at low GPU counts where the optimizer states take a lot of memory as they won't be sharded enough.

In RL, in contrast with pretraining, we end up with many gradient accumulation steps, so the cost of offloading the optimizer states is not as high as in pretraining, and indeed barely noticeable.

```
[trainer.optim]
optim_cpu_offload = true
```

## :warning: FSDP CPU offloading

FSDP CPU offloading offloads the parameters, gradients, and optimizer states to CPU to reduce the memory usage of the trainer.

This will make training significantly slower and is not recommended most of the time.

```
[trainer.model]
fsdp_cpu_offload = true
```

## :warning: Lora training

LoRA training significantly reduces the memory usage of the trainer at the cost of smaller gradient updates.

```
[trainer.model.lora]
rank = 8
```

