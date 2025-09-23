# Benchmarking

We provide a convenient way to benchmark the performance, mainly measured in throughput and MFU, of the inference engine and trainer using the `--bench` flag. It will run each module in isolation for a few steps and log performance benchmark results in a rich table to the console.

## SFT

Benchmark on the default fake data configuration

```bash
uv run sft ... --data.type fake --bench
```

Benchmark with variable-length, instead of fixed-length, fake data to more closely simulate real data.

```bash
uv run sft ... --data.type fake --data.length variable --bench
```

Benchmark different batch configurations, i.e. the (micro) batch size and sequence length

```bash
uv run sft ... --data.type fake --data.seq-len 4096 --data.batch-size 64 --data.micro-batch-size 2 --bench
```

Benchmark against a real dataset

```bash
uv run sft ... --data.name PrimeIntellect/Reverse-Text-SFT --bench
```

Benchmark against a training configuration

```bash
uv run sft @ path/to/config.toml --bench
```

## RL

### Trainer

Benchmark on a fake data loader

```bash
uv run trainer ... --data.fake --bench
```

Benchmark different batch configurations, i.e. the (micro) batch size and sequence length

```bash
uv run trainer ... --data.fake.seq-len 4096 --data.fake.batch-size 64 --data.fake.micro-batch-size 2 --bench
```

*Note, that it is not yet possible to benchmark the RL trainer against real data when benchmarking the RL trainer in isolation.*

### Inference

To benchmark the inference engine in isolation, start the inference server with the correct configuration file and run the orchestrator with the `--bench` flag.

```bash
uv run inference @ path/to/config.toml
```

```bash
uv run orchestrator @ path/to/config.toml --bench
```

*Note, that it is not yet possible to benchmark the inference engine against fake data.*

## Trainer + Inference

To benchmark the full RL training, you can add the `--bench` flag to your RL entrypoint. This will benchmark the RL trainer against fake data and the inference engine against real data from the orchestrator.

```bash
uv run rl   \
  --trainer @ path/to/train.toml  \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --bench
```