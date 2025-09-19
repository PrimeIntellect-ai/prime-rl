# Benchmarking

We provide a convenient way to benchmark the performance of the inference engine and trainer using the `--bench` flag. It will run each module in isolation for a few steps and log performance statistics to the console and, optionally, W&B.

**Inference**

To benchmark inference, first spin up the inference server with an experiment configuration

```bash
uv run inference @ configs/reverse_text/infer.toml
```

Then, start the orchestrator with the matching configuration file in benchmark mode

```bash
uv run orchestrator @ configs/reverse_text/orch.toml --bench
```

**Trainer**

To benchmark the RL trainer, simply run the trainer against a fake data loader with batch certain specifications.

```bash
uv run trainer @ configs/reverse_text/train.toml --bench --data.fake.micro_batch_size 8 --data.fake.batch_size 128 --data.fake.seq_len 128
```

**RL**

You can benchmark both the RL trainer and inference at the same time with the `rl.py` entrypoint. Note, that the benchmarking is still decoupled.

```bash
uv run rl   \
  --trainer @ configs/reverse_text/train.toml  \
  --orchestrator @ configs/reverse_text/orch.toml \
  --inference @ configs/reverse_text/infer.toml \
  --bench
```

**SFT**

Benchmark the SFT trainer against `fixed` or `variable` length fake data by specifyin `--data.fake.type`

```bash
uv run sft --bench --data.fake.type fixed --data.micro-batch-size 8 --data.batch-size 8 --data.seq-len 128
```