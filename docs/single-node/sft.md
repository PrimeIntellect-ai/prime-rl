# SFT (Single Node)

We provide a fairly straight-forward SFT trainer which is capable of fine-tuning any conversational model on multi-turn conversation with tool calling. It shares a lot of components with the RL trainer, such as the modeling code, parallelism techniques, checkpoint format, logger, etc. which ensures a seamless post-training workflow.

## Dataset Format

Prepare a dataset in [prompt-completion format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion) (we do not support any other format). Single-turn fine-tuning should be compatible with the chat templates of most models. However, to properly handle loss masking, we require that the tokenizer's chat template satisfies a prefix property: the tokenization of any conversation prefix must be a prefix of the tokenization of the full conversation.

## Single GPU

```bash
uv run sft @ path/to/config.toml
```

## Multi-GPU

Use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nproc-per-node`:

```bash
uv run torchrun \
  --local-rank-filter 0 \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ path/to/config.toml
```

The `--local-rank-filter` flag is used to only log from the master rank, as detailed in [Logging](../logging.md).

See all available configuration options with `uv run sft --help`.
