# SFT

We demonstrate how to setup an SFT warmup using the toy [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment. We have generated a small dataset ([PrimeIntellect/Reverse-Text-SFT](https://huggingface.co/PrimeIntellect/Reverse-Text-SFT)) of examples where the prompt is a small chunk of text and the completion is the reverse of that chunk. We will fine-tune `PrimeIntellect/Qwen3-0.6B` (`Qwen/Qwen3-0.6B` but with Qwen-2.5 chat template) on this dataset. Again, because of the small context, training should be extremely quick.

To check all available configuration options, run `uv run sft --help`.


### Single-Node Training

On a single GPU, start the training with the `sft` entrypoint

```bash
uv run sft @ examples/reverse_text/sft.toml
```

If you have access to multiple GPUs, use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nproc-per-node` to start the training. 

```bash
uv run torchrun \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft.toml
```

### Multi-Node Training

On multiple nodes (potentially with multiple GPUs), use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nnodes` and `--nproc-per-node` to start the training. You need to set up the rendezvous endpoint to allow the nodes to find each other. This should be the private IP address of your master node that needs to be reachable from all other nodes. For more details, see the [PyTorch documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).

```bash
export RDZV_ENDPOINT=...
```

```bash
uv run torchrun \
  --nnodes=2 \
  --nproc-per-node 8 \
  --rdzv-endpoint=$RDZV_ENDPOINT \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft.toml
```