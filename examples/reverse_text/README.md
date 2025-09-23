We demonstrate how to setup an SFT warmup using the toy [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment. We have generated a small dataset ([PrimeIntellect/Reverse-Text-SFT](https://huggingface.co/PrimeIntellect/Reverse-Text-SFT)) of examples where the prompt is a small chunk of text and the completion is the reverse of that chunk. We will fine-tune `PrimeIntellect/Qwen3-0.6B` (`Qwen/Qwen3-0.6B` but with Qwen-2.5 chat template) on this dataset. Again, because of the small context, training should be extremely quick.

```bash
uv run sft @ examples/reverse_text/sft.toml
```

If you have access to multiple GPUs, use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nproc-per-node` to start the training. 

```bash
uv run torchrun \
  --nproc-per-node 8 \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft.toml
```

### Single-Node Training

First, start a pre-layouted `tmux` session to view the logs from all submodules.

```bash
bash scripts/tmux.sh
```

Then, start the training with the `rl` entrypoint 

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --inference @ examples/reverse_text/rl/infer.toml
```

By default, this command will spin up and tear down the inference server with each invocation. For development purposes it is often useful to start the inference server once and keep it alive across experiments to avoid suffering the vLLM startup time repeatedly.

```bash
# Run this in the `Inference` pane
uv run inference @ examples/reverse_text/rl/infer.toml
```

Then, you can repeatedly restart the trainer and orchestrator in the `Trainer` pane.

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml
```
