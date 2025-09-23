# Reverse Text

We demonstrate how to train `Qwen3-0.6B` to reverse a small chunk of text. This will require a small SFT warmup to get some initial reward and then some RL in [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment.


## Setup

Before starting, make sure that you have access to one or more GPUs with at least 48GB unified memory. These docs were written running on a 2x4090. If you run on a different setup, you may need to adjust the commands to suit your setup.

First, ensure that the environment is installed.

```bash
uv run python -c "import reverse_text"
```

Let's check how well `Qwen3-0.6B` does out-of-the-box on the `reverse-text` environment. First, let's start a `tmux` session which we will use throughout the experiment.

```bash
bash scripts/tmux.sh
```

Then, start the inference server

```bash
# Run this in the `Inference` pane
uv run inference --model.name Qwen/Qwen3-0.6B
```

```bash
# Run this in the `Trainer` pane
uv run vf-eval reverse-text -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

This is of course just a quick vibe check and no full-fledged evaluation, but we can see that the model *struggles a lot*. In this specific instance, we got an **average reward of ~0.05** across the 20x3 rollouts. Let's do some training!

## SFT

We have generated a dataset ([willcb/R1-reverse-wikipedia-paragraphs-v1-1000](https://huggingface.co/willcb/R1-reverse-wikipedia-paragraphs-v1-1000)) of 1000 examples with small paragraphs to reverse the prompt is a small chunk of text and the completion is the reverse of that chunk. 

We will fine-tune `PrimeIntellect/Qwen3-0.6B`, which is a clone of `Qwen/Qwen3-0.6B` but with an adapted chat template. We do 100 steps at batch size 16 and sequence length 4096 and save the final checkpoint to disk.

On a single GPU, run

```bash
uv run sft @ examples/reverse_text/sft.toml \
  --wandb.project ... \
  --wandb.name ... \
  --ckpt
```

On multiple GPUs, run

```bash
uv run torchrun \
  --nproc-per-node ... \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft.toml
  --wandb.project ... \
  --wandb.name ... \
  --ckpt
```

This should write a weight checkpoint in `outputs/weights/step_100`. Upload it to HF for the next step.

```bash
uv run hf upload <user>/<name> outputs/weights/step_100
```

We did the same and uploaded it to `PrimeIntellect/Qwen3-0.6B-SFT-Reverse-Text`.

## RL

First, start a pre-layouted `tmux` session to view the logs from all submodules.

```bash
bash scripts/tmux.sh
```

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --inference @ examples/reverse_text/rl/infer.toml
```