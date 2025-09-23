# Reverse Text

We demonstrate how to train `Qwen3-0.6B` to reverse a small chunk of text. This will require a SFT warmup to learn text reversal and then a quick RL in [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment. We use a similar setup in our CI at the moment.

To run this example, you need access to one or more GPUs with at least 48GB unified memory. If you run on a different setup, you may need to adjust the commands to suit your setup.

## Setup

Ensure that the environment is installed (should be included in `pyproject.toml`)

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

This is of course just a quick vibe check and no full-fledged evaluation, but we can see that the model struggles with this task. In this specific instance, we got an **average reward of ~0.05** across the 20x3 rollouts. Let's do some training!

## SFT

We have generated a prompt-completion SFT dataset ([willcb/R1-reverse-wikipedia-paragraphs-v1-1000](https://huggingface.co/willcb/R1-reverse-wikipedia-paragraphs-v1-1000)) of 1000 examples of text reversal.

We will fine-tune `PrimeIntellect/Qwen3-0.6B`, which is a clone of `Qwen/Qwen3-0.6B` with an adapted chat template. 

On a single GPU, run

```bash
# In the `Trainer` pane
uv run sft @ examples/reverse_text/sft/train.toml \
  --wandb.project ... \
  --wandb.name ... \
  --weights
```

On multiple GPUs, run

```bash
uv run torchrun \
  --nproc-per-node ... \
  src/prime_rl/trainer/sft/train.py @ examples/reverse_text/sft/train.toml \
  --wandb.project ... \
  --wandb.name ... \
  --weights
```

This should write a weight checkpoint in `outputs/weights/step_100`. Upload it to HF to be able to use it as the base model for RL.

```bash
uv run hf upload <user>/Qwen3-0.6B-Reverse-Text-SFT outputs/weights/step_100
```

We have run the same commands as above. Check out the run in [W&B](https://wandb.ai/primeintellect/examples?nw=s3p14m48jod). Find our final artifact on HF [`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT).

## RL

For the RL we will only do 20 steps at 8x16 rollouts, for a total batch size of 128 and sequence length 128. Because of the small context, training should be extremely quick.

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --inference @ examples/reverse_text/rl/infer.toml \
  --no-trainer.model.load-using-meta \
  --model.name ... \
  --wandb.project ... \
  --wandb.name ...
```

This will write a weight checkpoint in `outputs/weights/step_20`. As before, let's upload it to HF.

```bash
uv run hf upload <user>/Qwen3-0.6B-Reverse-Text-RL outputs/weights/step_20
```

We have run the same commands as above. Check out the run in [W&B](https://wandb.ai/primeintellect/examples?nw=yxjwjc556do). Find our final artifact on HF [`PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL).

## Evals

Let's see how our final RL checkpoints perform on the `reverse-text` environment.

```bash
# Run this in the `Inference` pane
uv run inference --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
```

```bash
# Run this in the `Trainer` pane
uv run vf-eval reverse-text -m PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

Way better! Now we get an **average reward of ~0.8**.