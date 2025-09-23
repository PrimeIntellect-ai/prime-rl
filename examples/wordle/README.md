# Wordle

We demonstrate how to train `Qwen3-1.7B` to play Wordle. This will require a SFT warmup to learn format of the environment and then RL training in the multi-turn [`wordle`](https://app.primeintellect.ai/dashboard/environments/primeintellect/wordle) environment.

We have developed this example on 2xH200 GPUs. If you run on a different setup, you may need to adjust the commands to suit your setup.

## Setup

First, let's install the environment using the `prime` CLI.

```bash
prime env install will/wordle
```

Verify your installation by trying to import the environment.

```bash
uv run python -c "import wordle"
```

Let's check how well `Qwen3-1.7B` does out-of-the-box on the `wordle` environment. First, let's start a `tmux` session which we will use throughout the experiment.

```bash
bash scripts/tmux.sh
```

Then, start the inference server

```bash
# Run this in the `Inference` pane
uv run inference --model.name Qwen/Qwen3-1.7B
```

```bash
# Run this in the `Trainer` pane
uv run vf-eval wordle -m Qwen/Qwen3-1.7B -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

This is of course full-fledged evaluation, but can give a good first impression of how the model performs on this task. In this case, we got an **average reward of ~0.209** across the 20x3 rollouts, but most of the reward is coming from format and partial rewards. In this batch of eval samples, it has not guessed correctly within a game. Upon inspection of the samples using `vf-tui`, we can see that the model is repeatedly submitting guesses in the wrong format. Let's do some SFT warmup to get the model to learn the format of the environment.

## SFT

We have generated a prompt-completion SFT dataset ([willcb/V3-wordle](https://huggingface.co/willcb/V3-wordle)) of examples of Wordle games.

We will fine-tune `PrimeIntellect/Qwen3-1.7B`, which is a clone of `Qwen/Qwen3-1.7B` with an adapted chat template. 

On a single GPU, run

```bash
# In the `Trainer` pane
uv run sft @ examples/wordle/sft.toml \
  --wandb.project ... \
  --wandb.name ... \
  --weights
```

On multiple GPUs, run

```bash
uv run torchrun \
  --nproc-per-node ... \
  src/prime_rl/trainer/sft/train.py @ examples/wordle/sft.toml \
  --wandb.project ... \
  --wandb.name ... \
  --weights
```

This should write a weight checkpoint in `outputs/weights/step_20`. Upload it to HF to be able to use it as the base model for RL.

```bash
uv run hf upload <user>/Qwen3-1.7B-Wordle-SFT outputs/weights/step_20
```

We have run the same commands as above. Check out the run in [W&B](https://wandb.ai/primeintellect/examples?nw=h8yesgpmst). Find our final artifact on HF [`PrimeIntellect/Qwen3-1.7B-Wordle-SFT`](https://huggingface.co/PrimeIntellect/Qwen3-1.7B-Wordle-SFT).

## RL

We will do 100 RL training steps with 64x16 rollouts, for a total batch size of 1024, at a context length of 4096.

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/wordle/rl/train.toml \
  --orchestrator @ examples/wordle/rl/orch.toml \
  --inference @ examples/wordle/rl/infer.toml \
  --no-trainer.model.load-using-meta \
  --model.name ... \
  --wandb.project ... \
  --wandb.name ...
```

This will write a weight checkpoint in `outputs/weights/step_100`. As before, let's upload it to HF.

```bash
uv run hf upload <user>/Qwen3-1.7B-Wordle-RL outputs/weights/step_100
```

We have run the same commands as above. Check out the run in [W&B](https://wandb.ai/primeintellect/examples?nw=2isof8knxo5). Find our final artifact on HF [`PrimeIntellect/Qwen3-1.7B-Wordle-RL`](https://huggingface.co/PrimeIntellect/Qwen3-1.7B-Wordle-RL).

## Evals

Let's see how our final RL checkpoints perform on the `wordle` environment.

```bash
# Run this in the `Inference` pane
uv run inference --model.name PrimeIntellect/Qwen3-1.7B-Wordle-RL
```

```bash
# Run this in the `Trainer` pane
uv run vf-eval wordle -m PrimeIntellect/Qwen3-1.7B-Wordle-RL -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

Way better! Now we get an **average reward of ~XXX**.