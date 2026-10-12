# Reverse Text

We demonstrate how to train `Qwen3-0.6B` to reverse a small chunk of text. We will use a SFT warmup to learn the skill of text reversal on longer documents and then a quick RL run on the `reverse-text` taskset. We use a similar setup in our CI and for development.

> The commands in this example were designed to be run on 2 GPUs (one trainer and one inference GPU). It is possible to run on less or more GPUs using different deployment strategies. If you run on a different setup, you may need to adjust the start commands.

## Setup

The `reverse-text` taskset is included through the Verifiers workspace. After syncing the repository, verify it with:

```bash
uv run python -c "import reverse_text"
```

We'll use two terminals: one for the inference server, one for everything else. To watch the run while it trains — metrics, resolved configs, rollout traces, and logs in one place — start the local dashboard and open http://localhost:7788:

```bash
uv run dashboard
```

Let's check how well `Qwen3-0.6B` does out-of-the-box on the `reverse-text` environment. 

```bash
# Run this in the inference terminal
uv run inference --vllm.model Qwen/Qwen3-0.6B
```

```bash
# Run this in the other terminal
uv run eval @ examples/basic/reverse-text/eval.toml
```

This is of course just a quick vibe check and no full-fledged evaluation, but we can see that the model struggles with this task. In this specific instance, we got an **average reward of ~0.05** across the 20x3 rollouts. Let's do some training!

## SFT

We will fine-tune `PrimeIntellect/Qwen3-0.6B` ([HF](https://huggingface.co/PrimeIntellect/Qwen3-0.6B)), which is a clone of `Qwen/Qwen3-0.6B` ([HF](https://huggingface.co/Qwen/Qwen3-0.6B)) with a chat template suitable for multi-turn RL, on `willcb/R1-reverse-wikipedia-paragraphs-v1-1000` ([HF](https://huggingface.co/datasets/willcb/R1-reverse-wikipedia-paragraphs-v1-1000)) which contains 1K examples of reversals of small paragraphs.

*Check out the logs of the SFT run on [W&B](https://wandb.ai/primeintellect/examples?nw=s3p14m48jod).*

To train on a single GPU, run

```bash
# Run this in the other terminal
uv run sft @ examples/basic/reverse-text/sft.toml \
  --run.name sft \
  --monitors.wandb.project ... \
  --monitors.wandb.name ...
```

To train on multiple GPUs, run

```bash
# Run this in the other terminal
uv run torchrun \
  --local-ranks-filter 0 \
  --nproc-per-node ... \
  src/prime_rl/trainer/sft/train.py @ examples/basic/reverse-text/sft.toml \
  --monitors.wandb.project ... \
  --monitors.wandb.name ...
```

This should write a DCP checkpoint in `outputs/sft/checkpoints/step_100`.

We have uploaded the final model as [`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT).

## RL

For the RL we will only do 20 steps at 8x16 rollouts, for a total batch size of 128 and sequence length 128. Because of the small context, training should be extremely quick.

*Check out the logs of the RL run on [W&B](https://wandb.ai/primeintellect/examples?nw=yxjwjc556do).*

```bash
# Run this in the other terminal
uv run rl @ examples/basic/reverse-text/rl.toml \
  --model.name ... \
  --run.name rl \
  --monitors.wandb.project ... \
  --monitors.wandb.name ...
```

This will write a DCP checkpoint in `outputs/rl/checkpoints/step_20`.

We have uploaded the final model as [`PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL).

## Evals

Let's see how our final RL checkpoints perform on the `reverse-text` environment.

```bash
# Run this in the inference terminal
uv run inference --vllm.model PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
```

```bash
# Run this in the other terminal
uv run eval @ examples/basic/reverse-text/eval.toml -m PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
```

Way better! Now we get an **average reward of ~0.8**.
