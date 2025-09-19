
```bash
prime env info will/wordle
```

```bash
prime env install will/wordle
```

```bash
uv run python -c "import wordle"
```

```bash
uv run inference --model.name willcb/Qwen3-4B
```

```bash
uv run eval \
    --model.name willcb/Qwen3-4B \
    --environment-ids wordle \
    --sampling.max-tokens 4096
```

```bash
uv run vf-eval wordle -m willcb/Qwen3-4B -b http://localhost:8000/v1 -n 5 -r 3
```

We do not get any reward with the models out of the box and it seems like this is because it's not following the formatting of the environment. We do a light-weight SFT to warmup the model to ensure it gets some initial reward.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("willcb/V3-wordle", split="train")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def tokenize(example):
    return {"input_ids": tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=True)}


ds = ds.map(tokenize, desc="Tokenizing dataset")
print(ds.to_pandas().input_ids.apply(len).describe())
```

```bash
uv run sft @ examples/wordle/sft.toml --wandb.project ... --wandb.name ...
```

On 1xH100, this will take ~ .This will write checkpoints for steps 10 and 20 to `outputs/weights`. Let's evaluate these checkpoints.

We can try evaling all of these checkpoints again

```bash
uv run eval \
    --model.name willcb/Qwen3-4B \
    --environment-ids wordle \
    --sampling.max-tokens 4096 \
    --weights-dir outputs/weights \
    --no-eval-base
```

Base: Evaluated wordle in 18.82s (Avg@1=0.2200, Completion Length: 1050.15 (±445.21, ∈[590.00, 2732.00]), Truncated: 0.0%) 
Step 10: Evaluated wordle in 61.37s (Avg@1=0.7346, Completion Length: 2471.05 (±2361.75, ∈[633.00, 8834.00]), Truncated: 15.0%)
Step 20: Evaluated wordle in 66.37s (Avg@1=1.2530, Completion Length: 3002.50 (±2396.19, ∈[407.00, 9462.00]), Truncated: 20.0%)

The final checkpoint seems best, let's it to the HF hub and eval using `verifiers` to understand the subrewards.

```bash
export HF_TOKEN=...
hf upload Qwen3-4B-SFT-Wordle outputs/weights/step_20
```

```bash
uv run vf-eval wordle -m mikasenghaas/Qwen3-4B-SFT-Wordle -b http://localhost:8000/v1 -n 5 -r 3
```

Nice, we are getting way better rewards after the SFT! Let's try to RL in it now.

Before starting the RL training, we will benchmark the trainer and inference to decide on a good hardware setup.

```bash
uv run inference @ examples/wordle/rl/infer.toml
```

```bash
uv run orchestrator @ examples/wordle/rl/orch.toml --bench
```

```bash
uv run trainer @ examples/wordle/rl/train.toml \
    --data.fake.batch-size 64 \
    --data.fake.seq-len 4096 \
    --data.fake.micro-batch-size 2 \
    --bench
```

We will go with a 1:3 split for trainer and inference GPUs.

```bash
uv run inference @ examples/wordle/rl/infer.toml
```

```bash
bash scripts/tmux.sh -s exp1 -o outputs1
```

```bash
uv run inference @ examples/wordle/rl/infer.toml
```

```bash
uv run rl \
    --trainer @ examples/wordle/rl/train.toml \
    --orchestrator @ examples/wordle/rl/orch.toml \
    --trainer-gpus 1 \
    --inference-gpus 3 \
    --log.level debug \
    --wandb.project mika \
    --wandb.name wordle-rl-1024 \
    --output-dir outputs1
```

We also try with lower batch size

```bash
bash scripts/tmux.sh -s exp2 -o outputs2
```

```bash
uv run inference @ examples/wordle/rl/infer.toml --server.port 8001
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run rl \
    --trainer @ examples/wordle/rl/train.toml \
    --orchestrator @ examples/wordle/rl/orch.toml \
    --trainer-gpus 1 \
    --inference-gpus 3 \
    --log.level debug \
    --wandb.project mika \
    --wandb.name wordle-rl-512 \
    --orchestrator.batch-size 512 \
    --orchestrator.client.base-url http://localhost:8001/v1 \
    --output-dir outputs2
```