# Iterative Training Guide: Building on Trained Models

This guide explains how to use a completed training run as a base for future experiments with different datasets or reward functions.

## Overview

After your current run completes (reaches step 1000), you can:
1. **Extract the final checkpoint** to HuggingFace format
2. **Start a new training run** using that checkpoint as the base model
3. **Modify datasets/reward functions** in your config
4. **Only adopt changes** that improve model performance

## Step 1: Extract Final Checkpoint to HuggingFace Format

After your run completes, convert the final checkpoint to HuggingFace format:

```bash
cd /workspace/prime-rl

# Find the final checkpoint step
FINAL_STEP=$(ls -1 outputs/dakota-v12-1000steps/checkpoints/ | grep step_ | sed 's/step_//' | sort -n | tail -1)
echo "Final step: $FINAL_STEP"

# Extract to HuggingFace format
python scripts/extract_hf_from_ckpt.py \
  --ckpt-dir outputs/dakota-v12-1000steps/checkpoints/step_${FINAL_STEP}/trainer \
  --output-dir outputs/dakota-v12-1000steps/hf_model \
  --utils-repo-id Qwen/Qwen3-0.6B \
  --dtype bfloat16
```

This creates a HuggingFace-compatible model at `outputs/dakota-v12-1000steps/hf_model/`.

## Step 2: Upload to HuggingFace Hub (Optional)

If you want to share or version your model:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Upload the model
cd outputs/dakota-v12-1000steps/hf_model
huggingface-cli upload YOUR_USERNAME/dakota-v12-1000steps . --repo-type model
```

Or keep it locally - you can reference it by path in your config.

## Step 3: Create New Experiment Configs

Create new config files for your next experiment:

```bash
mkdir -p configs/dakota_grammar_gym/rl/experiment_v2
```

### Option A: Use Local Checkpoint Path

**`configs/dakota_grammar_gym/rl/experiment_v2/train.toml`:**
```toml
max_steps = 2000  # New max steps

[model]
name = "/workspace/prime-rl/outputs/dakota-v12-1000steps/hf_model"  # Local checkpoint path

[optim]
lr = 1e-6  # Lower learning rate for fine-tuning

[wandb]
project = "dakota-grammar-gym"
name = "dakota-v13-2000steps-experiment-v2"

[ckpt]
interval = 50
keep = 5
```

**`configs/dakota_grammar_gym/rl/experiment_v2/orch.toml`:**
```toml
batch_size = 128
rollouts_per_example = 16
seq_len = 2048
max_steps = 2000

[model]
name = "/workspace/prime-rl/outputs/dakota-v12-1000steps/hf_model"  # Same checkpoint path

[sampling]
max_tokens = 2048

[[env]]
id = "harleycooper/dakota1890"
# You can modify environment args here to change reward function behavior
# args = { reward_scale = 2.0, difficulty = "hard" }  # Example

[wandb]
project = "dakota-grammar-gym"
name = "dakota-v13-2000steps-experiment-v2-orchestrator"

[ckpt]
interval = 50
keep = 5
```

**`configs/dakota_grammar_gym/rl/experiment_v2/infer.toml`:**
```toml
[model]
name = "/workspace/prime-rl/outputs/dakota-v12-1000steps/hf_model"  # Same checkpoint path
```

### Option B: Use HuggingFace Hub Model

If you uploaded to HuggingFace Hub:

```toml
[model]
name = "YOUR_USERNAME/dakota-v12-1000steps"  # HuggingFace repo ID
```

## Step 4: Modify Datasets or Reward Functions

### Changing the Environment/Dataset

To use a different dataset or environment:

**In `orch.toml`:**
```toml
[[env]]
id = "harleycooper/dakota1890"  # Change to different environment
args = { version = "0.1.13" }  # Or modify environment args
```

Or use multiple environments:
```toml
[[env]]
id = "harleycooper/dakota1890"
args = { difficulty = "easy" }

[[env]]
id = "harleycooper/dakota1890"
args = { difficulty = "hard" }
```

### Changing Reward Functions

The reward function is typically defined within the environment package (`dakota1890`). To change it:

1. **Modify the environment package** and republish to Prime Intellect
2. **Or use environment args** to change reward behavior (if supported):
   ```toml
   [[env]]
   id = "harleycooper/dakota1890"
   args = { 
     reward_function = "custom",
     reward_scale = 1.5,
     # ... other reward-related args
   }
   ```

## Step 5: Launch New Training Run

Start your new experiment:

```bash
cd /workspace/prime-rl
export PATH="$HOME/.local/bin:$PATH"

uv run rl \
  --trainer @ configs/dakota_grammar_gym/rl/experiment_v2/train.toml \
  --orchestrator @ configs/dakota_grammar_gym/rl/experiment_v2/orch.toml \
  --inference @ configs/dakota_grammar_gym/rl/experiment_v2/infer.toml \
  --inference-gpu-ids 0,1,2,3,4,5 \
  --trainer-gpu-ids 6,7 \
  --inference.parallel.dp 6 \
  --output-dir outputs/dakota-v13-2000steps-experiment-v2
```

## Step 6: Compare Performance

Monitor both runs in WandB to compare:
- **Baseline**: `dakota-v12-1000steps-trainer`
- **Experiment**: `dakota-v13-2000steps-experiment-v2`

Compare metrics:
- Mean reward
- Reward trend over time
- Sample quality
- Training stability

## Step 7: Adopt Only Improvements

If the new experiment shows improvement:
1. **Keep the new checkpoint** as your new baseline
2. **Document what changed** (dataset, reward function, hyperparameters)
3. **Start next iteration** from this improved checkpoint

If the new experiment doesn't improve:
1. **Revert to previous checkpoint**
2. **Try different modifications** (adjust learning rate, change reward function differently, etc.)
3. **Iterate until you find improvements**

## Best Practices

1. **Always checkpoint frequently** - Set `ckpt.interval` to save regularly
2. **Keep multiple checkpoints** - Use `ckpt.keep` to maintain history
3. **Version your configs** - Use descriptive names like `experiment_v2`, `experiment_v3`
4. **Track in WandB** - Compare runs systematically
5. **Document changes** - Keep notes on what you changed and why
6. **Lower learning rate** - When fine-tuning from a checkpoint, use a lower LR (e.g., 1e-6 instead of 3e-6)

## Quick Reference: Key Config Changes

| What to Change | Where | Example |
|---------------|-------|---------|
| Base model | `train.toml`, `orch.toml`, `infer.toml` → `[model] name` | `name = "/path/to/checkpoint"` |
| Dataset/Environment | `orch.toml` → `[[env]] id` | `id = "harleycooper/dakota1890"` |
| Reward function | `orch.toml` → `[[env]] args` | `args = { reward_scale = 2.0 }` |
| Learning rate | `train.toml` → `[optim] lr` | `lr = 1e-6` |
| Max steps | `train.toml`, `orch.toml` → `max_steps` | `max_steps = 2000` |

## Example Workflow

```bash
# 1. Current run completes at step 1000
# 2. Extract checkpoint
python scripts/extract_hf_from_ckpt.py \
  --ckpt-dir outputs/dakota-v12-1000steps/checkpoints/step_1000/trainer \
  --output-dir outputs/dakota-v12-1000steps/hf_model \
  --utils-repo-id Qwen/Qwen3-0.6B

# 3. Create new configs pointing to checkpoint
# (Edit configs as shown above)

# 4. Launch new experiment
uv run rl \
  --trainer @ configs/dakota_grammar_gym/rl/experiment_v2/train.toml \
  --orchestrator @ configs/dakota_grammar_gym/rl/experiment_v2/orch.toml \
  --inference @ configs/dakota_grammar_gym/rl/experiment_v2/infer.toml \
  --inference-gpu-ids 0,1,2,3,4,5 \
  --trainer-gpu-ids 6,7 \
  --inference.parallel.dp 6 \
  --output-dir outputs/dakota-v13-2000steps-experiment-v2

# 5. Compare in WandB and decide whether to adopt
```

