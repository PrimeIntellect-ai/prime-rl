# RL Training Setup Guide for Dakota1890 Environment

This guide documents the successful setup and launch of RL training for the `harleycooper/dakota1890` environment using Prime Intellect's prime-rl framework.

## Overview

Successfully configured and launched a 1000-step RL training run using:
- **Environment**: `harleycooper/dakota1890` (version 0.1.12)
- **Model**: `Qwen/Qwen3-0.6B`
- **GPUs**: 8x A100 80GB (6 for inference, 2 for training)
- **Framework**: prime-rl

## Prerequisites

1. Prime Intellect GPU instance with at least 2 GPUs
2. Prime RL (RFT) image deployed
3. SSH access configured with public key

## Setup Steps

### 1. SSH into Instance

```bash
ssh -i /path/to/your/private_key root@<INSTANCE_IP> -p <PORT>
```

### 2. Navigate to prime-rl Directory

```bash
cd /workspace/prime-rl
```

### 3. Install Environment

The environment can be installed using either method:

**Option A: Using Prime CLI (Recommended)**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Prime CLI
uv tool install prime

# Install environment
prime env install harleycooper/dakota1890
```

**Option B: Using uv pip directly**
```bash
cd /workspace/prime-rl
uv pip install dakota1890 --extra-index-url https://hub.primeintellect.ai/harleycooper/simple/
```

**Note**: If `prime env install` fails with build dependency errors, use Option B.

### 4. Verify Environment Installation

```bash
cd /workspace/prime-rl
export PATH="$HOME/.local/bin:$PATH"
uv run python -c "from verifiers import load_environment; env = load_environment('harleycooper/dakota1890'); print('✓ Environment loaded successfully')"
```

### 5. Create Configuration Files

Create the following directory structure:
```bash
mkdir -p configs/dakota_grammar_gym/rl
```

#### `configs/dakota_grammar_gym/rl/train.toml`
```toml
max_steps = 1000

[model]
name = "Qwen/Qwen3-0.6B"

[optim]
lr = 3e-6
```

#### `configs/dakota_grammar_gym/rl/orch.toml`
```toml
batch_size = 128
rollouts_per_example = 16
seq_len = 2048
max_steps = 1000
mask_truncated_completions = false

[model]
name = "Qwen/Qwen3-0.6B"

[sampling]
max_tokens = 2048

[[env]]
id = "harleycooper/dakota1890"
```

#### `configs/dakota_grammar_gym/rl/infer.toml`
```toml
[model]
name = "Qwen/Qwen3-0.6B"
```

**Important**: All string values in TOML files must be quoted (e.g., `name = "Qwen/Qwen3-0.6B"` not `name = Qwen/Qwen3-0.6B`).

### 6. Launch RL Training

For 8 GPUs (6 for inference, 2 for training):

```bash
cd /workspace/prime-rl
export PATH="$HOME/.local/bin:$PATH"

uv run rl \
  --trainer @ configs/dakota_grammar_gym/rl/train.toml \
  --orchestrator @ configs/dakota_grammar_gym/rl/orch.toml \
  --inference @ configs/dakota_grammar_gym/rl/infer.toml \
  --inference-gpu-ids 0,1,2,3,4,5 \
  --trainer-gpu-ids 6,7 \
  --inference.parallel.dp 6 \
  --output-dir outputs/dakota-v12-1000steps
```

For 2 GPUs (1 for inference, 1 for training):

```bash
uv run rl \
  --trainer @ configs/dakota_grammar_gym/rl/train.toml \
  --orchestrator @ configs/dakota_grammar_gym/rl/orch.toml \
  --inference @ configs/dakota_grammar_gym/rl/infer.toml \
  --inference-gpu-ids 0 \
  --trainer-gpu-ids 1 \
  --output-dir outputs/dakota-v12-1000steps
```

### 7. Monitor Training

**View live logs:**
```bash
tail -f outputs/dakota-v12-1000steps/logs/orchestrator.stdout
```

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check training progress:**
```bash
ls -la outputs/dakota-v12-1000steps/weights/
```

## Troubleshooting

### Issue: TOML Parse Error - "Invalid value"

**Symptom**: Error like `tomli._parser.TOMLDecodeError: Invalid value (at line 4, column 8)`

**Solution**: Ensure all string values in TOML files are properly quoted:
- ✅ Correct: `name = "Qwen/Qwen3-0.6B"`
- ❌ Wrong: `name = Qwen/Qwen3-0.6B`

### Issue: Environment Not Found

**Symptom**: `ValueError: Could not import 'harleycooper/dakota1890' environment`

**Solution**: 
1. Verify installation: `uv pip list | grep dakota1890`
2. Reinstall: `uv pip install dakota1890 --extra-index-url https://hub.primeintellect.ai/harleycooper/simple/`
3. Test import: `uv run python -c "import dakota1890"`

### Issue: CUDA Out of Memory

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory`

**Solution**: 
- Ensure GPUs are properly allocated (inference and trainer on separate GPUs)
- Reduce `batch_size` or `rollouts_per_example` in `orch.toml`
- Check for other processes using GPU memory: `nvidia-smi`

### Issue: Prime CLI Installation Fails

**Symptom**: `error: No virtual environment found` when running `prime env install`

**Solution**: Use `uv pip install` directly instead:
```bash
cd /workspace/prime-rl
uv pip install dakota1890 --extra-index-url https://hub.primeintellect.ai/harleycooper/simple/
```

## Expected Output

When running successfully, you should see orchestrator logs like:

```
INFO Starting orchestrator step X
INFO Waiting for weight checkpoint Y
INFO Updating weights to weight checkpoint Y
SUCCESS Step X | Time: 13.04s | Reward: 0.1160 | Throughput: 4577.7 tokens/s | Seq. Length: 396.8 tokens/sample
```

## Configuration Notes

- **max_steps**: Set to 1000 for full training run
- **batch_size**: 128 (adjust based on GPU memory)
- **rollouts_per_example**: 16 (number of rollouts per training example)
- **seq_len**: 2048 (maximum sequence length)
- **lr**: 3e-6 (learning rate for optimizer)
- **model**: Using Qwen3-0.6B (adjust based on available GPUs)

## Files Created

- `configs/dakota_grammar_gym/rl/train.toml` - Trainer configuration
- `configs/dakota_grammar_gym/rl/orch.toml` - Orchestrator configuration  
- `configs/dakota_grammar_gym/rl/infer.toml` - Inference server configuration

## References

- [Prime RL Documentation](https://github.com/PrimeIntellect-ai/prime-rl)
- [Verifiers Framework](https://github.com/PrimeIntellect-ai/verifiers)
- [Dakota1890 Repository](https://github.com/HarleyCoops/Dakota1890)

## Success Indicators

✅ Environment loads without errors  
✅ Config files parse correctly  
✅ RL run starts and shows orchestrator steps  
✅ GPU memory is properly allocated  
✅ Logs show consistent reward values and throughput

