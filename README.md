<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<h3 align="center">
PRIME-RL: Decentralized RL Training at Scale
</h3>

---

## Installation

**Quick Installation (Recommended)**

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

<details>
<summary>
Manual Installation
</summary>

1. Clone the repository

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Synchronize the environment

```bash
uv sync && uv sync --extra fa
```

</details>


<details>
<summary>
Validate your environment setup
</summary>

1. Check that environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run training debug mode (*this requires 1 GPU*)

```bash
uv run trainer @ configs/trainer/debug.toml
```

4. Check that you can run the orchestrator against an inference server (*this requires 1 GPU*)

```bash
uv run inference @ configs/inference/debug.toml
```

```bash
uv run orchestrator @ configs/orchestrator/debug.toml
```

5. Check that you can run a toy RL run (*this requires 2 GPUs and lasts 5min, see more below*)

```bash
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --orchestrator @ configs/orchestrator/reverse_text.toml \
  --inference @ configs/inference/reverse_text.toml
```

</details>


## Entrypoints

### Setup

1. If you want to log your runs to W&B (`wandb`), log in

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

2. If you require gated/ private models or datasets from HuggingFace, log in

```bash
uv run huggingface-cli login
# Or set `export HF_TOKEN=...`
```

3. You may want to increase the number of open files to prevent errors like `Too many open files`.

```bash
ulimit -n 32000
```

### RL

We provide a convenience endpoint `rl` for single-node RL experiments. It configures and startsthe trainer, orchestrator and, optionally, an inference server. It enforces correctly setting shared configs (e.g. the model name or async level should be the same across all modules) and dispatches and monitors subprocesses. To stream the logs from each module, we use file logging which can be automatically viewed from a `tmux` layout defined in `.tmuxinator.yaml`. The recommended workflow is:

1. Start a pre-layouted `tmux` session using `tmuxinator`

```bash
tmuxinator
```

2. Start the inference server separately in the `Inference` pane (to keep it alive across experiments with the same model)

```bash
uv run inference @ configs/inference/reverse_text.toml
```

3. Start the trainer and orchestrator in the `Trainer` pane.

```bash
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --orchestrator @ configs/orchestrator/reverse_text.toml
```


**Reverse Text**

Train a tiny model (`willcb/Qwen2.5-0.5B-Reverse-SFT`) to learn to reverse a small chunk of text. Training is extremely quick because we allow a maximum context of 128 tokens. 

```bash
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --orchestrator @ configs/orchestrator/reverse_text.toml \
  --inference @ configs/inference/reverse_text.toml
```

*With two small GPUs (e.g. RTX 3090/ 4090), this experiment should finish in less than 5 minutes.*

**Hendrycks Math**

Train a small model (`willcb/DeepSeek-R1-Distill-Qwen-1.5B`) on high-school level math questions. It is recommended to have at least 2xA100-80GB GPUs or more for this experiment.

On two GPUs, run the following command to run the experiment.

```bash
uv run rl \
  --trainer @ configs/trainer/hendrycks_math/1b.toml \
  --orchestrator @ configs/orchestrator/hendrycks_math/1b.toml \
  --inference @ configs/inference/hendrycks_math/1b.toml \
  --inference.parallel.dp 1
```

**INTELLECT-2 Math**

Train a small model (`willcb/DeepSeek-R1-Distill-Qwen-1.5B`) on complex math questions from the INTELLECT-2 dataset.

```bash
uv run rl \
  --trainer @ configs/trainer/intellect_math/1b.toml \
  --orchestrator @ configs/orchestrator/intellect_math/1b.toml \
  --inference @ configs/inference/intellect_math/1b.toml \
  --trainer-gpus 2 --inference-gpus 6
```

*NB: This setup requires 8 GPUs - 2 are used for the FSDP trainer, 6 are used for inference with TP=2 and DP=3.*


### Evals

*TBD*

### Synthetic Data

*TBD*

### Benchmark

*TBD*

## Developer

*For now, development is only possible on CUDA-enabled devices. However, we build production-ready images for both CUDA (NVIDIA) and ROCM (AMD) GPUs that should work out of the box.*

### Setup

1. Optionally, install [pre-commit](https://pre-commit.com) hooks

```bash
uv run pre-commit install
```

### Configs

We use `pydantic-settings` to configure `prime-rl`. To get an overview of the available configurations, run the following command:

```bash
uv run trainer --help
```

```bash
uv run orchestrator --help
```

```bash
uv run inference --help
```

**Sources**

We support the following sources for configuration, in this order of precedence:

1. **Command-line arguments**: You can pass (nested) arguments as `--key.subkey value` to the script. For example, to set the model name you can run `--model.name`

2. **Config files**: You can pass `.toml` config files (defined in the `configs` directory) using the `@` prefix. For example, to use the `debug.toml` config file, you can run `uv run inference @ configs/inference/debug.toml`. (*If you leave a space between the `@` and the config file, you will get shell path auto-completions.*)

3. **Environment variables**: You can set environment variables to override the config values. All environment variables must be prefixed with `PRIME_` and use the `__` delimiter to nest the keys. For example, to set the model name you can run `export PRIME_MODEL__NAME=Qwen/Qwen3-0.6B`.

4. **Defaults**: For almost all config arguments, we have a default value which will be used if no other source is provided.

In general we recommend setting configurations via config files to define reproducible experiments and use command-line arguments to override the config values to run variants of the same experiment. Environment variables are usually only used in production settings to communicate with the [Prime Protocol](https://github.com/PrimeIntellect-ai/protocol) worker. In most cases, you should not need to use environment variables.

The precendence order will be important if multiple sources try to configure the same argument. For example, in the following command, all sources will define a model name

```toml
# qwen8b.toml
[model]
name = "Qwen/Qwen3-8B"
```

```toml
# qwen14b.toml
[model]
name = "Qwen/Qwen-14B"
```

```bash
PRIME_MODEL__NAME=Qwen/Qwen3-4B uv run inference @qwen8b.toml @qwen14b.toml --model.name Qwen/Qwen3-32B
```

In this example, the CLI argument `--model.name Qwen/Qwen3-32B` will take precendence and the script will use `Qwen/Qwen3-32B` as the model name. If the CLI argument wasn't set, then the second config file would take precedence and the script would use `Qwen/Qwen-14B` as the model name. If the second config file wasn't set, then the first config file would take precedence and the script would use `Qwen/Qwen3-8B` as the model name. Finally, if the first config file wasn't set, then the environment variable would take precedence and the script would use `Qwen/Qwen-4B` as the model name. If the environment variable wasn't set, then the default value would be used and the script would use `Qwen/Qwen3-0.6B` as the model name.

### Checkpointing

Our codebase supports checkpointing. Because of the trainer/ orchestrator design, as well as the natural asynchrony checkpointing is non-standard.

- Trainer (`src/prime_rl/trainer/ckpt.py`): Checkpoints FSDP model shard, optimizer state and progress (training step, total samples, total tokens)
- Orchestrator (`src/prime_rl/trainer/ckpt.py`): Checkpoints orchestrator progress

*NB: Each run with asynchrony level `async_level` and some checkpoint step `x`, requires weight checkpoints in the step range `[x-async_level, x]`. Currently we do not duplicate weight checkpoints into the `checkpoints` directory but simply keep them around in `weights`, by keeping the trainer from cleaning up weight checkpoints that are required for resuming training. This way, the orchestrator only needs to checkpoint its progress (read: step) to load the correct weights into the inference engine upon resuming.*

The default checkpoint directory is `checkpoints` and each checkpoint step will live in a subdirectory enumerated by the step, i.e. `checkpoints/step_{step}`. The trainer checkpoint is called `trainer.pt` for single GPU workloads, else `trainer_{local_rank}.pt`. The orchestrator checkpoint is callec `orchestrator.pt`. Thus, this is a typical directory structure:

```bash
checkpoints
├── step_10
│   ├── orchestrator.pt
│   └── trainer.pt
├── step_25
│   ├── orchestrator.pt
│   └── trainer.pt
└── step_30
    ├── orchestrator.pt
    └── trainer.pt
```

Checkpointing is configured by the `CheckpointConfig`, with the config key `--ckpt`. One can specify:
- `--ckpt.path` to change the checkpoint directory (default: `checkpoints`)
- `--ckpt.interval` to change the interval frequency (default: `50`)
- `--ckpt.save-async` to save the checkpoint asynchronously (default: `False`)

By default, runs do no write checkpoints to save disk space. To checkpoint every 10 steps on our debug RL run, run the following command

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/trainer/reverse_text.toml --ckpt.interval 10 
```

To resume a run use the `--ckpt.resume-step` flag. To resume from the checkpoint stpe 10 from the previous command, run the following command

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/trainer/reverse_text.toml --ckpt.resume_step 10
```

Because we save progress information, resuming from a checkpoint is fully W&B compatible. By default, resuming from a checkpoint, will simply create a new run. To resume the same W&B run, you'd have to pass the same W&B run ID for both the trainer and the orchestrator, e.g.

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/trainer/reverse_text.toml \
  --monitor.wandb.project <project> \
  --monitor.wandb.group <group> \
  --ckpt.resume-step 10 \
  --monitor.wandb.id <trainer-run-id> \
  --orchestrator.monitor.wandb.id <orchestrator-run-id>
```


### Tests

Run the full test suite 

```bash
uv run pytest -v
```

To run unit tests, run

```bash
uv run pytest tests/unit -v
```

To run integration tests, run

```bash
uv run pytest tests/integration -v
```

To run CPU-only tests, use the inverse of the `gpu` marker:

```bash
uv run pytest -v -m "not gpu"
```

To run fast tests, use the inverse of the `slow` marker:

```bash
uv run pytest -v -m "not slow"
```

### Step Definition

At each step `n`, all artifacts (e.g., checkpoint, rollout, gradient) are tagged with the same step number.
- Step 0:
  - Uses checkpoint 0 on rollout 0 to compute gradient 0.
  - Then computes checkpoint 1 as: `ckpt 1 = ckpt 0 - grad 0`

In general, the model used for generating rollouts at step `n` is from `ckpt[n - async_level]`.

- When async_level = 0, the rollout and gradient are based on the same model version.
  This is equivalent to synchronous on-policy training.

## Citation

*TBD*