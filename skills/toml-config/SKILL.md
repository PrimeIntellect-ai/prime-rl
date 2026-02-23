---
name: toml-config
description: How to write and use TOML configs in prime-rl. Use when creating config files, running commands with configs, or overriding config values via CLI.
---

# TOML Config

All prime-rl commands use pydantic-settings with TOML configs and CLI overrides.

## Running with configs

```bash
# Load a config file with @ syntax
uv run inference @ configs/debug/infer.toml
uv run sft @ configs/debug/sft/train.toml
uv run rl @ configs/debug/rl/train.toml

# CLI overrides (take precedence over TOML)
uv run inference @ config.toml --model.name Qwen/Qwen3-0.6B --server.port 8001

# Boolean flags: no value needed
uv run inference --model.enforce_eager          # sets to true
uv run inference --no-model.enforce_eager       # sets to false

# CLI-only (no TOML file)
uv run inference --model.name Qwen/Qwen3-0.6B --model.max_model_len 2048
```

## TOML structure

Top-level fields must come before any `[section]` header — this is a TOML rule.

```toml
# Top-level fields first
gpu_memory_utilization = 0.5
seed = 42

# Then sections
[model]
name = "Qwen/Qwen3-0.6B"
max_model_len = 4096

[server]
port = 8000
```

Putting a top-level field after a section header nests it inside that section, which causes validation errors.

## Config inheritance

Configs can inherit from other TOML files:

```toml
toml_files = ["base.toml"]

[model]
name = "Qwen/Qwen3-0.6B"  # overrides base
```

Paths in `toml_files` are relative to the file containing the field.

## Setting None

Use the string `"None"` in TOML to set a field to None:

```toml
max_model_len = "None"
```

## SLURM mode

The `rl` command supports SLURM execution via an optional `[slurm]` section. When present, the run is submitted as a SLURM job instead of running locally.

```toml
output_dir = "/shared/experiments/my-run"

[slurm]
job_name = "my-rl-job"
num_train_nodes = 2
num_infer_nodes = 1
gpus_per_node = 8
# dry_run = true          # generate script without submitting
# slurm_template = "path/to/custom.sh.j2"
# nodes_per_fsdp_group = 1
# project_dir = "/path/to/project"
# hf_hub_offline = false
```

When `[slurm]` is set:
- `output_dir` must be explicitly set (the default `outputs` is rejected)
- Teacher inference (`teacher_gpu_ids`, `teacher_inference`) is not supported
- Local-only fields (`inference_gpu_ids`, `trainer_gpu_ids`, `clean`, `bench`) are ignored

## Available commands

All accept `@ config.toml` and CLI overrides:

| Command | Config class | Description |
|---------|-------------|-------------|
| `uv run rl` | full RL pipeline | Orchestrator + inference + trainer (local or SLURM) |
| `uv run inference` | `InferenceConfig` | vLLM inference server |
| `uv run trainer` | trainer config | RL trainer |
| `uv run orchestrator` | orchestrator config | Rollout orchestrator |
| `uv run env-server` | env server config | Environment server |
| `uv run sft` | SFT config | Supervised fine-tuning |

## Key files

- `src/prime_rl/utils/pydantic_config.py` — `parse_argv`, `BaseSettings`, `@` syntax parsing
- `src/prime_rl/rl.py` — unified RL entrypoint (local + SLURM)
- `src/prime_rl/rl_config.py` — `BaseRLConfig`, `SlurmConfig`, `write_subconfigs`
- `configs/` — all config files, organized by task
