# GPT-OSS

This 20-step RL smoke run uses `unsloth/gpt-oss-20b-BF16` on eight H200 GPUs: four for training with EP=4 and four for inference. It exercises FlashAttention 4 with learned sinks, QKV and gate/up fusion, compilation, selective activation checkpointing, and NCCL policy reloads.

```bash
uv run rl @ examples/advanced/gpt-oss/rl.toml
```

For Slurm, add your cluster settings to the config:

```toml
[slurm]
partition = "all"
nodelist = "YOUR_H200_NODE"
time = "01:00:00"
```

The same command then submits the Slurm job. Run it from the repository root after installing the project and its GPU extras.

Rollout concurrency starts at 32 and is capped at 64 to bound local subprocess use. Inference DP is set explicitly so the NCCL broadcast group contains all four inference workers.

The reverse-text task provides a small workload for measuring `mismatch_kl/all/mean` in the trainer metrics. The 512-token completion budget can truncate reasoning; this configuration is intended for numerical consistency checks. For longer completions, increase both the completion budget and the sequence limits.

Use a BF16 checkpoint. Loading the original MXFP4 weights is not supported by this native training implementation.
