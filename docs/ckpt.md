# Checkpointing

Our codebase supports checkpointing. Because of the trainer/ orchestrator design, as well as the natural asynchrony checkpointing is non-standard.

- Trainer (`src/prime_rl/trainer/ckpt.py`): Checkpoints FSDP model shard, optimizer state and progress (training step, total samples, total tokens)
- Orchestrator (`src/prime_rl/orchestrator/ckpt.py`): Checkpoints orchestrator progress

*NB: Each run with asynchrony level `async_level` and some checkpoint step `x`, requires weight checkpoints in the step range `[x-async_level, x]`. Currently we do not duplicate weight checkpoints into the `checkpoints` directory but simply keep them around in `weights`, by keeping the trainer from cleaning up weight checkpoints that are required for resuming training. This way, the orchestrator only needs to checkpoint its progress (read: step) to load the correct weights into the inference engine upon resuming.*

The default checkpoint directory is `checkpoints` and each checkpoint step will live in a subdirectory enumerated by the step, i.e. `checkpoints/step_{step}`. The trainer checkpoint is called `trainer.pt` for single GPU workloads, else `trainer_{local_rank}.pt`. The orchestrator checkpoint is called `orchestrator.pt`. Thus, this is a typical directory structure:

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

Checkpointing is configured by the `CheckpointConfig`, with the config key `--ckpt`. One can specify the interval (`--ckpt.interval`, defaults to `50`), whether to save checkpoints asynchronoously  (`--ckpt.save-async`, defaults to `False`), and how many recent step checkpoints to keep on disk (`--ckpt.keep`, defaults to `None` which means no cleanup).

By default, runs do no write checkpoints to save disk space. To checkpoint every 10 steps on our debug RL run, run the following command

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml --ckpt.interval 10 
```

To resume a run use the `--ckpt.resume-step` flag. To resume from the checkpoint step 10 from the previous command, run the following command

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml --ckpt.resume-step 10
```

Because we save progress information, resuming from a checkpoint is fully W&B compatible. By default, resuming from a checkpoint, will simply create a new run. To resume the same W&B run, you'd have to pass the same W&B run ID for both the trainer and the orchestrator, e.g.

```bash
CUDA_VISIBLE_DEVICES=1 uv run trainer @ configs/reverse_text/train.toml \
  --monitor.wandb.project <project> \
  --ckpt.resume-step 10 \
  --monitor.wandb.id <trainer-run-id> \
```

You also need to restart the orchestrator from a checkpoint, the api is the same as the trainer, e.g.

```bash
uv run orchestrator @ configs/reverse_text/orch.toml \
  --monitor.wandb.project <project> \
  --ckpt.resume-step 10 \
  --monitor.wandb.id <orchestrator-run-id>
```

If you started your run using `rl.py`, you can resume the same run by passing the same W&B run ID for both the trainer and the orchestrator, e.g.

```bash
uv run rl \
  --trainer @ configs/reverse_text/train.toml \
  --orchestrator @ configs/reverse_text/orch.toml \
  --ckpt.resume-step 10 \
  --trainer.monitor.wandb.id <trainer-run-id> \
  --orchestrator.monitor.wandb.id <orchestrator-run-id> 
```

You don't need to restart the inference server if started manually, the orchestrator will automatically send the right checkpoint to the inference server when resuming.