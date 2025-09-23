# Checkpointing

Checkpointing is non-standard due to trainer/orchestrator separation and natural asynchrony.

- SFT+RL Trainer: Checkpoints FSDP model shard (using DCP), optimizer and scheduler state, and progress (training step, total samples, total tokens)
- Orchestrator: Checkpoints orchestrator progress (training step, total tokens, total samples, total problems)
- Inference: Inference is stateless. Upon restart, the orchestrator will reload the correct weights into the inference engine. No checkpointing is required.

The default checkpoint directory is `checkpoints` and each checkpoint step will live in a step subdirectory, i.e. `checkpoints/step_{step}`.

Checkpointing is configured with the config key `--ckpt`. One can specify the interval (`--ckpt.interval`), whether to save checkpoints asynchronoously  (`--ckpt.save-async`), and how many recent step checkpoints to keep on disk (`--ckpt.keep`). By default, we do not checkpoint to save disk space. 

## SFT

Let's split the reverse text training SFT example, which does 40 steps by default, into two runs of 20 steps each. 

First, run the first 20 steps and append  `--ckpt` flag will enable the default checkpoint configuration which will only write the final checkpoint to disk, but no intermediate checkpoints.

```bash
uv run sft ... --max-steps 20 --ckpt
```

Then, to resume the training from step 20, run the following command

```bash
uv run sft ... --max-steps 40 --ckpt.resume-step 20
```

## RL

Similarly, let's split the reverse text training RL example, which does 20 steps by default, into two runs of 10 steps each. 

First, start the inference server. It can stay running across restarts as the orchestrator will automatically send the right checkpoint to the inference server when resuming.

```bash
uv run inference ...
```

Then, run the first 20 steps and write the final checkpoint to disk

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --max-steps 10 \
  --ckpt
```

And finally, resume the training to do the remaining 10 steps

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --max-steps 20 \
  --ckpt.resume-step 10
```