# Troubleshooting

> My API keeps timing out.

We already set much larger timeout limits for the API clients that we use for training and evals. If you still encounter API timeout or connection errors, then this may be caused by your OS limiting the number of open file descriptors. Try increasing the maximum number of open files with

```bash
ulimit -n 32000
```

> I'm getting CUDA out of memory errors.

Assuming this is happening on the RL or SFT trainer, you can try the following:
- Use full activation checkpointing (`--model.ac`)
- Reduce the the micro batch size (`--data.micro-batch-size`) and sequence length (`--data.seq-len`)
- (*Experimental*) Use context parallelism with `--model.cp`

> I cannot pass my TOML config file

Check that you didn't leave a whitespace between the `@` and the config file (e.g. `uv run ... @ path/to/config.toml` instead of `uv run ... @path/to/config.toml`). Also, make sure that your TOML config matches the configuration schema. If not, the Pydantic error message (which arguably is quite ugly) will hopefully point you in the right direction.

