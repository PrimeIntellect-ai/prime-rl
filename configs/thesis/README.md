
## Debug

Run the debug config, to try out the different modes

```bash
# baseline: synchronous on-policy RL
uv run rl @ configs/thesis/debug.toml --max-async-level 0

# method 1: fixed-off policy steps
uv run rl @ configs/thesis/debug.toml --orchestrator.strict-async-level --max-async-level 1
uv run rl @ configs/thesis/debug.toml --orchestrator.strict-async-level --max-async-level 8
uv run rl @ configs/thesis/debug.toml --orchestrator.strict-async-level --max-async-level 16

# method 2: areal/ pipelinerl /w in-flight weight updates
uv run rl @ configs/thesis/debug.toml
```