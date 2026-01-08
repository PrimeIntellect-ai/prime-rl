
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

## Single-turn math

```bash
# baseline: synchronous on-policy RL
uv run rl @ configs/thesis/single_turn_math.toml --max-async-level 0 --wandb.name single-turn-math-sync-on-policy

# method 1: fixed-off policy steps
uv run rl @ configs/thesis/single_turn_math.toml --orchestrator.strict-async-level --max-async-level 1 --wandb.name single-turn-math-1-step-off-policy
uv run rl @ configs/thesis/single_turn_math.toml --orchestrator.strict-async-level --max-async-level 8 --wandb.name single-turn-math-8-step-off-policy
uv run rl @ configs/thesis/single_turn_math.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name single-turn-math-16-step-off-policy

# method 2: areal/ pipelinerl /w in-flight weight updates
uv run rl @ configs/thesis/single_turn_math.toml --wandb.name single-turn-math-in-flight-off-policy
```

## Wiki Search

```bash
# baseline: synchronous on-policy RL
uv run rl @ configs/thesis/wiki_search.toml --max-async-level 0 --wandb.name wiki-search-sync-on-policy

# method 1: fixed-off policy steps
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 1 --wandb.name wiki-search-1-step-off-policy
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 8 --wandb.name wiki-search-8-step-off-policy
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy

# method 2: areal/ pipelinerl /w in-flight weight updates
uv run rl @ configs/thesis/wiki_search.toml --wandb.name wiki-search-in-flight-off-policy
```