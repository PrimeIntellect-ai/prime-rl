
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

# fixed off-policy steps without any masking
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 1 --wandb.name wiki-search-1-step-off-policy-no-mask --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 8 --wandb.name wiki-search-8-step-off-policy-no-mask --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-no-mask --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10

# fixed-off policy steps
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 1 --wandb.name wiki-search-1-step-off-policy
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 8 --wandb.name wiki-search-8-step-off-policy
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy

# method 2: areal/ pipelinerl /w in-flight weight updates
uv run rl @ configs/thesis/wiki_search.toml --wandb.name wiki-search-in-flight-off-policy
```

got crashing run with 16 step off-policy and otherwise defaults.

```bash
# token mask
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-token-mask-0.25-4 --trainer.loss.token-mask-low 0.25 --trainer.loss.token-mask-high 4
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-token-mask-0.5-2 --trainer.loss.token-mask-low 0.5 --trainer.loss.token-mask-high 2

# sequence masking

# kl tau
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-no-mask-kl-tau-1e-4 --trainer.loss.kl-tau 1e-4 --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-no-mask-kl-tau-2e-3 --trainer.loss.kl-tau 2e-3 --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-no-mask-kl-tau-1e-2 --trainer.loss.kl-tau 1e-2 --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-no-mask-kl-tau-1e-1 --trainer.loss.kl-tau 1e-1 --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-no-mask-kl-tau-1 --trainer.loss.kl-tau 1 --trainer.loss.token-mask-low 0 --trainer.loss.token-mask-high 1e10 --trainer.loss.sequence-mask-low 0 --trainer.loss.sequence-mask-high 1e10 --trainer.loss.geo-mask-low 0 --trainer.loss.geo-mask-high 1e10 --trainer.loss.sequence-clip-high 1e10

# kl tau + icepop (1/8, 8)
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-kl-tau-1e-4 --trainer.loss.kl-tau 1e-4
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-kl-tau-2e-3 --trainer.loss.kl-tau 2e-3
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-kl-tau-1e-2 --trainer.loss.kl-tau 1e-2
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-kl-tau-1e-1 --trainer.loss.kl-tau 1e-1
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 16 --wandb.name wiki-search-16-step-off-policy-kl-tau-1 --trainer.loss.kl-tau 1
```

to try later:
- sequence masking only
- geometric sequence masking
- combine masking strategies

```bash
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 32 --wandb.name wiki-search-32-step-off-policy-kl-tau-1e-2 --trainer.loss.kl-tau 1e-2
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 64 --wandb.name wiki-search-64-step-off-policy-kl-tau-1e-2 --trainer.loss.kl-tau 1e-2
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 128 --wandb.name wiki-search-128-step-off-policy-kl-tau-1e-2 --trainer.loss.kl-tau 1e-2
```

```bash
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 32 --wandb.name wiki-search-32-step-off-policy-kl-tau-1e-1 --trainer.loss.kl-tau 1e-1
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 64 --wandb.name wiki-search-64-step-off-policy-kl-tau-1e-1 --trainer.loss.kl-tau 1e-1
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 128 --wandb.name wiki-search-128-step-off-policy-kl-tau-1e-1 --trainer.loss.kl-tau 1e-1
```

```bash
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 32 --wandb.name wiki-search-32-step-off-policy-kl-tau-5e-2 --trainer.loss.kl-tau 5e-2
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 64 --wandb.name wiki-search-64-step-off-policy-kl-tau-5e-2 --trainer.loss.kl-tau 5e-2
uv run rl @ configs/thesis/wiki_search.toml --orchestrator.strict-async-level --max-async-level 128 --wandb.name wiki-search-128-step-off-policy-kl-tau-5e-2 --trainer.loss.kl-tau 5e-2
```
