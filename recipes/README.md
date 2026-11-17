# Recipes

Tested RL configs, one per model and hardware type. Each recipe fits 256k context, so it
works for any agentic task. A recipe sets the deployment, the trainer parallelism and the
inference engine. It sets no task: you add sources, algorithm and optimizer with a task config.

| Model | Hardware | Trainer | Inference replica | Trainer tok/s | Status |
|---|---|---|---|---|---|
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/) | 8x H200 | 1 node (ep 8, cp 4) | 1 node (tp 2, dp 4), ~3 per trainer node | 21.7k (scaleswe) | 5-step smoke run |

## Run a recipe

Put the recipe first and the task second:

```bash
uv run rl @ recipes/qwen3.6-35b-a3b/h200.toml @ recipes/tasks/swe.toml
```

Each recipe starts one inference replica. Add replicas with `--deployment.num-infer-replicas N`.
Agentic tasks spend most of their time in tool calls, so they usually need several replicas per
trainer node; `meta.toml` gives the measured suggestion.

Hosted training (`prime train run`) takes one file. A recipe and a task never set the same
table, so you can join them:

```bash
cat recipes/qwen3.6-35b-a3b/h200.toml recipes/tasks/swe.toml > run.toml
```

## Files

- `<model>/<hardware>.toml`: the recipe.
- `<model>/meta.toml`: what was measured, on which commit, and the results file.
- `tasks/`: task configs that compose with any recipe.

The sweep harness and the raw results are in [`benchmarks/recipes/`](../benchmarks/recipes).
The rules and the fitting playbook are in [`skills/recipes/SKILL.md`](../skills/recipes/SKILL.md).
