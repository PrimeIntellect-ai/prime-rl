# Recipes

Tested RL configs, one per model and hardware type. Each recipe fits 256k context, so it
works for any agentic task. A recipe sets the deployment, the trainer parallelism and the
inference engine. It sets no task: you add sources, algorithm and optimizer with a task config.

| Model | Hardware | Trainer | Inference replica | Trainer tok/s | Status |
|---|---|---|---|---|---|
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/) | 8x H200 | 1 node (ep 8, cp 4) | 3 × 1 node (tp 2, dp 4) | 21.7k (scaleswe) | 5-step smoke run |

## Run a recipe

Put the recipe first and the task second:

```bash
uv run rl @ recipes/qwen3.6-35b-a3b/h200.toml @ recipes/tasks/swe.toml
```

Each recipe sets the inference replicas that kept the trainer busy on its smoke task. Change
them with `--deployment.num-infer-replicas N`.

Hosted training (`prime train run`) takes one file. A recipe and a task never set the same
table, so you can join them:

```bash
cat recipes/qwen3.6-35b-a3b/h200.toml recipes/tasks/swe.toml > run.toml
```

## Files

- `<model>/<hardware>.toml`: the recipe.
- `tasks/`: task configs that compose with any recipe.

The rules, the fitting method and the measurements behind each recipe are in
[`skills/recipes/SKILL.md`](../skills/recipes/SKILL.md).
