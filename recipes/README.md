# Recipes

Tested configs, one per model, entrypoint and hardware type. Each recipe fits 256k context,
so it works for any agentic task. A recipe sets the deployment and the parallelism. It sets
no task or data.

| Model | Entrypoint | Hardware | Trainer | Inference replica | Trainer tok/s | Status |
|---|---|---|---|---|---|---|
| [Qwen3.6-35B-A3B](qwen3.6-35b-a3b/) | `rl` | 8x H200 | 1 node (ep 8, cp 4) | 3 × 1 node (tp 2, dp 4) | 21.7k (scaleswe) | 5-step smoke run |

## Run an RL recipe

An RL recipe needs a task config from `rl/tasks/`. Put the recipe first and the task second:

```bash
uv run rl @ recipes/qwen3.6-35b-a3b/rl/h200.toml @ recipes/rl/tasks/swe.toml
```

Each RL recipe sets the inference replicas that kept the trainer busy on its smoke task.
Change them with `--deployment.num-infer-replicas N`.

Hosted training (`prime train run`) takes one file. A recipe and a task never set the same
table, so you can join them:

```bash
cat recipes/qwen3.6-35b-a3b/rl/h200.toml recipes/rl/tasks/swe.toml > run.toml
```

## Files

- `<model>/<entrypoint>/<hardware>.toml`: the recipe.
- `rl/tasks/`: RL task configs (sources and algorithm) that compose with any RL recipe.

The rules and the fitting method are in [`skills/recipes/SKILL.md`](../skills/recipes/SKILL.md).
