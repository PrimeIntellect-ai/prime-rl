# glyphbench.plotting

Plotting utilities for benchmark results.

## Architecture

This module does NOT import from `runner`, `providers`, `harness`, or `envs`.

## Modules

### style.py

Matplotlib rcParams for NeurIPS-style figures.

```python
from glyphbench.plotting.style import plot_style

with plot_style():      # scoped context manager
    plt.plot(...)
```

Key settings: Computer Modern serif, 300 DPI save, no top/right spines, subtle grid.

### violins.py

Violin plots are the default notebook inspection view for GlyphBench returns.
They show the full rollout distribution, keep the y-axis fixed to the task
contract range, and make suite/task variance visible without hiding outliers.
The single public notebook entry point is
[`notebooks/return_violins.ipynb`](../../../notebooks/return_violins.ipynb).

```python
from glyphbench.plotting import plot_return_violins
from glyphbench.plotting.rollouts import load_rollouts

df = load_rollouts("runs/qwen35-single-task")
fig, ax = plot_return_violins(df, group_col="suite")
fig.savefig("runs/qwen35-single-task/figures/return_violins.png")
```

For per-task inspection, use `group_col="env_id"` and filter to one suite first
so labels stay readable.

## Contract

- All functions accept string paths, not Path objects, for CLI ergonomics.
- Plots use `PLOT_STYLE` via the `plot_style()` context manager.
- Notebook return inspection should prefer violin plots over single-value bars
  when rollout-level records are available.
