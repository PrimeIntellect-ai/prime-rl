"""Violin plot helpers for notebook result inspection."""

from __future__ import annotations

from itertools import cycle
from typing import Any

_SUITE_COLORS = {
    "minigrid": "#7de38b",
    "minihack": "#64d9e8",
    "classics": "#f4c95d",
    "craftax": "#ff9f7f",
    "miniatari": "#ff7f9a",
    "procgen": "#c7d36f",
}
_FALLBACK_COLORS = (
    "#7de38b",
    "#64d9e8",
    "#f4c95d",
    "#ff9f7f",
    "#ff7f9a",
    "#c7d36f",
    "#c8c29f",
)


def plot_return_violins(
    df: Any,
    *,
    group_col: str = "suite",
    value_col: str = "episodic_return",
    phase: str | None = "eval",
    order: list[str] | None = None,
    ax: Any | None = None,
    title: str = "GlyphBench return distributions",
    show_points: bool = True,
) -> tuple[Any, Any]:
    """Plot return distributions as violins and return ``(fig, ax)``.

    ``df`` is usually the DataFrame returned by
    :func:`glyphbench.plotting.rollouts.load_rollouts`. The default view groups by
    suite and plots native episodic returns without clipping. Reward scales
    differ across environments, so compare scores within the same task.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import seaborn as sns  # type: ignore[import-not-found]

        from glyphbench.plotting.style import plot_style
    except ImportError as e:
        raise ImportError(
            "plot_return_violins requires the analysis extra: "
            "`uv sync --extra analysis`"
        ) from e

    work = df.copy()
    if phase is not None and "phase" in work.columns:
        work = work[work["phase"] == phase]
    missing = [col for col in (group_col, value_col) if col not in work.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")

    keep_cols = [group_col, value_col]
    if "suite" in work.columns and "suite" not in keep_cols:
        keep_cols.append("suite")
    work = work[keep_cols].dropna(subset=[group_col, value_col])
    work[value_col] = work[value_col].astype(float)
    if order is None:
        order = (
            work.groupby(group_col, dropna=False)[value_col]
            .median()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
    work[group_col] = work[group_col].astype(str)
    palette = _palette_for(work, group_col, order)

    with plot_style():
        if ax is None:
            fig_width = max(7.2, min(20.0, 0.28 * len(order)))
            fig, ax = plt.subplots(figsize=(fig_width, 4.2))
        else:
            fig = ax.figure
        fig.patch.set_facecolor("#0e100b")
        ax.set_facecolor("#12150d")
        sns.violinplot(
            data=work,
            x=group_col,
            y=value_col,
            order=order,
            hue=group_col,
            palette=palette,
            inner="quartile",
            cut=0,
            density_norm="width",
            linewidth=0.85,
            saturation=0.95,
            legend=False,
            ax=ax,
        )
        if show_points:
            sns.stripplot(
                data=work,
                x=group_col,
                y=value_col,
                order=order,
                color="#f4f1df",
                alpha=0.34,
                size=2.2,
                jitter=0.18,
                linewidth=0,
                ax=ax,
            )
        ax.axhline(0, color="#f4c95d", linewidth=0.9, linestyle="--", alpha=0.65)
        ax.set_xlabel(group_col.replace("_", " "))
        ax.set_ylabel(value_col.replace("_", " "))
        ax.set_title(title, color="#f4c95d", pad=12, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=25)
        ax.tick_params(axis="both", colors="#c8c29f")
        ax.xaxis.label.set_color("#f4f1df")
        ax.yaxis.label.set_color("#f4f1df")
        ax.grid(True, axis="y", color="#5b5a31", alpha=0.36, linewidth=0.6)
        ax.grid(False, axis="x")
        for spine in ax.spines.values():
            spine.set_color("#5b5a31")
        fig.tight_layout()
    return fig, ax


def _palette_for(work: Any, group_col: str, order: list[str]) -> dict[str, str]:
    """Return stable GlyphBench colors keyed by plotted group label."""
    suite_by_group: dict[str, str] = {}
    if "suite" in work.columns:
        for group, suite in (
            work[[group_col, "suite"]].dropna().drop_duplicates(group_col).itertuples(index=False)
        ):
            suite_by_group[str(group)] = str(suite)

    fallback = cycle(_FALLBACK_COLORS)
    palette: dict[str, str] = {}
    for group in order:
        suite = group if group_col == "suite" else suite_by_group.get(str(group))
        if suite is not None and str(suite) in _SUITE_COLORS:
            palette[str(group)] = _SUITE_COLORS[str(suite)]
        else:
            palette[str(group)] = next(fallback)
    return palette
