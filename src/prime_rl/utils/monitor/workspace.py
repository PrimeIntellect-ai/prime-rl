"""Programmatic creation of a curated "overview" W&B saved view.

prime-rl logs many metrics; the default workspace auto-generates a panel per key, which buries the
few that matter. This builds a named saved view grouping the important metrics into sections, so a
new project gets a usable overview without hand-picking panels. Panels are left untitled, so each
shows its raw metric name.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_gql import gql

OVERVIEW_NAME = "overview"

# prime-rl logs the training step as "step" (see WandbMonitor); use it as the x-axis instead of W&B's
# internal "Step" (_step). This must be set per-panel — LinePlot defaults x to "Step", which overrides
# the workspace-level x_axis setting.
X_AXIS = "step"

# The orchestrator's periodic logger emits the inference/* metrics on a wall-clock timer
# (step_metric="_timestamp"), not per training step, so plot them against wall time. "WallTime" maps
# to W&B's internal "_timestamp" axis.
X_AXIS_WALL = "WallTime"


def _x_for(metric: str) -> str:
    return X_AXIS_WALL if metric.startswith("inference/") else X_AXIS


# Per-rollout metrics (under "<scope>/all/") shown for the train aggregate and per env.
_TRAIN_METRICS = [
    "reward/mean",
    "has_error/mean",
    "is_truncated/mean",
    "num_total_tokens/mean",
    "num_turns/mean",
    "num_branches/mean",
]

_STABILITY = ["optim/grad_norm", "entropy/all/mean", "mismatch_kl/all/mean", "kl_ent_ratio/mean"]

_PERFORMANCE = [
    "perf/throughput",
    "perf/throughput_per_gpu",
    "perf/mfu",
    "perf/peak_memory",
    "time/step",
    "inference/agg/throughput",
    "inference/agg/running_requests",
    "inference/agg/waiting_requests",
    "inference/agg/kv_cache_usage_mean",
    "inference/agg/prefix_cache_hit_rate",
]

# Dense grid: more, smaller panels per row and enough rows that sections don't paginate.
_COLUMNS = 4
_ROWS = 6


def _line_panels(metrics: Sequence[str], regexes: Sequence[str]) -> list[wr.LinePlot]:
    return [wr.LinePlot(x=_x_for(m), y=[m]) for m in metrics] + [wr.LinePlot(x=X_AXIS, metric_regex=r) for r in regexes]


def _section(name: str, metrics: Sequence[str] = (), regexes: Sequence[str] = ()) -> ws.Section:
    return ws.Section(
        name=name,
        is_open=True,
        panels=_line_panels(metrics, regexes),
        layout_settings=ws.SectionLayoutSettings(columns=_COLUMNS, rows=_ROWS),
    )


def _train_section(name: str, scope: str) -> ws.Section:
    return _section(name, metrics=[f"{scope}/all/{m}" for m in _TRAIN_METRICS])


def _eval_section(name: str, env_pattern: str) -> ws.Section:
    # Eval has no aggregate, so it's always per-env. avg@k has a dynamic k, so match it by regex; the
    # regex form also lets one section serve any env (env_pattern=".*").
    return _section(
        name,
        regexes=[
            f"eval/{env_pattern}/all/avg@.*",
            f"eval/{env_pattern}/effective/avg@.*",
            f"eval/{env_pattern}/all/has_error/mean",
            f"eval/{env_pattern}/all/is_truncated/mean",
        ],
    )


def _build_sections(train_envs: Sequence[str] = (), eval_envs: Sequence[str] = ()) -> list[ws.Section]:
    # With one env the aggregate == that env, so show only its section. With several, put the
    # cross-env aggregate on top followed by a section per env.
    if len(train_envs) == 1:
        sections = [_train_section(f"train/{train_envs[0]}", f"train/{train_envs[0]}")]
    elif len(train_envs) > 1:
        sections = [_train_section("train/agg", "train/agg")]
        sections += [_train_section(f"train/{env}", f"train/{env}") for env in train_envs]
    else:
        # Env names unknown (e.g. SFT): fall back to the aggregate.
        sections = [_train_section("train", "train/agg")]
    if eval_envs:
        sections += [_eval_section(f"eval/{env}", re.escape(env)) for env in eval_envs]
    else:
        # Env names unknown (e.g. SFT): one regex section matching any eval env.
        sections.append(_eval_section("eval", ".*"))
    sections.append(_section("stability", metrics=_STABILITY))
    sections.append(_section("performance", metrics=_PERFORMANCE))
    return sections


def _existing_view_names(entity: str, project: str) -> set[str]:
    query = gql(
        """
        query Views($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            allViews(viewType: "project-view") { edges { node { displayName } } }
          }
        }
        """
    )
    res = wandb.Api().client.execute(query, variable_values={"entity": entity, "project": project})
    edges = ((res.get("project") or {}).get("allViews") or {}).get("edges") or []
    return {e["node"]["displayName"] for e in edges if e.get("node")}


def ensure_overview_view(
    entity: str,
    project: str,
    name: str = OVERVIEW_NAME,
    train_envs: Sequence[str] = (),
    eval_envs: Sequence[str] = (),
) -> str | None:
    """Create the overview saved view unless one already exists. Returns its URL if created, else None."""
    if name in _existing_view_names(entity, project):
        return None
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=_build_sections(train_envs, eval_envs),
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis=X_AXIS),
    )
    workspace.save()
    return workspace.url
