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

# Per-rollout metrics (under "<scope>/all/") shown for the train aggregate and per env.
TRAIN_METRICS = [
    "reward/mean",
    "has_error/mean",
    "is_truncated/mean",
    "num_total_tokens/mean",
    "num_turns/mean",
    "num_branches/mean",
]

STABILITY_METRICS = ["optim/grad_norm", "entropy/all/mean", "mismatch_kl/all/mean", "kl_ent_ratio/mean"]

PERFORMANCE_METRICS = [
    "perf/throughput",
    "perf/throughput_per_gpu",
    "perf/mfu",
    "perf/peak_memory",
    "time/step",
    "time/wait_for_batch",
    "inference/agg/throughput",
    "inference/agg/running_requests",
    "inference/agg/waiting_requests",
    "inference/agg/kv_cache_usage_mean",
    "inference/agg/prefix_cache_hit_rate",
]

# Dense grid: more, smaller panels per row and enough rows that sections don't paginate.
COLUMNS = 4
ROWS = 6


def x_for(metric: str) -> str:
    # inference/* is logged on a wall-clock timer (step_metric="_timestamp"), so plot it against wall
    # time ("WallTime" == W&B's "_timestamp"). Everything else uses "step" (prime-rl's logged training
    # step) rather than W&B's internal "Step"/_step. This must be per-panel — LinePlot defaults x to
    # "Step", which overrides the workspace-level x_axis setting.
    return "WallTime" if metric.startswith("inference/") else "step"


def line_panels(metrics: Sequence[str], regexes: Sequence[str]) -> list[wr.LinePlot]:
    return [wr.LinePlot(x=x_for(m), y=[m]) for m in metrics] + [wr.LinePlot(x="step", metric_regex=r) for r in regexes]


def section(name: str, metrics: Sequence[str] = (), regexes: Sequence[str] = ()) -> ws.Section:
    return ws.Section(
        name=name,
        is_open=True,
        panels=line_panels(metrics, regexes),
        layout_settings=ws.SectionLayoutSettings(columns=COLUMNS, rows=ROWS),
    )


def train_section(name: str, scope: str) -> ws.Section:
    return section(name, metrics=[f"{scope}/all/{m}" for m in TRAIN_METRICS])


def eval_section(name: str, env_pattern: str) -> ws.Section:
    # Eval has no aggregate, so it's always per-env. avg@k has a dynamic k, so match it by regex; the
    # regex form also lets one section serve any env (env_pattern=".*").
    return section(
        name,
        regexes=[
            f"eval/{env_pattern}/all/avg@.*",
            f"eval/{env_pattern}/effective/avg@.*",
            f"eval/{env_pattern}/all/has_error/mean",
            f"eval/{env_pattern}/all/is_truncated/mean",
        ],
    )


def build_sections(train_envs: Sequence[str] = (), eval_envs: Sequence[str] = ()) -> list[ws.Section]:
    # With one env the aggregate == that env, so show only its section. With several, put the
    # cross-env aggregate on top followed by a section per env.
    if len(train_envs) == 1:
        sections = [train_section(f"train/{train_envs[0]}", f"train/{train_envs[0]}")]
    elif len(train_envs) > 1:
        sections = [train_section("train/agg", "train/agg")]
        sections += [train_section(f"train/{env}", f"train/{env}") for env in train_envs]
    else:
        # Env names unknown (e.g. SFT): fall back to the aggregate.
        sections = [train_section("train", "train/agg")]
    if eval_envs:
        sections += [eval_section(f"eval/{env}", re.escape(env)) for env in eval_envs]
    else:
        # Env names unknown (e.g. SFT): one regex section matching any eval env.
        sections.append(eval_section("eval", ".*"))
    sections.append(section("stability", metrics=STABILITY_METRICS))
    sections.append(section("performance", metrics=PERFORMANCE_METRICS))
    return sections


def list_views(entity: str, project: str) -> list[tuple[str, str]]:
    """``(display_name, internal_name)`` for every saved view in the project."""
    query = gql(
        """
        query Views($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            allViews(viewType: "project-view") { edges { node { name displayName } } }
          }
        }
        """
    )
    res = wandb.Api().client.execute(query, variable_values={"entity": entity, "project": project})
    edges = ((res.get("project") or {}).get("allViews") or {}).get("edges") or []
    return [(e["node"]["displayName"], e["node"]["name"]) for e in edges if e.get("node")]


def env_signature(train_envs: Sequence[str], eval_envs: Sequence[str]) -> tuple:
    return (tuple(sorted(train_envs)), tuple(sorted(eval_envs)))


def view_env_signature(sections: Sequence[ws.Section]) -> tuple:
    """Reconstruct the ``(train, eval)`` env set a view was built for from its section names."""
    train = sorted(s.name[len("train/") :] for s in sections if s.name.startswith("train/") and s.name != "train/agg")
    evals = sorted(s.name[len("eval/") :] for s in sections if s.name.startswith("eval/"))
    return (tuple(train), tuple(evals))


def next_overview_name(base: str, existing: Sequence[str]) -> str:
    if base not in existing:
        return base
    prefix = f"{base}-v"
    versions = [1] + [int(n[len(prefix) :]) for n in existing if n.startswith(prefix) and n[len(prefix) :].isdigit()]
    return f"{base}-v{max(versions) + 1}"


def ensure_overview_view(
    entity: str,
    project: str,
    name: str = OVERVIEW_NAME,
    train_envs: Sequence[str] = (),
    eval_envs: Sequence[str] = (),
) -> str | None:
    """Ensure an overview saved view exists for this run's env set. Reuses an existing overview built
    for the same envs; when the env set is new, creates a fresh versioned view (``overview`` →
    ``overview-v2`` → …). Returns the URL of a newly created view, else None."""
    target = env_signature(train_envs, eval_envs)
    overviews = [(dn, iname) for dn, iname in list_views(entity, project) if dn == name or dn.startswith(f"{name}-v")]
    for _, internal_name in overviews:
        slug = internal_name.removeprefix("nw-").removesuffix("-v")
        existing = ws.Workspace.from_url(f"https://wandb.ai/{entity}/{project}?nw={slug}")
        if view_env_signature(existing.sections) == target:
            return None
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=next_overview_name(name, [dn for dn, _ in overviews]),
        sections=build_sections(train_envs, eval_envs),
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace.save()
    return workspace.url
