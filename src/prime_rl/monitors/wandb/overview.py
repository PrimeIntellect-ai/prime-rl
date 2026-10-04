"""Curated "overview" W&B saved view.

prime-rl logs many metrics; the default workspace auto-generates a panel per key, which buries the
few that matter. These build a named saved view grouping the important metrics into sections, so a
new project gets a usable overview without hand-picking panels. Panels are untitled — each shows
its raw metric name.

The panels live in `prime_rl/monitors/panels.json`, which the dashboard's overview tab reads too.
A panel is `{"metric": key}`, `{"metrics": [keys]}` (one multi-series plot) or `{"regex": pattern}`;
`split` (one card per matched key) only matters to the dashboard.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_gql import gql

from prime_rl.utils.logger import get_logger

OVERVIEW_NAME = "overview"

# Each training mode (rl, sft, eval) declares its own full list of sections. Rollout sections are
# per env. Quality metrics read the effective subset — the all subset includes errored rollouts,
# whose zero values skew the distributions. has_error only exists on all (effective drops errors by
# construction). The count metrics are episode-level exact keys; the trace-level metrics (reward,
# truncation, errors) live under the per-agent subtree, whose names are data-dependent — matched by
# regex, one panel per agent. Train scores with "reward/mean", eval with "avg@k" (its k dynamic, so a
# regex); eval panels are all regexes so one section can also serve any env. Inference panels pair
# the fleet aggregate (mean/sum) with the cross-engine tail (max for pressure metrics, min for health
# metrics): one saturated engine hides inside fleet means.
PANELS = json.loads((Path(__file__).parents[1] / "panels.json").read_text())

# Dense grid: more, smaller panels per row and enough rows that sections don't paginate.
COLUMNS = 4
ROWS = 6


def line_plot(panel: dict) -> wr.LinePlot:
    if "regex" in panel:
        return wr.LinePlot(x="step", metric_regex=panel["regex"])
    # inference/* and the dispatcher gauges are logged against time (step_metric="_timestamp"), plotted
    # on "RelativeTime(Wall)" (== W&B's "_absolute_runtime", seconds since run start) so runs started at
    # different times overlay; everything else on "step" (prime-rl's logged training step, not internal
    # "Step"). x is set per-panel because LinePlot defaults it to "Step", which overrides the workspace x_axis.
    y = panel.get("metrics") or [panel["metric"]]
    time_keyed = any(m.startswith(("inference/", "dispatcher/")) for m in y)
    return wr.LinePlot(x="RelativeTime(Wall)" if time_keyed else "step", y=y)


def section(name: str, panels: Sequence[dict]) -> ws.Section:
    return ws.Section(
        name=name,
        is_open=True,
        panels=[line_plot(p) for p in panels],
        layout_settings=ws.SectionLayoutSettings(columns=COLUMNS, rows=ROWS),
    )


def env_sections(name: str, panels: Sequence[dict], envs: Sequence[str]) -> list[ws.Section]:
    # A per-env section's panels sit under "<name>/<env>/". Env names may carry regex metacharacters
    # (e.g. "+"), so the env is escaped in the regex panels.
    def scoped(title: str, key_prefix: str, regex_prefix: str) -> ws.Section:
        return section(
            title,
            [
                {"regex": f"{regex_prefix}/{p['regex']}"} if "regex" in p else {"metric": f"{key_prefix}/{p['metric']}"}
                for p in panels
            ],
        )

    if not envs:
        # Env names unknown: train shows the cross-env aggregate, eval one section matching any env.
        fallback = f"{name}/agg" if name == "train" else f"{name}/.*"
        return [scoped(name, fallback, fallback)]
    # With one train env the aggregate == that env, so show only its section. With several, put the
    # cross-env aggregate on top.
    sections = [scoped(f"{name}/agg", f"{name}/agg", f"{name}/agg")] if name == "train" and len(envs) > 1 else []
    return sections + [scoped(f"{name}/{env}", f"{name}/{env}", f"{name}/{re.escape(env)}") for env in envs]


def build_sections(
    train_envs: Sequence[str] = (), eval_envs: Sequence[str] = (), flavor: Literal["rl", "sft", "eval"] = "rl"
) -> list[ws.Section]:
    envs = {"train": train_envs, "eval": eval_envs}
    sections = []
    for decl in PANELS[flavor]:
        if decl.get("per_env"):
            sections += env_sections(decl["name"], decl["panels"], envs[decl["name"]])
        else:
            sections.append(section(decl["name"], decl["panels"]))
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


def view_signature(sections: Sequence[ws.Section]) -> tuple:
    train = sorted(s.name[len("train/") :] for s in sections if s.name.startswith("train/") and s.name != "train/agg")
    evals = sorted(s.name[len("eval/") :] for s in sections if s.name.startswith("eval/"))
    panels = {
        (getattr(p.x, "name", p.x), tuple(getattr(m, "name", m) for m in p.y or ()), p.metric_regex)
        for s in sections
        for p in s.panels
        if isinstance(p, wr.LinePlot)
    }
    return (tuple(train), tuple(evals), tuple(sorted(panels, key=str)))


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
    flavor: Literal["rl", "sft", "eval"] = "rl",
) -> str | None:
    """Ensure an overview saved view exists for this run's env set. Reuses an existing overview built
    for the same envs; when the env set is new, creates a fresh versioned view (``overview`` →
    ``overview-v2`` → …). Returns the URL of a newly created view, else None."""
    sections = build_sections(train_envs, eval_envs, flavor)
    target = view_signature(sections)
    overviews = [(dn, iname) for dn, iname in list_views(entity, project) if dn == name or dn.startswith(f"{name}-v")]
    for _, internal_name in overviews:
        slug = internal_name.removeprefix("nw-").removesuffix("-v")
        try:
            existing = ws.Workspace.from_url(f"https://wandb.ai/{entity}/{project}?nw={slug}")
            matches = view_signature(existing.sections) == target
        except Exception as e:
            # A single unreadable view must not abort reuse detection / versioning for the rest.
            get_logger().warning(f"Could not inspect overview view {internal_name} - {e}")
            continue
        if matches:
            return None
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=next_overview_name(name, [dn for dn, _ in overviews]),
        sections=sections,
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace.save()
    return workspace.url
