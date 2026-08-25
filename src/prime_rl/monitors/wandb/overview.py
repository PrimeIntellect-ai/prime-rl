"""Curated "overview" W&B saved view.

The dashboard mirrors these sections in `prime_rl/dashboard/static/app.js`
(buildSections + the *_METRICS/INFERENCE_PANELS constants) — keep the two in
sync when editing panels here.

prime-rl logs many metrics; the default workspace auto-generates a panel per key, which buries the
few that matter. These build a named saved view grouping the important metrics into sections, so a
new project gets a usable overview without hand-picking panels. Panels are untitled — each shows
its raw metric name.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_gql import gql

from prime_rl.utils.logger import get_logger

OVERVIEW_NAME = "overview"

# Rollout metrics (under "<scope>/") shown for BOTH train and eval. Quality metrics read the
# effective subset — the all subset includes errored rollouts, whose zero values skew the
# distributions. has_error only exists on all (effective drops errors by construction). The count
# metrics are episode-level exact keys; the trace-level metrics (reward, truncation, errors) live
# under the per-agent subtree, whose names are data-dependent — matched by regex, one panel per
# agent. Only the score metric differs — train scores with "reward/mean", eval with "avg@k" (its k
# dynamic, so also a regex) — and each section builder prepends its own.
COMMON_METRICS = [
    "all/dispatch_failure/mean",
    "effective/num_total_tokens/mean",
    "effective/num_turns/mean",
    "effective/num_branches/mean",
]
COMMON_REGEXES = [
    "all/[^/]+/has_error/mean",
    "effective/[^/]+/is_truncated/mean",
]

STABILITY_METRICS = ["optim/grad_norm", "entropy/all/mean", "mismatch_kl/all/mean", "kl_ent_ratio/mean"]

PERFORMANCE_METRICS = [
    "perf/mfu",
    "time/step",
    "time/wait_for_batch",
    "time/wait_for_policy",
]

# Inference health panels: each pairs the fleet aggregate (mean/sum) with the cross-engine
# tail that flags a single sick engine - max for pressure metrics, min for health metrics.
# One saturated engine thrashing its KV cache (preempt -> re-prefill -> cache eviction) hides
# inside fleet means; the max/min series is what surfaces it.
INFERENCE_PANELS = [
    [
        "inference/agg/kv_cache_usage_perc/mean",
        "inference/agg/kv_cache_usage_perc/min",
        "inference/agg/kv_cache_usage_perc/max",
    ],
    ["inference/agg/num_preemptions_total:rate/sum", "inference/agg/num_preemptions_total:rate/max"],
    [
        "inference/agg/num_requests_running/mean",
        "inference/agg/num_requests_running/min",
        "inference/agg/num_requests_running/max",
    ],
    [
        "inference/agg/num_requests_waiting/mean",
        "inference/agg/num_requests_waiting/min",
        "inference/agg/num_requests_waiting/max",
    ],
    ["inference/agg/prefix_cache_hit_rate/pooled", "inference/agg/prefix_cache_hit_rate/min"],
    ["inference/agg/generation_tokens_total:rate/sum", "inference/agg/generation_tokens_total:rate/min"],
    ["inference/agg/prompt_tokens_total:rate/sum", "inference/agg/prompt_tokens_total:rate/max"],
    ["dispatcher/inflight/train", "dispatcher/inflight/eval"],
]

# SFT flavor: no rollout-based train sections — the training signal is the loss curve.
SFT_TRAIN_METRICS = ["loss/mean", "loss/perplexity", "val/loss", "val/perplexity", "progress/epoch"]
SFT_STABILITY_METRICS = ["optim/grad_norm", "optim/lr", "loss/nan_count"]
SFT_PERFORMANCE_METRICS = [
    "perf/mfu",
    "perf/throughput",
    "perf/peak_memory",
    "time/step",
    "time/forward_backward",
    "time/save_ckpt",
]

# Dense grid: more, smaller panels per row and enough rows that sections don't paginate.
COLUMNS = 4
ROWS = 6


def line_panels(metrics: Sequence[str], regexes: Sequence[str]) -> list[wr.LinePlot]:
    # inference/* is logged against time (step_metric="_timestamp"), plotted on "RelativeTime(Wall)"
    # (== W&B's "_absolute_runtime", seconds since run start) so runs started at different times
    # overlay; everything else on "step" (prime-rl's logged training step, not internal "Step").
    # x is set per-panel because LinePlot defaults it to "Step", which overrides the workspace x_axis.
    return [wr.LinePlot(x="RelativeTime(Wall)" if m.startswith("inference/") else "step", y=[m]) for m in metrics] + [
        wr.LinePlot(x="step", metric_regex=r) for r in regexes
    ]


def inference_section() -> ws.Section:
    # Multi-series panels (aggregate + tail), on wall time like all inference/* metrics.
    return ws.Section(
        name="inference",
        is_open=True,
        panels=[wr.LinePlot(x="RelativeTime(Wall)", y=list(series)) for series in INFERENCE_PANELS],
        layout_settings=ws.SectionLayoutSettings(columns=COLUMNS, rows=ROWS),
    )


def section(name: str, metrics: Sequence[str] = (), regexes: Sequence[str] = ()) -> ws.Section:
    return ws.Section(
        name=name,
        is_open=True,
        panels=line_panels(metrics, regexes),
        layout_settings=ws.SectionLayoutSettings(columns=COLUMNS, rows=ROWS),
    )


def train_section(name: str, scope: str) -> ws.Section:
    # Env names may carry regex metacharacters (e.g. "+"), so the scope is escaped in the
    # regex-matched per-agent panels.
    pattern = re.escape(scope)
    return section(
        name,
        metrics=[f"{scope}/{m}" for m in COMMON_METRICS],
        regexes=[f"{pattern}/all/[^/]+/reward/mean", f"{pattern}/effective/[^/]+/reward/mean"]
        + [f"{pattern}/{r}" for r in COMMON_REGEXES],
    )


def eval_section(name: str, env_pattern: str) -> ws.Section:
    # Same metrics as train, but eval's reward is the per-agent "avg@k" (dynamic k → regex).
    # Everything is a regex so one section can also serve any env (env_pattern=".*").
    return section(
        name,
        regexes=[f"eval/{env_pattern}/all/[^/]+/avg@.*", f"eval/{env_pattern}/effective/[^/]+/avg@.*"]
        + [f"eval/{env_pattern}/{m}" for m in COMMON_METRICS]
        + [f"eval/{env_pattern}/{r}" for r in COMMON_REGEXES],
    )


def build_sections(
    train_envs: Sequence[str] = (), eval_envs: Sequence[str] = (), flavor: Literal["rl", "sft"] = "rl"
) -> list[ws.Section]:
    # SFT trains on a dataset, not rollouts: the train section is the loss/perplexity
    # curves, eval sections are the same rollout-based ones as RL.
    if flavor == "sft":
        sections = [section("train", metrics=SFT_TRAIN_METRICS)]
        if eval_envs:
            sections += [eval_section(f"eval/{env}", re.escape(env)) for env in eval_envs]
        else:
            sections.append(eval_section("eval", ".*"))
        sections.append(section("stability", metrics=SFT_STABILITY_METRICS))
        sections.append(section("performance", metrics=SFT_PERFORMANCE_METRICS))
        return sections
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
    sections.append(inference_section())
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
    flavor: Literal["rl", "sft"] = "rl",
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
