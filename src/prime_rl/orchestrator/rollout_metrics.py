"""Train and eval rollout metrics.

``compute_train_metrics`` and ``compute_eval_metrics`` each build a wandb dict for one slice of a
batch (the agg pool or a single env, one of the all/effective subsets), keyed
``{prefix}/{subset}/<metric>/<stat>``. Both layer on ``_common_metrics`` for the metrics shared
across train and eval, then add their own — train the reward distribution + filter-pipeline rates,
eval the ``avg@<k>`` score + pass@k / pass^k.

No I/O, no pandas — plain Python over the ``vf.Trace`` properties each rollout exposes.
Aggregation is flat over the rollout list (means/min/max/rates) except the solve rates, which
group by ``group_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from prime_rl.orchestrator.eval_utils import compute_pass_metrics

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

Subset = Literal["all", "effective"]

# Distributional metrics shared by train and eval: (name, getter). Each emits mean/max/min.
_DIST_SPECS: list[tuple[str, Callable[["Rollout"], float]]] = [
    ("num_total_tokens", lambda r: r.num_total_tokens),
    ("num_input_tokens", lambda r: r.num_input_tokens),
    ("num_output_tokens", lambda r: r.num_output_tokens),
    ("num_turns", lambda r: r.num_turns),
    ("num_branches", lambda r: r.num_branches),
]


def _dist(prefix: str, values: list[float]) -> dict[str, float]:
    """mean/max/min for a non-empty value list; ``{}`` for an empty one."""
    if not values:
        return {}
    return {
        f"{prefix}/mean": sum(values) / len(values),
        f"{prefix}/max": float(max(values)),
        f"{prefix}/min": float(min(values)),
    }


def _rate(flags: list[bool]) -> float:
    """Fraction True (0.0 for an empty list)."""
    return sum(flags) / len(flags) if flags else 0.0


def _common_metrics(
    rollouts: list["Rollout"], *, prefix: str, subset: Subset, env_group_size: dict[str, int]
) -> dict[str, float]:
    """The metrics common to train and eval over one ``{prefix}/{subset}`` slice. Empty → ``{}``."""
    if not rollouts:
        return {}
    p = f"{prefix}/{subset}"
    out: dict[str, float] = {}

    # Distributional token / turn / branch metrics
    for name, getter in _DIST_SPECS:
        out |= _dist(f"{p}/{name}", [float(getter(r)) for r in rollouts])

    # Timing: per-phase spans + the full end-to-end total (unused phases default to a 0 span)
    setup = [r.timing.setup.duration for r in rollouts]
    generation = [r.timing.generation.duration for r in rollouts]
    finalize = [r.timing.finalize.duration for r in rollouts]
    scoring = [r.timing.scoring.duration for r in rollouts]
    out |= _dist(f"{p}/timing/setup", setup)
    out |= _dist(f"{p}/timing/generation", generation)
    out |= _dist(f"{p}/timing/finalize", finalize)
    out |= _dist(f"{p}/timing/scoring", scoring)
    out |= _dist(f"{p}/timing/total", [sum(spans) for spans in zip(setup, generation, finalize, scoring)])

    # Custom env @metrics and per-reward-component values — union of keys, averaged over the
    # rollouts that report each (the summed reward is added by the train/eval wrappers)
    for name in sorted({name for r in rollouts for name in r.metrics}):
        out |= _dist(f"{p}/metrics/{name}", [r.metrics[name] for r in rollouts if name in r.metrics])
    for name in sorted({name for r in rollouts for name in r.rewards}):
        out |= _dist(f"{p}/rewards/{name}", [r.rewards[name] for r in rollouts if name in r.rewards])

    # Per-rollout boolean rates; error_rate is structurally 0 on `effective`, so log it on `all` only
    out[f"{p}/is_truncated/mean"] = _rate([r.is_truncated for r in rollouts])
    out[f"{p}/is_completed/mean"] = _rate([r.is_completed for r in rollouts])
    if subset == "all":
        out[f"{p}/error_rate"] = _rate([r.has_error for r in rollouts])

    # Stop-condition breakdown: generation_truncated over all rollouts, per-condition rate over the
    # rollouts that recorded a condition
    out[f"{p}/stop_condition/generation_truncated"] = _rate(
        [r.is_truncated and r.stop_condition != "prompt_too_long" for r in rollouts]
    )
    conditions = [r.stop_condition for r in rollouts if r.stop_condition is not None]
    for condition in sorted(set(conditions)):
        out[f"{p}/stop_condition/{condition}"] = conditions.count(condition) / len(conditions)

    # Solve rates (per group): none / all of the group's configured slots earned reward; some is the
    # mixed-reward remainder (the groups that carry GRPO signal)
    groups: dict = {}
    for r in rollouts:
        groups.setdefault(r.group_id, []).append(r)
    solved_none = sum(1 for g in groups.values() if sum(r.reward for r in g) == 0)
    solved_all = sum(
        1 for g in groups.values() if sum(r.reward for r in g) == env_group_size.get(g[0].env_name, len(g))
    )
    n_groups = len(groups)
    out[f"{p}/solved_none"] = solved_none / n_groups
    out[f"{p}/solved_all"] = solved_all / n_groups
    out[f"{p}/solved_some"] = 1 - (solved_none + solved_all) / n_groups
    return out


def compute_train_metrics(
    rollouts: list["Rollout"], *, prefix: str, subset: Subset, env_group_size: dict[str, int]
) -> dict[str, float]:
    """Train metrics: the common set plus the reward distribution and filter-pipeline rates."""
    out = _common_metrics(rollouts, prefix=prefix, subset=subset, env_group_size=env_group_size)
    if not rollouts:
        return out
    p = f"{prefix}/{subset}"
    out |= _dist(f"{p}/reward", [float(r.reward) for r in rollouts])
    out[f"{p}/is_filtered/mean"] = _rate([r.is_filtered for r in rollouts])
    for name in sorted({name for r in rollouts for name in r.filter_results}):
        out[f"{p}/filters/{name}/mean"] = _rate([r.filter_results.get(name, False) for r in rollouts])
    return out


def compute_eval_metrics(
    rollouts: list["Rollout"], *, prefix: str, subset: Subset, group_size: int
) -> dict[str, float]:
    """Eval metrics: the common set plus the ``avg@<group_size>`` score, and — on the effective
    subset, for binary-reward tasks — pass@k / pass^k averaged over examples."""
    env_group_size = {r.env_name: group_size for r in rollouts}
    out = _common_metrics(rollouts, prefix=prefix, subset=subset, env_group_size=env_group_size)
    if not rollouts:
        return out
    p = f"{prefix}/{subset}"
    out[f"{p}/avg@{group_size}"] = sum(float(r.reward) for r in rollouts) / len(rollouts)
    rewards = [r.reward for r in rollouts]
    if subset == "effective" and set(rewards).issubset({0.0, 1.0}):
        by_example: dict = {}
        for r in rollouts:
            by_example.setdefault(r.group_id, []).append(r.reward)
        per_example = [compute_pass_metrics(rs) for rs in by_example.values()]
        for key in sorted({k for d in per_example for k in d}):
            values = [d[key] for d in per_example if key in d]
            out[f"{p}/{key}"] = sum(values) / len(values)
    return out
