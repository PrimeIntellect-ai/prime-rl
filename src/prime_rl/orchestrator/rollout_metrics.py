"""Unified rollout-metric aggregation.

One pure function computes the rollout metric set over an arbitrary list of rollouts and
returns a wandb dict already keyed ``{prefix}/{subset}/<metric>/<stat>``. It is called once
per cell of the ``{train,eval} / {agg,<env>} / {all,effective}`` matrix at the batch boundary
(see ``orchestrator.finalize_train_batch`` / ``finalize_eval_batch``).

No I/O, no pandas — plain Python aggregation over the ``vf.Trace`` properties each rollout
already exposes. Aggregation is flat over the rollout list (means/min/max/rates), except the
solve rates, which group by ``group_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from prime_rl.orchestrator.eval_utils import compute_pass_metrics

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

Subset = Literal["all", "effective"]

# Distributional metrics: (name, getter). Each emits mean/max/min over the rollout list.
_DIST_SPECS: list[tuple[str, Callable[["Rollout"], float]]] = [
    ("num_total_tokens", lambda r: r.num_total_tokens),
    ("num_input_tokens", lambda r: r.num_input_tokens),
    ("num_output_tokens", lambda r: r.num_output_tokens),
    ("num_turns", lambda r: r.num_turns),
    ("num_branches", lambda r: r.num_branches),
    ("reward", lambda r: r.reward),
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


def compute_rollout_metrics(
    rollouts: list["Rollout"],
    *,
    prefix: str,
    subset: Subset,
    env_group_size: dict[str, int],
    include_filters: bool = False,
    include_pass_at_k: bool = False,
) -> dict[str, float]:
    """Aggregate one slice of a batch into a wandb dict.

    ``prefix`` is the ``{train,eval}/{agg,<env>}`` head; ``subset`` is ``all`` or
    ``effective`` and completes the key path. ``env_group_size`` maps env name → configured
    group size (the full-solve threshold; the slice can pool multiple envs, so it is looked up
    per group's env). ``include_filters`` (train) adds the filter-pipeline rates;
    ``include_pass_at_k`` (eval) adds pass@k / pass^k. Empty input → ``{}``.
    """
    if not rollouts:
        return {}
    p = f"{prefix}/{subset}"
    out: dict[str, float] = {}

    # Distributional metrics
    for name, getter in _DIST_SPECS:
        out |= _dist(f"{p}/{name}", [float(getter(r)) for r in rollouts])

    # Timing (per-rollout span durations from the v1 Trace; total is the full end-to-end across
    # all four phases — unused phases default to a 0-duration span)
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
    # rollouts that report each (the summed reward is the `reward` metric above)
    for name in sorted({name for r in rollouts for name in r.metrics}):
        out |= _dist(f"{p}/metrics/{name}", [r.metrics[name] for r in rollouts if name in r.metrics])
    for name in sorted({name for r in rollouts for name in r.rewards}):
        out |= _dist(f"{p}/rewards/{name}", [r.rewards[name] for r in rollouts if name in r.rewards])

    # Per-rollout boolean rates
    out[f"{p}/is_truncated/mean"] = _rate([r.is_truncated for r in rollouts])
    out[f"{p}/is_completed/mean"] = _rate([r.is_completed for r in rollouts])
    # error_rate is structurally 0 on `effective` (it excludes errored) — surface it on `all` only
    if subset == "all":
        out[f"{p}/error_rate"] = _rate([r.has_error for r in rollouts])

    # Stop-condition breakdown: generation_truncated over all rollouts, then per-condition rate
    # over the rollouts that recorded a condition (matches the previous value_counts(normalize)).
    out[f"{p}/stop_condition/generation_truncated"] = _rate(
        [r.is_truncated and r.stop_condition != "prompt_too_long" for r in rollouts]
    )
    conditions = [r.stop_condition for r in rollouts if r.stop_condition is not None]
    for condition in sorted(set(conditions)):
        out[f"{p}/stop_condition/{condition}"] = conditions.count(condition) / len(conditions)

    # Solve rates (per group): solved_none if no rollout earned reward, solved_all if every
    # configured slot did (sum == env group size), solved_some the remainder — the mixed-reward
    # groups (the ones that carry GRPO signal).
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

    # Train-only: the filtered-out rate (a top-level rollout metric) plus per-filter detection
    # rates under filters/ (eval has no filter pipeline, so these are gated off there).
    if include_filters:
        out[f"{p}/is_filtered/mean"] = _rate([r.is_filtered for r in rollouts])
        for name in sorted({name for r in rollouts for name in r.filter_results}):
            out[f"{p}/filters/{name}/mean"] = _rate([r.filter_results.get(name, False) for r in rollouts])

    # Eval-only: pass@k / pass^k, averaged over examples (only for binary-reward tasks)
    if include_pass_at_k:
        rewards = [r.reward for r in rollouts]
        if set(rewards).issubset({0.0, 1.0}):
            by_example: dict = {}
            for r in rollouts:
                by_example.setdefault(r.group_id, []).append(r.reward)
            per_example = [compute_pass_metrics(rs) for rs in by_example.values()]
            for key in sorted({k for d in per_example for k in d}):
                values = [d[key] for d in per_example if key in d]
                out[f"{p}/{key}"] = sum(values) / len(values)

    return out
