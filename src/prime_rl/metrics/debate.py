"""Tier-2 debate metric aggregator.

Consumes rollout primitives emitted by verifiers' ``DebateRubric``
(``first_answer/{member}``, ``final_answer/{member}``, ``flipped/{member}``,
per-member rewards, MAR episode_categorical winner) and produces
per-training-step scalars for W&B + a sidecar JSONL.

Scope: what survives without judge logprobs. Calibration (Brier,
Reliability, Resolution) is tracked separately and requires a
logprob-capable judge backend (see joanvelja/verifiers#5).

Design: one pure function ``compute_step_metrics`` for aggregation,
plus ``write_step_metrics`` for the fire-and-forget write path. The
orchestrator calls ``write_step_metrics`` once per ``save_rollouts``
site. Debate-specific: skips rollouts where the env isn't a debate env
(no ``winner`` in ``mar_score.episode_categorical``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import verifiers as vf

from prime_rl.utils.monitor.base import Monitor


def _debate_winner(rollout: vf.RolloutOutput) -> tuple[bool, str | None]:
    """Return (is_debate_rollout, winner). Single lookup for the two
    facts every downstream loop needs — 'winner' in episode_categorical
    is the authoritative marker; absence = single-agent env.

    Assumes ``mar_score`` is already a dict (orchestrator serializes via
    save_rollouts; test fixtures pass dicts directly).
    """
    cats = (rollout.get("mar_score") or {}).get("episode_categorical") or {}
    if "winner" not in cats:
        return False, None
    return True, cats["winner"]


def _truth_member(rollout: vf.RolloutOutput) -> str | None:
    """The member whose committed answer matches ground truth.

    ``info.learner_seat`` alone doesn't identify truth — truth_member is
    assigned at dataset-prep time by gpqa_debate and stored via
    ``truth_member``'s mapping to per-member ``final_correct``. We
    infer: the debater with ``final_correct=1`` is truth-side (when
    exactly one debater is correct — the resolvable case).
    """
    final_correct = {}
    for member in ("debater_a", "debater_b"):
        key = f"final_correct/{member}"
        if key in rollout:
            final_correct[member] = rollout[key]
    if len(final_correct) != 2:
        return None
    a, b = final_correct["debater_a"], final_correct["debater_b"]
    if a == b:
        return None  # both correct or both wrong — un-resolvable
    return "debater_a" if a > b else "debater_b"


def _completion_tokens_by_member(rollout: vf.RolloutOutput) -> dict[str, int]:
    """Sum of completion token lengths per member across the trajectory.

    Returns {} when trajectory is stripped (dump_trajectory=False) or
    when no step has tokens attached (externally-authored turns).
    """
    out: dict[str, int] = {}
    for step in rollout.get("trajectory") or []:
        mid = step["extras"]["member_id"]
        tokens = step["tokens"]
        if tokens is None:
            continue
        out[mid] = out.get(mid, 0) + len(tokens["completion_ids"])
    return out


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0 when input has fewer than 2
    samples or either variable is constant-valued (undefined). Callers
    pass paired lists of the same length. Dependency-free rank with
    average-rank tie handling."""
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx2 = sum((rx[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ry[i] - my) ** 2 for i in range(n))
    if dx2 == 0 or dy2 == 0:
        return 0.0
    return num / math.sqrt(dx2 * dy2)


def compute_step_metrics(rollouts: Iterable[vf.RolloutOutput]) -> dict[str, float]:
    """Aggregate tier-2 debate metrics over a batch of rollouts.

    Inputs: an iterable of RolloutOutputs. Non-debate rollouts (those
    missing ``mar_score.episode_categorical.winner``) are silently
    skipped — this lets mixed-env runs call us uniformly.

    Outputs: flat ``{metric_name: float}``, safe to pass to monitor.log().

    Metric families:
      - twc_3way, twc_2way_cond: 3-way and conditional-on-non-tie TWC
      - tie_rate: Pr[W=tie | C_a != C_b]
      - position_bias: |TWC(truth=a) - TWC(truth=b)|
      - mind_change_{good,bad}_rate: per-debater wrong→right and right→wrong
      - length_bias_corr: Spearman(length_delta, winner-is-seat-a)
      - parse_fail_rate: fraction of rollouts with errored mar_score
      - truncation_rate: fraction with is_truncated=True
      - avg_turns/{member}: mean turn count
      - completion_tokens_mean/{member}: mean model-generated token length
      - n_rollouts, n_resolvable: sample-size diagnostics
    """
    # One pre-pass: filter to debate-shaped rollouts and memoize every
    # derivation downstream loops would otherwise recompute (winner, truth,
    # per-member token counts, and the W==truth correctness indicator).
    rows: list[dict[str, Any]] = []
    for r in rollouts:
        is_debate, winner = _debate_winner(r)
        if not is_debate:
            continue
        truth = _truth_member(r)
        rows.append(
            {
                "r": r,
                "winner": winner,
                "truth": truth,
                "correct": truth is not None and winner == truth,
                "tokens": _completion_tokens_by_member(r),
            }
        )
    n = len(rows)
    if n == 0:
        return {}

    # Resolvable = exactly one debater correct (C_a != C_b). Ties and
    # both-correct / both-wrong go into separate diagnostic counts but
    # out of TWC.
    resolvable = [row for row in rows if row["truth"] is not None]
    n_resolvable = len(resolvable)

    metrics: dict[str, float] = {
        "n_rollouts": float(n),
        "n_resolvable": float(n_resolvable),
        "resolvable_rate": n_resolvable / n,
    }

    # ---- TWC family --------------------------------------------------
    if n_resolvable > 0:
        metrics["twc_3way"] = _safe_mean([float(row["correct"]) for row in resolvable])
        metrics["twc_3way_null"] = 1.0 / 3.0  # reference line for plots

        non_tie = [row for row in resolvable if row["winner"] != "tie"]
        if non_tie:
            metrics["twc_2way_cond"] = _safe_mean([float(row["correct"]) for row in non_tie])
            metrics["twc_2way_cond_null"] = 0.5
            metrics["n_non_tie"] = float(len(non_tie))

        # Partition complement: tie count is the residual after non_tie.
        metrics["tie_rate"] = (n_resolvable - len(non_tie)) / n_resolvable

        by_seat: dict[str, list[float]] = {"a": [], "b": []}
        for row in resolvable:
            # _truth_member returns only debater_{a,b} or None; resolvable
            # filtered None, so the suffix is always 'a' or 'b'.
            seat = row["truth"].split("_")[1]
            by_seat[seat].append(float(row["correct"]))
        if by_seat["a"] and by_seat["b"]:
            twc_a, twc_b = _safe_mean(by_seat["a"]), _safe_mean(by_seat["b"])
            metrics["twc_by_seat_a"] = twc_a
            metrics["twc_by_seat_b"] = twc_b
            metrics["position_bias"] = abs(twc_a - twc_b)

    # ---- Mind-change: wrong→right (good) vs right→wrong (bad) --------
    for member in ("debater_a", "debater_b"):
        good = bad = total = 0
        for row in rows:
            r = row["r"]
            ic = r.get(f"initial_correct/{member}")
            fc = r.get(f"final_correct/{member}")
            if ic is None or fc is None:
                continue
            total += 1
            if ic == 0.0 and fc == 1.0:
                good += 1
            elif ic == 1.0 and fc == 0.0:
                bad += 1
        if total:
            metrics[f"mind_change_good_rate/{member}"] = good / total
            metrics[f"mind_change_bad_rate/{member}"] = bad / total

    # ---- Parse-fail + truncation + per-member turns/tokens/flips -----
    metrics["error_rate"] = sum(1 for row in rows if row["r"]["error"] is not None) / n
    metrics["truncation_rate"] = sum(1 for row in rows if row["r"]["is_truncated"]) / n

    for member in ("debater_a", "debater_b", "judge"):
        turns = [row["r"][f"turns/{member}"] for row in rows if f"turns/{member}" in row["r"]]
        if turns:
            metrics[f"avg_turns/{member}"] = _safe_mean(turns)
        tokens = [row["tokens"][member] for row in rows if member in row["tokens"]]
        if tokens:
            metrics[f"completion_tokens_mean/{member}"] = _safe_mean([float(x) for x in tokens])

    for member in ("debater_a", "debater_b"):
        flips = [row["r"][f"flipped/{member}"] for row in rows if f"flipped/{member}" in row["r"]]
        if flips:
            metrics[f"flip_rate/{member}"] = _safe_mean(flips)

    # ---- Length bias: Spearman(length_delta, winner-is-a). Requires
    # per-member token counts (i.e. dump_trajectory=true) and at least one
    # non-tie decision to regress against.
    paired: list[tuple[float, float]] = []
    for row in rows:
        w = row["winner"]
        if w == "tie" or w is None:
            continue
        la = row["tokens"].get("debater_a")
        lb = row["tokens"].get("debater_b")
        if la is None or lb is None:
            continue
        paired.append((float(la - lb), 1.0 if w == "debater_a" else 0.0))
    if len(paired) >= 2:
        metrics["length_bias_corr"] = _spearman([p[0] for p in paired], [p[1] for p in paired])

    return metrics


def write_step_metrics(
    rollouts: list[vf.RolloutOutput],
    *,
    path: Path,
    step: int,
    monitor: Monitor,
    prefix: str,
) -> None:
    """Compute + persist tier-2 step metrics.

    Writes one JSON object to ``path`` (sidecar to ``*_rollouts.jsonl``)
    and logs the same scalars to the monitor under ``{prefix}/{key}``.
    No-op when no debate-shaped rollouts are present (mixed-env safety).

    The prefix namespaces W&B scalars so they don't collide with the
    per-member MA rubric fanout (which writes reward/debater_a etc. at
    the top level).
    """
    scalars = compute_step_metrics(rollouts)
    if not scalars:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"step": step, **scalars}, f)
    # Inject "step" into the payload — PrimeMonitor.log ignores its step
    # kwarg (forwards only metrics to the API), so backends that consume
    # the payload directly would otherwise have no step axis to index on.
    payload = {f"{prefix}/{k}": v for k, v in scalars.items()} | {"step": step}
    monitor.log(payload, step)
