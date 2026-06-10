from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import verifiers as vf

from prime_rl.utils.monitor.base import Monitor


def _debate_winner(rollout: vf.RolloutOutput) -> tuple[bool, str | None]:
    cats = (rollout.get("mar_score") or {}).get("episode_categorical") or {}
    if "winner" not in cats:
        return False, None
    return True, cats["winner"]


def _truth_member(rollout: vf.RolloutOutput) -> str | None:
    """Which debater holds the truth side of this episode, if resolvable.

    Symmetric packs (both debaters declare answers): the truth side is the
    one whose final answer is correct; both-right/both-wrong is unresolvable.

    Asymmetric packs (e.g. pcd4_final's answerless critic): the single
    answer-declaring member determines the truth side — itself when correct,
    the opposing member when wrong. Asymmetry is distinguished from a
    grader/extraction failure via the rubric's episode keys: the role-aware
    ``any_answer_member_correct`` is emitted only when every answer-declaring
    member resolved, while the legacy ``all_debaters_correct`` alias is
    emitted only when *all* debaters declare answers.
    """
    final_correct = {}
    for member in ("debater_a", "debater_b"):
        key = f"final_correct/{member}"
        if key in rollout:
            final_correct[member] = rollout[key]
    if len(final_correct) == 2:
        a, b = final_correct["debater_a"], final_correct["debater_b"]
        if a == b:
            return None
        return "debater_a" if a > b else "debater_b"
    if len(final_correct) == 1 and "any_answer_member_correct" in rollout and "all_debaters_correct" not in rollout:
        ((member, value),) = final_correct.items()
        opponent = "debater_b" if member == "debater_a" else "debater_a"
        return member if value == 1.0 else opponent
    return None


def _completion_tokens_by_member(rollout: vf.RolloutOutput) -> dict[str, int]:
    out: dict[str, int] = {}
    for step in rollout.get("trajectory") or []:
        member_id = step["extras"]["member_id"]
        tokens = step["tokens"]
        if tokens is None:
            continue
        out[member_id] = out.get(member_id, 0) + len(tokens["completion_ids"])
    return out


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float:
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
            avg = (i + j) / 2 + 1
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
    rows: list[dict[str, Any]] = []
    for rollout in rollouts:
        is_debate, winner = _debate_winner(rollout)
        if not is_debate:
            continue
        truth = _truth_member(rollout)
        rows.append(
            {
                "rollout": rollout,
                "winner": winner,
                "truth": truth,
                "correct": truth is not None and winner == truth,
                "tokens": _completion_tokens_by_member(rollout),
            }
        )
    n = len(rows)
    if n == 0:
        return {}

    resolvable = [row for row in rows if row["truth"] is not None]
    n_resolvable = len(resolvable)
    metrics: dict[str, float] = {
        "n_rollouts": float(n),
        "n_resolvable": float(n_resolvable),
        "resolvable_rate": n_resolvable / n,
    }

    if n_resolvable > 0:
        metrics["twc_3way"] = _safe_mean([float(row["correct"]) for row in resolvable])
        metrics["twc_3way_null"] = 1.0 / 3.0

        non_tie = [row for row in resolvable if row["winner"] != "tie"]
        if non_tie:
            metrics["twc_2way_cond"] = _safe_mean([float(row["correct"]) for row in non_tie])
            metrics["twc_2way_cond_null"] = 0.5
            metrics["n_non_tie"] = float(len(non_tie))
        metrics["tie_rate"] = (n_resolvable - len(non_tie)) / n_resolvable

        by_seat: dict[str, list[float]] = {"a": [], "b": []}
        for row in resolvable:
            seat = row["truth"].split("_")[1]
            by_seat[seat].append(float(row["correct"]))
        if by_seat["a"] and by_seat["b"]:
            twc_a, twc_b = _safe_mean(by_seat["a"]), _safe_mean(by_seat["b"])
            metrics["twc_by_seat_a"] = twc_a
            metrics["twc_by_seat_b"] = twc_b
            metrics["position_bias"] = abs(twc_a - twc_b)

    for member in ("debater_a", "debater_b"):
        good = bad = total = 0
        for row in rows:
            rollout = row["rollout"]
            initial_correct = rollout.get(f"initial_correct/{member}")
            final_correct = rollout.get(f"final_correct/{member}")
            if initial_correct is None or final_correct is None:
                continue
            total += 1
            if initial_correct == 0.0 and final_correct == 1.0:
                good += 1
            elif initial_correct == 1.0 and final_correct == 0.0:
                bad += 1
        if total:
            metrics[f"mind_change_good_rate/{member}"] = good / total
            metrics[f"mind_change_bad_rate/{member}"] = bad / total

    metrics["error_rate"] = sum(1 for row in rows if row["rollout"].get("error") is not None) / n
    metrics["truncation_rate"] = sum(1 for row in rows if row["rollout"].get("is_truncated")) / n

    for member in ("debater_a", "debater_b", "judge"):
        turns = [row["rollout"][f"turns/{member}"] for row in rows if f"turns/{member}" in row["rollout"]]
        if turns:
            metrics[f"avg_turns/{member}"] = _safe_mean(turns)
        tokens = [row["tokens"][member] for row in rows if member in row["tokens"]]
        if tokens:
            metrics[f"completion_tokens_mean/{member}"] = _safe_mean([float(x) for x in tokens])

    for member in ("debater_a", "debater_b"):
        flips = [row["rollout"][f"flipped/{member}"] for row in rows if f"flipped/{member}" in row["rollout"]]
        if flips:
            metrics[f"flip_rate/{member}"] = _safe_mean(flips)

    paired: list[tuple[float, float]] = []
    for row in rows:
        winner = row["winner"]
        if winner == "tie" or winner is None:
            continue
        tokens_a = row["tokens"].get("debater_a")
        tokens_b = row["tokens"].get("debater_b")
        if tokens_a is None or tokens_b is None:
            continue
        paired.append((float(tokens_a - tokens_b), 1.0 if winner == "debater_a" else 0.0))
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
    scalars = compute_step_metrics(rollouts)
    if not scalars:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"step": step, **scalars}, f, sort_keys=True)
    monitor.log({f"{prefix}/{k}": v for k, v in scalars.items()} | {"step": step}, step)
