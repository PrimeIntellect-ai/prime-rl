from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_KS = (1, 3, 5, 8, 16)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float | None:
    """Chen et al. pass@k estimator for n sampled attempts and c successes."""
    if k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _posterior_value(row: dict[str, Any]) -> float | None:
    value = row.get("posterior_correct")
    if not isinstance(value, int | float):
        return None
    if not math.isfinite(float(value)):
        return None
    return min(1.0, max(0.0, float(value)))


def _odds(probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return float("inf")
    return probability / (1.0 - probability)


def _posterior_prior(row: dict[str, Any]) -> float:
    decision = row.get("judge_decision_last")
    if not isinstance(decision, dict):
        return 0.5
    support = decision.get("support")
    if not isinstance(support, dict):
        return 0.5
    policy = support.get("policy")
    if not isinstance(policy, dict):
        return 0.5
    alpha = policy.get("prior_alpha")
    beta = policy.get("prior_beta")
    if not isinstance(alpha, int | float) or not isinstance(beta, int | float):
        return 0.5
    total = float(alpha) + float(beta)
    if total <= 0.0:
        return 0.5
    return min(1.0, max(0.0, float(alpha) / total))


def _answer_class(row: dict[str, Any]) -> str:
    parsed = row.get("parsed_answer")
    if parsed not in (None, ""):
        return " ".join(str(parsed).strip().casefold().split())
    response = row.get("response")
    if response not in (None, ""):
        return "response:" + " ".join(str(response).strip().casefold().split())
    return f"trial:{row.get('trial_index')}"


def _posterior_success_distribution(rows: Sequence[dict[str, Any]], k: int) -> dict[int, float] | None:
    class_counts: Counter[str] = Counter()
    class_bayes_factors: dict[str, list[float]] = defaultdict(list)
    saw_posterior = False
    for row in rows[:k]:
        answer_class = _answer_class(row)
        class_counts[answer_class] += 1
        posterior = _posterior_value(row)
        if posterior is not None:
            saw_posterior = True
            prior = _posterior_prior(row)
            prior_odds = _odds(prior)
            posterior_odds = _odds(posterior)
            if prior_odds == 0.0:
                bayes_factor = 0.0 if posterior_odds == 0.0 else float("inf")
            elif math.isinf(prior_odds):
                bayes_factor = 1.0 if math.isinf(posterior_odds) else 0.0
            else:
                bayes_factor = posterior_odds / prior_odds
            class_bayes_factors[answer_class].append(bayes_factor)
        else:
            class_bayes_factors[answer_class].append(0.0)
    if not saw_posterior:
        return None

    class_weights = {
        answer_class: class_counts[answer_class] * (sum(values) / len(values))
        for answer_class, values in class_bayes_factors.items()
    }
    if any(math.isinf(weight) for weight in class_weights.values()):
        infinite_classes = [answer_class for answer_class, weight in class_weights.items() if math.isinf(weight)]
        class_weights = {answer_class: 0.0 for answer_class in class_weights}
        for answer_class in infinite_classes:
            class_weights[answer_class] = float(class_counts[answer_class])
        none_weight = 0.0
    else:
        # The unseen/none class gets one unit of prior weight per sampled draw.
        # With no evidence, sampled answer classes and "none of them" split mass
        # 50/50 regardless of how many distinct surface forms the model emitted.
        none_weight = float(sum(class_counts.values()))

    normalizer = none_weight + sum(class_weights.values())
    if normalizer <= 0.0:
        return {i: 0.0 for i in range(k + 1)}

    distribution = {i: 0.0 for i in range(k + 1)}
    distribution[0] = none_weight / normalizer
    for answer_class, weight in class_weights.items():
        distribution[class_counts[answer_class]] += weight / normalizer
    return distribution


def _summarize_posterior_distributions(
    distributions: Sequence[dict[int, float]], k: int
) -> dict[str, Any] | None:
    if not distributions:
        return None
    averaged = {
        i: sum(dist.get(i, 0.0) for dist in distributions) / len(distributions)
        for i in range(k + 1)
    }
    successes_mean = sum(i * probability for i, probability in averaged.items())
    return {
        "num_examples": len(distributions),
        "pass_at_k": 1.0 - averaged.get(0, 0.0),
        "successes_at_k_mean": successes_mean,
        "success_rate_at_k_mean": successes_mean / k if k else None,
        "all_pass_at_k": averaged.get(k, 0.0),
        "successes_at_k_distribution": {
            str(i): averaged.get(i, 0.0) for i in range(k + 1)
        },
    }


def summarize_records(records: Sequence[dict[str, Any]], ks: Iterable[int] = DEFAULT_KS) -> dict[str, Any]:
    """Aggregate flat per-rollout records into sampling, pass@k, and tokenomics."""
    ks = tuple(sorted(set(int(k) for k in ks)))
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_example[str(row["example_id"])].append(row)

    for rows in by_example.values():
        rows.sort(key=lambda r: int(r.get("trial_index", 0)))

    total = len(records)
    successes = sum(1 for r in records if r.get("correct") is True)
    num_examples = len(by_example)
    first_successes = sum(1 for rows in by_example.values() if rows and rows[0].get("correct") is True)

    per_k: dict[str, Any] = {}
    for k in ks:
        usable = [rows for rows in by_example.values() if len(rows) >= k]
        prefix_success_counts = [sum(1 for r in rows[:k] if r.get("correct") is True) for rows in usable]
        unbiased_vals = [
            pass_at_k_unbiased(len(rows), sum(1 for r in rows if r.get("correct") is True), k)
            for rows in by_example.values()
        ]
        unbiased_vals = [v for v in unbiased_vals if v is not None]
        histogram = Counter(prefix_success_counts)
        prefix_pass_at_k = (
            sum(1 for count in prefix_success_counts if count > 0) / len(usable) if usable else None
        )
        unbiased_pass_at_k = sum(unbiased_vals) / len(unbiased_vals) if unbiased_vals else None
        posterior_distributions = [
            distribution
            for rows in usable
            if (distribution := _posterior_success_distribution(rows, k)) is not None
        ]
        per_k[str(k)] = {
            "num_examples": len(usable),
            "pass_at_k": unbiased_pass_at_k,
            "pass_at_k_unbiased_estimate": unbiased_pass_at_k,
            "prefix_pass_at_k": prefix_pass_at_k,
            "successes_at_k_mean": (
                sum(prefix_success_counts) / len(prefix_success_counts) if prefix_success_counts else None
            ),
            "success_rate_at_k_mean": (
                sum(prefix_success_counts) / (len(prefix_success_counts) * k) if prefix_success_counts else None
            ),
            "all_pass_at_k": (
                sum(1 for count in prefix_success_counts if count == k) / len(usable) if usable else None
            ),
            "successes_at_k_histogram": {str(i): histogram.get(i, 0) for i in range(k + 1)},
        }
        posterior = _summarize_posterior_distributions(posterior_distributions, k)
        if posterior is not None:
            per_k[str(k)]["posterior"] = posterior

    input_tokens = sum(float(r.get("input_tokens") or 0.0) for r in records)
    output_tokens = sum(float(r.get("output_tokens") or 0.0) for r in records)
    total_ms = [float(r["total_ms"]) for r in records if isinstance(r.get("total_ms"), int | float)]
    generation_ms = [float(r["generation_ms"]) for r in records if isinstance(r.get("generation_ms"), int | float)]

    return {
        "num_examples": num_examples,
        "num_rollouts": total,
        "mean_sample_accuracy": successes / total if total else 0.0,
        "single_shot_accuracy": first_successes / num_examples if num_examples else 0.0,
        "successes": successes,
        "pass": per_k,
        "tokens": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "avg_input_tokens": input_tokens / total if total else 0.0,
            "avg_output_tokens": output_tokens / total if total else 0.0,
        },
        "latency": {
            "avg_total_ms": sum(total_ms) / len(total_ms) if total_ms else None,
            "avg_generation_ms": sum(generation_ms) / len(generation_ms) if generation_ms else None,
        },
        "truncation_rate": (
            sum(1 for r in records if r.get("is_truncated") is True) / total if total else 0.0
        ),
        "error_rate": sum(1 for r in records if r.get("error")) / total if total else 0.0,
    }
