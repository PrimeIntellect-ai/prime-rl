from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "martheballon/Omni-MATH-2"
DEFAULT_DATASET_REVISION = "1c10fd492252173c73468badf6dc1804225eb5bb"
DEFAULT_OUTPUT_DIR = Path("benchmarks/datasets/omni_math2")
DEFAULT_SAMPLE_SIZES = (500, 600)


def primary_domain(domain: Any) -> str:
    if not domain:
        return "missing"
    if isinstance(domain, str):
        first = domain
    else:
        first = str(domain[0]) if domain else ""
    parts = [part.strip() for part in first.split("->")]
    if parts and parts[0] == "Mathematics":
        parts = parts[1:]
    return parts[0] if parts and parts[0] else "missing"


def difficulty_label(value: Any) -> str:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return f"{float(value):g}"
    return str(value)


def source_bucket(source: Any, top_sources: set[str]) -> str:
    source_text = str(source)
    return source_text if source_text in top_sources else "other"


def sampling_stratum(row: Mapping[str, Any], top_sources: set[str]) -> str:
    return " | ".join(
        (
            primary_domain(row.get("domain")),
            difficulty_label(row.get("difficulty")),
            source_bucket(row.get("source"), top_sources),
        )
    )


def allocate_largest_remainder(
    counts: Mapping[str, int],
    sample_size: int,
    *,
    tie_breakers: Mapping[str, float] | None = None,
) -> dict[str, int]:
    total = sum(counts.values())
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds population size={total}")
    if total == 0:
        if sample_size:
            raise ValueError("cannot sample from an empty population")
        return {key: 0 for key in counts}

    quotas = {key: count * sample_size / total for key, count in counts.items()}
    allocation = {key: min(counts[key], int(math.floor(quotas[key]))) for key in counts}
    remaining = sample_size - sum(allocation.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            quotas[key] - math.floor(quotas[key]),
            counts[key],
            tie_breakers[key] if tie_breakers is not None else 0.0,
            str(key),
        ),
        reverse=True,
    )

    while remaining:
        changed = False
        for key in ranked:
            if allocation[key] >= counts[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            raise RuntimeError("allocation could not satisfy requested sample size")
    return allocation


def stratified_sample_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    stratum_key: Callable[[Mapping[str, Any]], str],
) -> tuple[list[int], dict[str, int], dict[str, float]]:
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_stratum[stratum_key(row)].append(index)

    counts = {key: len(indices) for key, indices in by_stratum.items()}
    total = sum(counts.values())
    quotas = {key: count * sample_size / total for key, count in counts.items()} if total else {}
    rng = random.Random(seed)
    allocation = allocate_largest_remainder(
        counts,
        sample_size,
        tie_breakers={key: rng.random() for key in by_stratum},
    )
    selected: list[int] = []
    for key in sorted(by_stratum):
        take = allocation[key]
        if take == 0:
            continue
        selected.extend(rng.sample(by_stratum[key], take))
    return sorted(selected), allocation, quotas


def stratified_sample_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    stratum_key: Callable[[Mapping[str, Any]], str],
) -> list[int]:
    indices, _, _ = stratified_sample_plan(
        rows,
        sample_size=sample_size,
        seed=seed,
        stratum_key=stratum_key,
    )
    return indices


def parse_sample_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("at least one sample size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sample sizes must be positive")
    return sizes


def _counter_share(counter: Counter[str], key: str, total: int) -> float:
    return counter.get(key, 0) / total if total else 0.0


def marginal_report(
    *,
    population: Sequence[Mapping[str, Any]],
    sample: Sequence[Mapping[str, Any]],
    top_sources: set[str],
) -> dict[str, Any]:
    dimensions: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "primary_domain": lambda row: str(row["primary_domain"]),
        "difficulty": lambda row: difficulty_label(row.get("difficulty")),
        "source_bucket": lambda row: source_bucket(row.get("source"), top_sources),
        "has_proof_tag": lambda row: str("proof" in (row.get("tags") or [])),
        "has_image_tag": lambda row: str("image" in (row.get("tags") or [])),
    }
    result: dict[str, Any] = {}
    for name, key_fn in dimensions.items():
        pop_counts = Counter(key_fn(row) for row in population)
        sample_counts = Counter(key_fn(row) for row in sample)
        keys = sorted(set(pop_counts) | set(sample_counts))
        rows = []
        max_abs_error = 0.0
        l1_error = 0.0
        for key in keys:
            pop_share = _counter_share(pop_counts, key, len(population))
            sample_share = _counter_share(sample_counts, key, len(sample))
            abs_error = abs(sample_share - pop_share)
            max_abs_error = max(max_abs_error, abs_error)
            l1_error += abs_error
            rows.append(
                {
                    "value": key,
                    "population_count": pop_counts.get(key, 0),
                    "sample_count": sample_counts.get(key, 0),
                    "population_share": pop_share,
                    "sample_share": sample_share,
                    "abs_error": abs_error,
                }
            )
        result[name] = {
            "max_abs_share_error": max_abs_error,
            "total_variation_distance": 0.5 * l1_error,
            "rows": rows,
        }
    return result


def enrich_rows(rows: Sequence[Mapping[str, Any]], top_sources: set[str]) -> list[dict[str, Any]]:
    enriched = []
    stratum_counts = Counter(sampling_stratum(row, top_sources) for row in rows)
    for row in rows:
        item = dict(row)
        item["primary_domain"] = primary_domain(row.get("domain"))
        item["source_bucket"] = source_bucket(row.get("source"), top_sources)
        item["difficulty_label"] = difficulty_label(row.get("difficulty"))
        item["sampling_stratum"] = sampling_stratum(row, top_sources)
        item["population_stratum_count"] = stratum_counts[item["sampling_stratum"]]
        enriched.append(item)
    return enriched


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")


def write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Omni-MATH-2 Stratified Samples",
        "",
        f"Dataset: `{manifest['dataset_name']}` split `{manifest['dataset_split']}`.",
        f"Revision: `{manifest['dataset_revision']}`. Fingerprint: `{manifest['dataset_fingerprint']}`.",
        f"Population rows: {manifest['population_size']}. Seed: {manifest['seed']}.",
        "",
        "Sampling strata: `primary_domain × difficulty × source_bucket`, where `source_bucket` keeps the top sources separate and folds the tail into `other`.",
        f"Sample relationship: `{manifest['sample_relationship']}`. Each sample manifest entry includes the realized allocation and fractional quota for every stratum.",
        "",
        "## Samples",
        "",
    ]
    for sample in manifest["samples"]:
        lines.extend(
            [
                f"### n={sample['sample_size']}",
                "",
                f"File: `{sample['path']}`",
                f"Unique strata selected: {sample['selected_strata']} / {manifest['population_strata']}.",
                "",
                "| marginal | max abs share error | total variation |",
                "|---|---:|---:|",
            ]
        )
        for name, stats in sample["marginals"].items():
            lines.append(f"| {name} | {stats['max_abs_share_error']:.4f} | {stats['total_variation_distance']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def build_samples(
    *,
    dataset_name: str,
    dataset_revision: str | None,
    dataset_split: str,
    output_dir: Path,
    sample_sizes: Sequence[int],
    seed: int,
    top_source_count: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    raw_dataset = load_dataset(dataset_name, split=dataset_split, revision=dataset_revision)
    raw_rows = [dict(row) for row in raw_dataset]
    source_counts = Counter(str(row.get("source")) for row in raw_rows)
    top_sources = {source for source, _ in source_counts.most_common(top_source_count)}
    population = enrich_rows(raw_rows, top_sources)
    stratum_counts = Counter(row["sampling_stratum"] for row in population)

    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_split": dataset_split,
        "dataset_fingerprint": getattr(raw_dataset, "_fingerprint", None),
        "population_size": len(population),
        "population_strata": len(stratum_counts),
        "seed": seed,
        "sample_relationship": "separate_stratified_draws_not_nested",
        "top_sources": sorted(top_sources),
        "samples": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_size in sample_sizes:
        indices, allocation, quotas = stratified_sample_plan(
            population,
            sample_size=sample_size,
            seed=seed,
            stratum_key=lambda row: str(row["sampling_stratum"]),
        )
        sample = [dict(population[index]) for index in indices]
        for row in sample:
            row["sample_size"] = sample_size
            row["sample_seed"] = seed
            row["sample_method"] = "largest_remainder(primary_domain,difficulty,source_bucket)"
        path = output_dir / f"omni_math2_stratified_{sample_size}_seed{seed}.jsonl"
        write_jsonl(path, sample)
        marginals = marginal_report(
            population=population,
            sample=sample,
            top_sources=top_sources,
        )
        manifest["samples"].append(
            {
                "sample_size": sample_size,
                "path": str(path),
                "selected_strata": len({row["sampling_stratum"] for row in sample}),
                "allocation": [
                    {
                        "stratum": stratum,
                        "population_count": stratum_counts[stratum],
                        "quota": quotas[stratum],
                        "allocated_count": allocation.get(stratum, 0),
                    }
                    for stratum in sorted(stratum_counts)
                ],
                "marginals": marginals,
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(output_dir / "README.md", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create stratified Omni-MATH-2 baseline task subsets.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", type=parse_sample_sizes, default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-source-count", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_samples(
        dataset_name=args.dataset_name,
        dataset_revision=args.dataset_revision,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
        top_source_count=args.top_source_count,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}, indent=2))
    for sample in manifest["samples"]:
        print(f"wrote n={sample['sample_size']} to {sample['path']}")


if __name__ == "__main__":
    main()
