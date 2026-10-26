"""Compare per-parameter grad norms of export_driver.py grads records, main vs branch, against main's repeat floor."""

import argparse
import itertools
import json
from pathlib import Path

from compare_export import map_prime_name

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ZERO_FLOOR_THRESHOLD = 1e-6
POWER_OF_TWO_FACTORS = [2.0**exponent for exponent in (-3, -2, -1, 1, 2, 3)]
NAME_PREFIX = "model.language_model."
NAME_WIDTH = 50
TOP_ROWS = 10


def relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


def max_pairwise(values_a: list[float], values_b: list[float] | None = None) -> float:
    pairs = itertools.combinations(values_a, 2) if values_b is None else itertools.product(values_a, values_b)
    return max((relative_difference(a, b) for a, b in pairs), default=0.0)


def power_of_two_factor(ratio: float) -> str:
    for factor in POWER_OF_TWO_FACTORS:
        if abs(ratio / factor - 1) < 0.01:
            return f"x{factor:g}"
    return ""


def short_name(name: str) -> str:
    name = name.removeprefix(NAME_PREFIX)
    return name if len(name) <= NAME_WIDTH else "..." + name[-(NAME_WIDTH - 3) :]


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        record = json.loads(path.read_text())
        assert record["mode"] == "grads", f"{path} is not a grads record"
        record["path"] = str(path)
        record["grad_norms"] = {map_prime_name(name): norm for name, norm in record["grad_norms"].items()}
        runs.append(record)
    return runs


def compare(main_runs: list[dict], branch_runs: list[dict]) -> list[dict]:
    rows = []
    names = sorted(set().union(*(run["grad_norms"].keys() for run in main_runs + branch_runs)))
    scalars = [("(loss)", "loss"), ("(total grad norm, pre-clip)", "total_grad_norm")]
    entries = [(label, [run[key] for run in main_runs], [run[key] for run in branch_runs]) for label, key in scalars]
    for name in names:
        main_values = [run["grad_norms"].get(name, "missing") for run in main_runs]
        branch_values = [run["grad_norms"].get(name, "missing") for run in branch_runs]
        entries.append((name, main_values, branch_values))
    for name, main_values, branch_values in entries:
        if any(not isinstance(value, float) for value in main_values + branch_values):
            presence_matches = len({repr(value) for value in main_values + branch_values}) == 1
            rows.append({"name": name, "flag": not presence_matches, "note": f"{main_values} vs {branch_values}"})
            continue
        floor = max_pairwise(main_values)
        arm = max_pairwise(main_values, branch_values)
        threshold = floor if floor > 0 else ZERO_FLOOR_THRESHOLD
        main_mean = sum(main_values) / len(main_values)
        ratio = (sum(branch_values) / len(branch_values)) / main_mean if main_mean else float("nan")
        rows.append(
            {
                "name": name,
                "main": main_mean,
                "ratio": ratio,
                "floor": floor,
                "arm": arm,
                "flag": arm > threshold,
                "note": power_of_two_factor(ratio),
                "is_ngram_table": ".ngram.tables." in name and name.endswith(".weight"),
            }
        )
    return rows


def table(rows: list[dict]) -> list[str]:
    lines = [
        f"| {'Parameter':<{NAME_WIDTH}} | main norm  | br / main | floor    | arm diff | Flag      |",
        f"| {'-' * NAME_WIDTH} | ---------- | --------- | -------- | -------- | --------- |",
    ]
    for row in rows:
        flag = ("FLAG " if row["flag"] else "") + row.get("note", "")
        if "main" not in row:
            lines.append(
                f"| {short_name(row['name']):<{NAME_WIDTH}} | {'':<10} | {'':<9} | {'':<8} | {'':<8} | {flag[:9]:<9} |"
            )
            continue
        lines.append(
            f"| {short_name(row['name']):<{NAME_WIDTH}} | {row['main']:>10.4e} | {row['ratio']:>9.6f} "
            f"| {row['floor']:>8.1e} | {row['arm']:>8.1e} | {flag[:9]:<9} |"
        )
    return lines


def run_lines(label: str, runs: list[dict]) -> list[str]:
    lines = []
    for run in runs:
        checks = run["ngram_replica_check"]
        replica = "not run" if checks is None else ("equal" if all(check["equal"] for check in checks) else "DIFFER")
        lines.append(
            f"- {label} r{run['repeat']}: `{run['path']}` ({run['arm']} at `{run['source']['commit'][:9]}`, "
            f"dirty={run['source']['dirty']}, optimizer step={run['optimizer_step']}, replica shards {replica})"
        )
    return lines


def format_report(main_runs: list[dict], branch_runs: list[dict], rows: list[dict]) -> str:
    layout = main_runs[0]["layout"]
    flagged = [row for row in rows if row["flag"]]
    numeric = [row for row in rows if "arm" in row]
    ngram_rows = [row for row in numeric if row["is_ngram_table"]]
    scalar_rows = [row for row in rows if row["name"].startswith("(")]
    top_rows = sorted(numeric, key=lambda row: row["arm"], reverse=True)[:TOP_ROWS]
    lines = [f"# Grad norms: {layout}", ""]
    lines += run_lines("main", main_runs) + run_lines("branch", branch_runs)
    lines += [
        "",
        f"- mesh {main_runs[0]['mesh']['world_mesh_dims']} {main_runs[0]['mesh']['world_mesh_shape']}, "
        f"world size {main_runs[0]['world_size']}, grad accum steps {main_runs[0]['grad_accum_steps']}",
        "- relative difference `|a - b| / max(|a|, |b|)`; floor: max over main repeat pairs; arm diff: max over "
        f"main x branch pairs; flag when arm diff > floor (> {ZERO_FLOOR_THRESHOLD:g} when the floor is 0)",
        f"- {len(numeric)} rows compared, {len(flagged)} flagged, max floor "
        f"{max((row['floor'] for row in numeric), default=0):.2e}, max arm diff "
        f"{max((row['arm'] for row in numeric), default=0):.2e}",
        "",
        "## n-gram tables, loss and total norm",
        "",
        *table(ngram_rows + scalar_rows),
        "",
        "## Flagged",
        "",
        *(table(flagged) if flagged else ["None."]),
        "",
        f"## Largest {TOP_ROWS} arm differences",
        "",
        *table(top_rows),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, nargs="+", required=True)
    parser.add_argument("--branch", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    main_runs, branch_runs = load_runs(args.main), load_runs(args.branch)
    all_runs = main_runs + branch_runs
    assert len({run["layout"] for run in all_runs}) == 1, "runs come from different layouts"
    assert len({run["world_size"] for run in all_runs}) == 1, "runs come from different world sizes"
    assert len({json.dumps(run["batch_input_ids_sha256"]) for run in all_runs}) == 1, "runs saw different batches"

    rows = compare(main_runs, branch_runs)
    report = format_report(main_runs, branch_runs, rows)
    out = args.out or RESULTS_DIR / (
        f"grads_{main_runs[0]['layout']}_{main_runs[0]['arm']}"
        f"-r{'-'.join(str(run['repeat']) for run in main_runs)}_vs_{branch_runs[0]['arm']}"
        f"-r{'-'.join(str(run['repeat']) for run in branch_runs)}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"Wrote {out}")
    flagged = [row["name"] for row in rows if row["flag"]]
    raise SystemExit(f"FLAGGED: {len(flagged)} rows" if flagged else 0)


if __name__ == "__main__":
    main()
