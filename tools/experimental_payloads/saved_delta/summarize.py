"""Publish measurements and code hashes, never fixture contents or answers."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize_exact(results, observations, output):
    records = []
    for path in sorted(results.glob("*/result.json")):
        result = json.loads(path.read_text())
        observer = json.loads((observations / path.parent.name / "observer.json").read_text())
        assert observer["returncode"] == 0
        assert result["validation"]["exact_native_episode_fingerprint_matches"]
        assert result["validation"]["native_episode_types_preserved"]
        receive = [row for row in result["rows"] if row["phase"] == "receive"]
        assert [row["complete_native_episodes"] for row in receive] == [32, 32]
        records.append(
            {
                "mode": result["mode"],
                "callbacks": result["callbacks_enabled"],
                "zero_copy_receive": result.get("zero_copy_receive", False),
                "receive_native_return_mean_seconds": statistics.mean(row["seconds"] for row in receive),
                "receive_cycle_seconds": [row["seconds"] for row in receive],
                "materialize_mean_seconds": statistics.mean(row["native_episode_return_seconds"] for row in receive),
                "workflow_measured_seconds": sum(row["seconds"] for row in result["rows"]),
                "receive_max_loop_lag_seconds": max(row["max_loop_lag_seconds"] for row in receive),
                "workflow_max_loop_lag_seconds": max(row["max_loop_lag_seconds"] for row in result["rows"]),
                "parent_peak_rss_gib": max(row["peak_rss_bytes"] for row in result["rows"]) / 1024**3,
                "receiver_tree_peak_pss_gib": observer["receiver_tree_peak_pss_bytes"] / 1024**3,
                "receiver_tree_peak_summed_rss_gib": observer["receiver_tree_peak_rss_bytes"] / 1024**3,
                "node": result["node"],
                "versions": result["versions"],
                "source_hashes": result["source_hashes"],
                "native_episode_fingerprint": result["native_episode_fingerprint"],
                "validation": result["validation"],
                "rows": result["rows"],
                "limits": result["limits"],
            }
        )
    assert len(records) == 10, "Require five decoder cases, three IPC controls and two zero-copy receive controls"
    assert len({record["native_episode_fingerprint"] for record in records}) == 1
    assert len({record["node"] for record in records}) == 1
    (output / "exact-harness-results.json").write_text(
        json.dumps(
            {
                "cases": records,
                "limits": "Original complete-native-return harness, same saved workload and sequential allocation. "
                "Two heap cycles, no confidence intervals. RSS/PSS peaks include setup; fake sender excluded. "
                "Typed Struct maps change raw dictionary insertion order.",
            },
            indent=2,
        )
    )
    fields = [key for key in records[0] if key not in ("versions", "source_hashes", "validation", "rows", "limits")]
    with (output / "exact-harness-results.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields})


def main(args):
    records = []
    for path in sorted(args.results.glob("*/result.json")):
        result = json.loads(path.read_text())
        observer_path = args.observations / path.parent.name / "observer.json"
        observer = json.loads(observer_path.read_text())
        assert observer["returncode"] == 0
        assert all(result["validation"].values())
        receive = [row for row in result["rows"] if row["phase"] == "receive"]
        verification = [row for row in result["rows"] if row["phase"] == "verify"]
        record = {
            "name": path.parent.name,
            "mode": result["mode"],
            "callbacks": result["callbacks"],
            "consumption": result["consumption"],
            "replicas": result["replicas"],
            "cycles": result["cycles"],
            "receive_native_return_mean_seconds": statistics.mean(row["seconds"] for row in receive),
            "receive_cycle_seconds": [row["seconds"] for row in receive],
            "materialize_mean_seconds": statistics.mean(row["materialize_seconds"] for row in receive),
            "verification_mean_seconds": statistics.mean(row["seconds"] for row in verification),
            "workflow_measured_seconds": sum(row["seconds"] for row in result["rows"]),
            "process_elapsed_seconds": observer["elapsed"],
            "receive_max_loop_lag_seconds": max(row["max_loop_lag_seconds"] for row in receive),
            "workflow_max_loop_lag_seconds": max(row["max_loop_lag_seconds"] for row in result["rows"]),
            "parent_peak_rss_gib": max(row["parent_peak_rss_bytes"] for row in result["rows"]) / 1024**3,
            "receiver_tree_peak_pss_gib": observer["receiver_tree_peak_pss_bytes"] / 1024**3,
            "receiver_tree_peak_summed_rss_gib": observer["receiver_tree_peak_rss_bytes"] / 1024**3,
            "complete_episodes_per_cycle": [row["complete_native_episodes"] for row in receive],
            "details": result,
            "observer": observer,
        }
        records.append(record)
    assert len(records) == 14, "Require all seven smoke and seven full cases"
    assert len({record["details"]["fixture_sha256"] for record in records}) == 1
    assert len({record["details"]["native_episode_fingerprint"] for record in records}) == 1
    full = [record for record in records if record["replicas"] == 32]
    baseline = next(record for record in full if record["mode"] == "thread" and not record["callbacks"])
    for record in full:
        record["receive_speedup_vs_thread_no_callbacks"] = (
            baseline["receive_native_return_mean_seconds"] / record["receive_native_return_mean_seconds"]
        )
        if record["mode"] != "deferred":
            assert record["complete_episodes_per_cycle"] == [32, 32]
        else:
            assert record["complete_episodes_per_cycle"] == [round(32 * record["consumption"])] * 2
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "saved-delta-results.json").write_text(
        json.dumps(
            {
                "workload": "Original saved-delta MessagePack workload, same archived native validation",
                "full_cases": full,
                "smoke_cases": [record for record in records if record["replicas"] == 1],
                "limits": "One saved episode, two heap cycles, not independent repeats or confidence intervals. "
                "Deferred 25% changes the return/admission contract; do not rank it as equivalent to full return. "
                "Dictionary insertion order is not preserved by typed Struct decoding. "
                "PSS and RSS include setup, exclude the fake wire sender. "
                "No fixtures, raw episodes, answers or judge verdicts included.",
            },
            indent=2,
        )
    )
    fields = [key for key in full[0] if key not in ("details", "observer")]
    with (args.output / "saved-delta-results.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in full:
            writer.writerow({key: record[key] for key in fields})
    print(
        json.dumps(
            [{key: value for key, value in record.items() if key not in ("details", "observer")} for record in full],
            indent=2,
        )
    )
    if args.exact_results is not None:
        assert args.exact_observations is not None
        summarize_exact(args.exact_results, args.exact_observations, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exact-results", type=Path)
    parser.add_argument("--exact-observations", type=Path)
    main(parser.parse_args())
