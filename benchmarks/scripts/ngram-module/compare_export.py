"""Compare two export_driver.py export records (main vs branch) and write a pass/fail table under results/."""

import argparse
import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NGRAM_NAME_MAP = [
    (re.compile(r"\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight$"), r".ngram.tables.\1.weight"),
    (
        re.compile(
            r"\.layers\.(\d+)\.ple\.ple_embedding\.(layer_multipliers|ngram_heads_vocab_sizes|ngram_heads_offsets)$"
        ),
        r".ngram.tables.\1.\2",
    ),
]
SHARD_RE = re.compile(r"\.layers\.(\d+)\..*\.shard_\d+\.weight$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
MAX_LISTED = 20


def map_prime_name(name: str) -> str:
    for pattern, replacement in NGRAM_NAME_MAP:
        name = pattern.sub(replacement, name)
    return name


def map_keys(mapping: dict) -> dict:
    mapped = {}
    for key, value in mapping.items():
        new_key = map_prime_name(key)
        assert new_key not in mapped, f"{key} maps onto an existing key {new_key}"
        mapped[new_key] = value
    return mapped


def packet_kind(packet: list[dict]) -> str:
    keys = [entry["key"] for entry in packet]
    if not keys:
        return "empty"
    shard_layers = {match.group(1) for key in keys if (match := SHARD_RE.search(key))}
    if len(shard_layers) == 1 and all(SHARD_RE.search(key) for key in keys):
        return f"layer {shard_layers.pop()} chunk"
    layers = sorted({int(match.group(1)) for key in keys if (match := LAYER_RE.search(key))})
    if not layers:
        return "non-layer"
    return "layer " + ",".join(map(str, layers))


def packet_layout(packets: list[list[dict]]) -> str:
    runs = []
    for index, packet in enumerate(packets):
        kind = packet_kind(packet)
        if runs and runs[-1][0] == kind:
            runs[-1][2] = index
            runs[-1][3] += len(packet)
        else:
            runs.append([kind, index, index, len(packet)])
    parts = []
    for kind, first, last, num_keys in runs:
        span = f"{first}" if first == last else f"{first}-{last}"
        parts.append(f"{span}: {kind} ({num_keys} keys)")
    return "; ".join(parts)


def tensor_mismatches(main: dict[str, dict], branch: dict[str, dict]) -> list[str]:
    problems = []
    for key in sorted(main.keys() & branch.keys()):
        for field in ("dtype", "shape", "sha256"):
            if main[key][field] != branch[key][field]:
                problems.append(f"{key}: {field} {main[key][field]} != {branch[key][field]}")
    return problems


def key_set_problems(main_keys, branch_keys) -> list[str]:
    return [f"only in main: {key}" for key in sorted(set(main_keys) - set(branch_keys))] + [
        f"only in branch: {key}" for key in sorted(set(branch_keys) - set(main_keys))
    ]


def compare_nccl(main: dict, branch: dict) -> list[tuple[str, bool, list[str], bool]]:
    rows = []
    rows.append(
        ("NCCL integer", main["integer"] == branch["integer"], [f"{main['integer']} vs {branch['integer']}"], False)
    )
    main_packets, branch_packets = main["packets"], branch["packets"]
    rows.append(
        (
            "NCCL packet count",
            len(main_packets) == len(branch_packets),
            [f"{len(main_packets)} vs {len(branch_packets)}"],
            False,
        )
    )
    set_problems = []
    order_notes = []
    for index, (main_packet, branch_packet) in enumerate(zip(main_packets, branch_packets)):
        main_keys = [entry["key"] for entry in main_packet]
        branch_keys = [entry["key"] for entry in branch_packet]
        if set(main_keys) != set(branch_keys):
            set_problems.extend(f"packet {index}: {problem}" for problem in key_set_problems(main_keys, branch_keys))
        elif main_keys != branch_keys:
            order_notes.append(f"packet {index}: same keys, different order")
    rows.append(("NCCL packet order and key sets", not set_problems, set_problems, False))
    main_tensors = {entry["key"]: entry for packet in main_packets for entry in packet}
    branch_tensors = {entry["key"]: entry for packet in branch_packets for entry in packet}
    tensor_problems = key_set_problems(main_tensors, branch_tensors) + tensor_mismatches(main_tensors, branch_tensors)
    rows.append(("NCCL dtype, shape, sha256 per key", not tensor_problems, tensor_problems, False))
    rows.append(("NCCL intra-packet key order (info)", not order_notes, order_notes, True))
    return rows


def compare_filesystem(main: dict, branch: dict) -> list[tuple[str, bool, list[str], bool]]:
    set_problems = key_set_problems(main["keys"], branch["keys"])
    tensor_problems = tensor_mismatches(main["keys"], branch["keys"])
    files_note = [f"{main['num_files']} vs {branch['num_files']} files"]
    return [
        ("Filesystem key set", not set_problems, set_problems, False),
        ("Filesystem dtype, shape, sha256 per key", not tensor_problems, tensor_problems, False),
        ("Filesystem shard file count (info)", main["num_files"] == branch["num_files"], files_note, True),
    ]


def compare_assignment(label: str, main: dict, branch: dict) -> tuple[str, bool, list[str], bool]:
    main_mapped, branch_mapped = map_keys(main), map_keys(branch)
    problems = key_set_problems(main_mapped, branch_mapped)
    problems += [
        f"{key}: {main_mapped[key]} != {branch_mapped[key]}"
        for key in sorted(main_mapped.keys() & branch_mapped.keys())
        if main_mapped[key] != branch_mapped[key]
    ]
    return (label, not problems, problems, False)


def format_report(main_path: Path, branch_path: Path, main: dict, branch: dict, rows: list) -> str:
    lines = [
        f"# Export identity: {main['layout']}",
        "",
        f"- main: `{main_path}` ({main['arm']} at `{main['source']['commit'][:9]}`, dirty={main['source']['dirty']})",
        f"- branch: `{branch_path}` ({branch['arm']} at `{branch['source']['commit'][:9]}`, "
        f"dirty={branch['source']['dirty']})",
        f"- world size {main['world_size']} vs {branch['world_size']}, mesh {main['mesh']['world_mesh_dims']} "
        f"{main['mesh']['world_mesh_shape']} vs {branch['mesh']['world_mesh_dims']} {branch['mesh']['world_mesh_shape']}",
        "",
        "| Check                                   | Result  | Detail                                         |",
        "| --------------------------------------- | ------- | ---------------------------------------------- |",
    ]
    for label, passed, problems, informational in rows:
        result = ("same" if passed else "differs") if informational else ("pass" if passed else "FAIL")
        detail = problems[0] if len(problems) == 1 else (f"{len(problems)} findings" if problems else "")
        lines.append(f"| {label:<39} | {result:<7} | {detail[:46]:<46} |")
    lines += ["", "## NCCL packet layout", "", f"- main: {packet_layout(main['nccl']['packets'])}"]
    lines += [f"- branch: {packet_layout(branch['nccl']['packets'])}", ""]
    lines += ["## n-gram tables (trainer side)", ""]
    for arm, record in (("main", main), ("branch", branch)):
        for name, layout in record["ngram_tables"].items():
            lines.append(f"- {arm} `{name}`: {layout}")
    for label, passed, problems, _ in rows:
        if problems and not (len(problems) == 1 and passed):
            lines += ["", f"## {label}", ""]
            lines += [f"- {problem}" for problem in problems[:MAX_LISTED]]
            if len(problems) > MAX_LISTED:
                lines.append(f"- ... {len(problems) - MAX_LISTED} more")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_json", type=Path)
    parser.add_argument("branch_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    main_record = json.loads(args.main_json.read_text())
    branch_record = json.loads(args.branch_json.read_text())
    assert main_record["mode"] == branch_record["mode"] == "export"
    rows = compare_nccl(main_record["nccl"], branch_record["nccl"])
    rows += compare_filesystem(main_record["filesystem"], branch_record["filesystem"])
    rows.append(
        (
            "NIXL group names",
            main_record["nixl"]["group_names"] == branch_record["nixl"]["group_names"],
            [f"{main_record['nixl']['group_names']} vs {branch_record['nixl']['group_names']}"],
            False,
        )
    )
    rows.append(
        compare_assignment(
            "NIXL group per key", main_record["nixl"]["key_to_group"], branch_record["nixl"]["key_to_group"]
        )
    )
    rows.append(
        compare_assignment("Save partition owner per key", main_record["partition"], branch_record["partition"])
    )

    report = format_report(args.main_json, args.branch_json, main_record, branch_record, rows)
    out = args.out or RESULTS_DIR / (
        f"export_{main_record['layout']}_{main_record['arm']}-r{main_record['repeat']}"
        f"_vs_{branch_record['arm']}-r{branch_record['repeat']}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"Wrote {out}")
    failed = [label for label, passed, _, informational in rows if not passed and not informational]
    raise SystemExit(f"FAIL: {', '.join(failed)}" if failed else 0)


if __name__ == "__main__":
    main()
