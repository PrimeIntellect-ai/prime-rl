"""Compare two export_driver.py export records (main vs branch) and write a pass/fail table under results/."""

import argparse
import json
import re
from collections import Counter
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
NGRAM_HF_KEY_RE = re.compile(
    r"\.layers\.(\d+)\.ple\.ple_embedding\."
    r"(?:ngram_embedding\.shard_\d+\.weight|layer_multipliers|ngram_heads_vocab_sizes|ngram_heads_offsets)$"
)
NGRAM_PRIME_KEY_RE = re.compile(r"\.ngram\.tables\.(\d+)\.")
CHUNK_KEY_RE = re.compile(r"\.shard_\d+\.weight$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
NON_LAYER = "non-layer"
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


def table_layer(key: str) -> int | None:
    match = NGRAM_HF_KEY_RE.search(key) or NGRAM_PRIME_KEY_RE.search(key)
    return int(match.group(1)) if match else None


def unit_of(key: str) -> str:
    layer = table_layer(key)
    if layer is None and (match := LAYER_RE.search(key)):
        layer = int(match.group(1))
    return NON_LAYER if layer is None else f"layer {layer}"


def nccl_phases(packets: list[list[dict]]) -> tuple[list[tuple[str, bool]], dict[str, str]]:
    """Label each packet with the send phase it belongs to: the non-layer phase, then one phase per layer.

    A phase opens with the packet the broadcaster sends for it, followed by the row-chunk packets of its tables.
    """
    phase = -2
    packet_phases = []
    key_phase = {}
    for index, packet in enumerate(packets):
        keys = [entry["key"] for entry in packet]
        is_chunk = bool(keys) and all(CHUNK_KEY_RE.search(key) for key in keys)
        if not is_chunk:
            phase += 1
        assert phase >= -1, f"packet {index} is a chunk packet before any phase packet"
        label = NON_LAYER if phase == -1 else f"layer {phase}"
        packet_phases.append((label, is_chunk))
        for key in keys:
            assert key not in key_phase, f"{key} sent twice"
            key_phase[key] = label
    return packet_phases, key_phase


def packet_layout(packets: list[list[dict]]) -> str:
    runs = []
    for index, ((label, is_chunk), packet) in enumerate(zip(nccl_phases(packets)[0], packets)):
        kind = f"{label} chunk" if is_chunk else label
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


def unit_placements(placement: dict) -> dict[str, set]:
    """Collect where each unit's keys other than n-gram table keys are placed; None means not transferred."""
    placements: dict[str, set] = {}
    for key, value in placement.items():
        if table_layer(key) is None and value is not None:
            placements.setdefault(unit_of(key), set()).add(value)
    return placements


def one_to_one_relabeling(main: dict, branch: dict) -> tuple[dict, list[str]]:
    """Map main's placement labels one-to-one onto the branch's over the keys other than n-gram table keys."""
    forward, backward, problems = {}, {}, []
    for key in sorted(main.keys() & branch.keys()):
        if table_layer(key) is not None:
            continue
        main_value, branch_value = main[key], branch[key]
        if forward.setdefault(main_value, branch_value) != branch_value or (
            backward.setdefault(branch_value, main_value) != main_value
        ):
            problems.append(f"{key}: {main_value} -> {branch_value} breaks a one-to-one relabeling")
    return {old: new for old, new in forward.items() if old != new}, problems


def compare_placement(
    label: str, main: dict, branch: dict, relabel: bool = False
) -> list[tuple[str, bool, list[str], bool]]:
    """Require identical placement for every key except n-gram table keys.

    A table key may instead move from where its layer's other keys go on main to where the non-layer keys go on
    the branch. With `relabel`, placement labels are interchangeable names (owner ranks), so the other keys only
    need the same grouping under a one-to-one renaming.
    """
    problems = key_set_problems(main, branch)
    main_units, branch_units = unit_placements(main), unit_placements(branch)
    for arm, units in (("main", main_units), ("branch", branch_units)):
        problems += [
            f"{arm} {unit} spans {sorted(map(str, values))}" for unit, values in units.items() if len(values) > 1
        ]
    other_problems = list(problems)
    other_summary = []
    if relabel:
        relabeling, problems = one_to_one_relabeling(main, branch)
        other_problems += problems
        if relabeling:
            other_summary = [f"same grouping, relabeled {relabeling}"]
    table_problems = []
    moved_layers = Counter()
    unchanged = 0
    for key in sorted(main.keys() & branch.keys()):
        main_value, branch_value = main[key], branch[key]
        layer = table_layer(key)
        if layer is None:
            if main_value != branch_value and not relabel:
                other_problems.append(f"{key}: {main_value} != {branch_value}")
            continue
        if main_value == branch_value:
            unchanged += 1
        elif main_units.get(f"layer {layer}") == {main_value} and branch_units.get(NON_LAYER) == {branch_value}:
            moved_layers[layer] += 1
        else:
            table_problems.append(f"{key}: {main_value} -> {branch_value}, not layer {layer} -> non-layer")
    layers = ",".join(map(str, sorted(moved_layers)))
    move_summary = [f"{moved_layers.total()} moved layers {layers} -> non-layer, {unchanged} unchanged"]
    return [
        (f"{label}, non-table keys", not other_problems, other_problems or other_summary, False),
        (f"{label}, n-gram table keys", not table_problems, table_problems or move_summary, False),
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
    main_tensors = {entry["key"]: entry for packet in main_packets for entry in packet}
    branch_tensors = {entry["key"]: entry for packet in branch_packets for entry in packet}
    tensor_problems = key_set_problems(main_tensors, branch_tensors) + tensor_mismatches(main_tensors, branch_tensors)
    rows.append(("NCCL dtype, shape, sha256 per key", not tensor_problems, tensor_problems, False))

    main_phases, main_key_phase = nccl_phases(main_packets)
    branch_phases, branch_key_phase = nccl_phases(branch_packets)
    rows += compare_placement("NCCL send phase", main_key_phase, branch_key_phase)

    def chunk_sequence(packets, phases):
        return [[entry["key"] for entry in packet] for packet, (_, is_chunk) in zip(packets, phases) if is_chunk]

    main_chunks, branch_chunks = (
        chunk_sequence(main_packets, main_phases),
        chunk_sequence(branch_packets, branch_phases),
    )
    chunk_problems = [] if main_chunks == branch_chunks else [f"{len(main_chunks)} vs {len(branch_chunks)} packets"]
    rows.append(("NCCL chunk packet sequence", not chunk_problems, chunk_problems, False))

    def phase_key_order(packets, phases):
        return {
            label: [entry["key"] for entry in packet if table_layer(entry["key"]) is None]
            for packet, (label, is_chunk) in zip(packets, phases)
            if not is_chunk
        }

    main_order, branch_order = (
        phase_key_order(main_packets, main_phases),
        phase_key_order(branch_packets, branch_phases),
    )
    order_notes = [
        f"{label}: same non-table keys, different order"
        for label in main_order.keys() & branch_order.keys()
        if main_order[label] != branch_order[label] and set(main_order[label]) == set(branch_order[label])
    ]
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
        "| Check                                     | Result  | Detail                                         |",
        "| ----------------------------------------- | ------- | ---------------------------------------------- |",
    ]
    for label, passed, problems, informational in rows:
        result = ("same" if passed else "differs") if informational else ("pass" if passed else "FAIL")
        detail = problems[0] if len(problems) == 1 else (f"{len(problems)} findings" if problems else "")
        lines.append(f"| {label:<41} | {result:<7} | {detail[:46]:<46} |")
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
    rows += compare_placement(
        "NIXL group", map_keys(main_record["nixl"]["key_to_group"]), map_keys(branch_record["nixl"]["key_to_group"])
    )
    rows += compare_placement(
        "Save partition owner",
        map_keys(main_record["partition"]),
        map_keys(branch_record["partition"]),
        relabel=True,
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
