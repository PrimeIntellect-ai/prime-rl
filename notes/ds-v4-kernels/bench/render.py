#!/usr/bin/env python
"""Render `bench/raw/*.json` into the markdown tables of `measured.md`.

The raw JSON is regenerable and not committed; this is what turns it into something that is.

    uv run notes/ds-v4-kernels/bench/render.py
"""

import json
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).parent / "raw"
GB = 1 << 30

MODULE_ORDER = [
    "attn-csa",
    "attn-hca",
    "attn-sliding",
    "indexer",
    "indexer-scorer",
    "compressor-csa",
    "compressor-hca",
    "hyperconnection",
    "rmsnorm",
    "rotary",
    "packed-context",
    "decoder-csa",
    "decoder-hca",
]


def load():
    points = defaultdict(dict)
    for path in sorted(RAW.glob("*.json")):
        record = json.loads(path.read_text())
        # The impl belongs in the key: without it two implementations at the same point land on
        # the same entry and whichever file sorts last silently wins. Records written before the
        # axis existed carry no `attn_impl` and are all `eager`.
        key = (record["module"], record.get("ac", "none"), record.get("attn_impl", "eager"), record["seq_len"])
        points[key][record["mode"]] = record
    return points


def gb(value):
    return "-" if value is None else f"{value / GB:.2f}"


def ms(entry):
    return "-" if entry is None else f"{entry['p50']:.2f}"


def spread(entry):
    return "-" if entry is None else f"{entry['p20']:.2f}/{entry['p80']:.2f}"


def main():
    points = load()
    modules = sorted({key[0] for key in points}, key=lambda m: MODULE_ORDER.index(m) if m in MODULE_ORDER else 99)

    lines = []
    for module in modules:
        for ac in sorted({key[1] for key in points if key[0] == module}):
            for impl in sorted({key[2] for key in points if key[0] == module and key[1] == ac}):
                rows = sorted(
                    (key[3], value)
                    for key, value in points.items()
                    if key[0] == module and key[1] == ac and key[2] == impl
                )
                if not rows:
                    continue
                qualifiers = ", ".join(filter(None, ["" if ac == "none" else f"ac={ac}", f"attn={impl}"]))
                title = f"{module} ({qualifiers})"
                first = next(iter(rows[0][1].values()))
                lines.append(f"\n### `{title}`\n")
                lines.append(f"{first.get('module_params', 0) / 1e6:.1f}M parameters. {first.get('note', '')}\n")
                lines.append(
                    "| t | status | fwd peak GB | retained after fwd GB | bwd peak GB "
                    "| fwd ms | fwd+bwd ms | bwd ms | p20/p80 fwd+bwd |"
                )
                lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
                for seq_len, modes in rows:
                    mem = modes.get("memory", {})
                    tim = modes.get("timing", {})
                    base = mem.get("baseline_bytes") or 0
                    fwd = mem.get("fwd_peak_allocated")
                    retained = mem.get("retained_after_fwd")
                    bwd = mem.get("bwd_peak_allocated")
                    lines.append(
                        "| {t} | {s} | {f} | {r} | {b} | {fm} | {fb} | {bm} | {sp} |".format(
                            t=seq_len,
                            s=mem.get("status", "?"),
                            f=gb(None if fwd is None else fwd - base),
                            r=gb(None if retained is None else retained - base),
                            b=gb(None if bwd is None else bwd - base),
                            fm=ms(tim.get("fwd_ms")),
                            fb=ms(tim.get("fwd_bwd_ms")),
                            bm="-" if tim.get("bwd_ms") is None else f"{tim['bwd_ms']:.2f}",
                            sp=spread(tim.get("fwd_bwd_ms")),
                        )
                    )

    ceiling = [
        "\n## OOM ceiling per module\n",
        "Peak allocations are reported net of the module's own parameters and inputs, "
        "which is what `baseline_bytes` records.\n",
        "| module | attn | largest t that fits | first t that OOMs |",
        "|---|---|---:|---:|",
    ]
    for module in modules:
        for impl in sorted({k[2] for k in points if k[0] == module}):
            fitted = [
                k[3]
                for k, v in points.items()
                if k[0] == module and k[2] == impl and v.get("memory", {}).get("status") == "ok"
            ]
            oomed = [
                k[3]
                for k, v in points.items()
                if k[0] == module and k[2] == impl and v.get("memory", {}).get("status") == "oom"
            ]
            ceiling.append(
                f"| `{module}` | {impl} | {max(fitted) if fitted else '-'} "
                f"| {min(oomed) if oomed else 'none in sweep'} |"
            )

    allocations = []
    for path in RAW.glob("*attribution*.json"):
        record = json.loads(path.read_text())
        for entry in record.get("top_allocations", []):
            allocations.append((entry["bytes"], record["module"], record["seq_len"], entry))
    alloc = []
    if allocations:
        allocations.sort(reverse=True, key=lambda item: item[0])
        alloc = [
            "\n## Top allocations across the sweep\n",
            "From the `TorchDispatchMode` allocation log, which keys on storage `data_ptr` so "
            "a view of a storage already counted is not counted twice.\n",
            "| GB | module | t | phase | op | shape | dtype |",
            "|---:|---|---:|---|---|---|---|",
        ]
        for nbytes, module, seq_len, entry in allocations[:20]:
            alloc.append(
                f"| {nbytes / GB:.2f} | `{module}` | {seq_len} | {entry['phase']} | "
                f"`{entry['op']}` | {tuple(entry['shape'])} | {entry['dtype']} |"
            )

    print("\n".join(ceiling + lines + alloc))


if __name__ == "__main__":
    main()
