"""Write the markdown summary of a DSv4 sparse attention baseline from bench.py and stream.py results.

usage (repo root):
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/summarize.py BENCH.json STREAM.json [--corpus DIR]
      > SUMMARY.md

Prints provenance, the synthetic-corpus caveat, a headline table of the single-row (cp1) items, a per-arm table
of the same items when the run has more than one arm, host overhead, correctness margins, the dynamic stream, and
per-item histograms of valid slots per query, which show how dynamic the corpus is. The full per-item tables come from `bench.py --compare` and `stream.py --compare`.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from common import BASELINE_BACKEND, DEFAULT_CORPUS_DIR, load_indices, load_manifest, select_items, slot_coverage

HISTOGRAM_BIN = 64
HISTOGRAM_BINS = 10
REFERENCE_ARM = "flashmla_fwd_ref"


def ms(us: float) -> str:
    return f"{us / 1e3:.2f}"


def timing(entry: dict, mode: str) -> dict | None:
    return entry.get("timing", {}).get(mode)


def print_provenance(bench: dict) -> None:
    prov = bench["provenance"]
    query = prov["gpu"]["query"]
    print(f"- GPU: {query['name']}, driver {query['driver_version']}, power limit {query['power.limit']},")
    print(f"  max SM clock {query['clocks.max.sm']}, host `{prov['hostname']}`.")
    dirty = " (dirty)" if prov["git"]["dirty"] else ""
    print(
        f"- Code: git `{prov['git']['sha'][:9]}`{dirty}, torch {prov['versions']['torch']}, tilelang {prov['versions']['tilelang']}."
    )
    print(f"- Corpus hash `{prov['corpus_hash'][:16]}`, {len(bench['items'])} grid items.")
    settings = bench["settings"]
    print(
        f"- Settings: {settings['rounds']} ABBA rounds x {settings['iters']} calls, {settings['warmup']} warmup calls,"
        f" L2 flushed and GPU idle before each call."
    )


def print_headline(bench: dict) -> None:
    print("## Single-row items (cp1)\n")
    print("Op-boundary time per call in ms (lower is better). `gpu` is the GPU busy time of the same call; the")
    print("difference is host overhead. TFLOP/s counts useful FLOPs over op-boundary time (higher is better);")
    print(f"`% peak` is against {bench['peak_dense_bf16_tflops']} dense BF16 TFLOP/s. `ref/TL` is the FlashMLA")
    print("forward reference's op-boundary time over tilelang's forward (below 1 means the reference is faster).\n")
    print("| item | fwd | fwd gpu | f+b | f+b gpu | f+b TFLOP/s | % peak | ref/TL fwd |")
    print("|---|---|---|---|---|---|---|---|")
    for item_id, item in bench["items"].items():
        if item["cp"] != 1:
            continue
        tilelang = item["backends"][BASELINE_BACKEND]
        fwd, fwd_bwd = timing(tilelang, "fwd"), timing(tilelang, "fwd_bwd")
        if fwd is None or fwd_bwd is None:
            print(f"| {item_id} | not timed | | | | | | |")
            continue
        rate = item["useful_flops"]["fwd_bwd"] / fwd_bwd["op_us"]["median"] / 1e6
        reference = timing(item["backends"].get(REFERENCE_ARM, {}), "fwd")
        ratio = f"{reference['op_us']['median'] / fwd['op_us']['median']:.2f}" if reference else "-"
        print(
            f"| {item_id} | {ms(fwd['op_us']['median'])} | {ms(fwd['gpu']['us']['median'])} | "
            f"{ms(fwd_bwd['op_us']['median'])} | {ms(fwd_bwd['gpu']['us']['median'])} | {rate:.0f} | "
            f"{100 * rate / bench['peak_dense_bf16_tflops']:.1f} | {ratio} |"
        )
    print()


def print_arms(bench: dict) -> None:
    arms = list(dict.fromkeys(name for item in bench["items"].values() for name in item["backends"]))
    if len(arms) < 2:
        return
    print("## Every arm on the single-row items (cp1)\n")
    print("Time per call in ms (lower is better): op-boundary, then GPU busy time. TFLOP/s counts useful FLOPs (valid")
    print("slots only) over op-boundary time (higher is better). `exec/useful` is the FLOPs of the slots the arm's")
    print("tiles touch over useful FLOPs (1 is no wasted work). `-` marks a mode the arm does not have.\n")
    print("| item | arm | fwd | fwd gpu | fwd TFLOP/s | exec/useful fwd | f+b | f+b gpu | f+b TFLOP/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for item_id, item in bench["items"].items():
        if item["cp"] != 1:
            continue
        for name in arms:
            entry = item["backends"].get(name, {})
            cells = []
            for mode, flops_key in (("fwd", "fwd"), ("fwd_bwd", "fwd_bwd")):
                t = timing(entry, mode)
                if t is None:
                    cells.append("- | - | -" if mode == "fwd_bwd" else "- | - | - | -")
                    continue
                rate = item["useful_flops"][flops_key] / t["op_us"]["median"] / 1e6
                cell = f"{ms(t['op_us']['median'])} | {ms(t['gpu']['us']['median'])} | {rate:.0f}"
                if mode == "fwd":
                    cell += f" | {entry.get('executed_over_useful', {}).get('fwd', float('nan')):.2f}"
                cells.append(cell)
            print(f"| {item_id} | {name} | {cells[0]} | {cells[1]} |")
    print()


def print_host_overhead(bench: dict) -> None:
    print("## Host overhead (op-boundary minus GPU busy time), µs\n")
    print("| arm | mode | min | median | max | items where it exceeds GPU time |")
    print("|---|---|---|---|---|---|")
    arms = sorted({name for item in bench["items"].values() for name in item["backends"]})
    for name in arms:
        for mode in ("fwd", "fwd_bwd"):
            pairs = [
                (t["op_us"]["median"] - t["gpu"]["us"]["median"], t["gpu"]["us"]["median"])
                for item in bench["items"].values()
                if (t := timing(item["backends"].get(name, {}), mode)) is not None
            ]
            if not pairs:
                continue
            host = [pair[0] for pair in pairs]
            dominated = sum(host_us > gpu_us for host_us, gpu_us in pairs)
            print(
                f"| {name} | {mode} | {min(host):.0f} | {statistics.median(host):.0f} | {max(host):.0f} | "
                f"{dominated} of {len(pairs)} |"
            )
    print()


def print_correctness(bench: dict) -> None:
    print("## Correctness gate\n")
    print("Largest relative error over all items (max deviation over the reference's max magnitude), and its bound.\n")
    print("| arm | basis | tensor | max relative error | bound | failures |")
    print("|---|---|---|---|---|---|")
    worst: dict[tuple, list] = {}
    for item in bench["items"].values():
        for name, entry in item["backends"].items():
            for basis, checks in entry.get("correctness", {}).items():
                for tensor, check in checks.items():
                    if name == BASELINE_BACKEND and basis == "vs_tilelang":
                        continue
                    record = worst.setdefault((name, basis, tensor), [0.0, check["rtol"], 0])
                    record[0] = max(record[0], check["relative_error"])
                    record[2] += not check["ok"]
    for (name, basis, tensor), (error, rtol, failures) in sorted(worst.items()):
        print(f"| {name} | {basis} | {tensor} | {error:.2e} | {rtol:.0e} | {failures} |")
    raised = [
        (item_id, name)
        for item_id, item in bench["items"].items()
        for name, e in item["backends"].items()
        if "error" in e
    ]
    print(f"\nArms that raised: {len(raised)}. {', '.join(f'{i} {n}' for i, n in raised)}\n")


def print_stream(stream: dict) -> None:
    print("## Dynamic stream\n")
    print(f"{len(stream['stream'])} items replayed once each in a fresh process with fresh compile caches; seconds,")
    print("lower is better. `rest` sums every item after the first, so it excludes per-process costs.\n")
    print("| arm | phase | first item | rest | compiles | disk loads | prime_rl import | TileLang compile |")
    print("|---|---|---|---|---|---|---|---|")
    for name, replayed in stream["backends"].items():
        for phase, result in replayed["phases"].items():
            startup = result["startup"]
            imports = sum(seconds for stage, seconds in startup.items() if stage.startswith("import prime_rl"))
            print(
                f"| {name} | {phase} | {result['first_item_s']:.2f} | {result['remainder_s']:.3f} | "
                f"{result['counts'].get('compiles', 0)} | {result['counts'].get('disk_loads', '-')} | {imports:.1f} | "
                f"{startup.get('first_item_jit_compile_s', 0.0):.1f} |"
            )
    print()


def print_histograms(corpus_dir: Path, manifest: dict, item_ids: list[str]) -> None:
    print("## Valid slots per query\n")
    edges = [HISTOGRAM_BIN * index for index in range(HISTOGRAM_BINS + 1)]
    print(f"Share of an item's queries (%) whose valid-slot count falls in each {HISTOGRAM_BIN}-slot bin; the last")
    print("bin is closed. `slots` is the item's gather width, `med` the median valid count per query.\n")
    header = " | ".join(f"{low}-" for low in edges[:-1])
    print(f"| item | slots | med | {header} |")
    print("|---|---|---|" + "---|" * HISTOGRAM_BINS)
    for item in select_items(manifest, item_ids):
        n_valid = slot_coverage(load_indices(corpus_dir, item))["n_valid"]
        bins = torch.clamp(n_valid // HISTOGRAM_BIN, max=HISTOGRAM_BINS - 1)
        shares = torch.bincount(bins, minlength=HISTOGRAM_BINS).float() / n_valid.numel() * 100
        cells = " | ".join("" if share == 0 else f"{share:.0f}" for share in shares.tolist())
        print(f"| {item.id} | {item.n_slots} | {int(n_valid.float().median())} | {cells} |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", type=Path)
    parser.add_argument("stream", type=Path)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args()

    bench, stream = (json.loads(path.read_text()) for path in (args.bench, args.stream))
    manifest = load_manifest(args.corpus)
    hashes = {bench["provenance"]["corpus_hash"], stream["provenance"]["corpus_hash"], manifest["corpus_hash"]}
    if len(hashes) > 1:
        raise SystemExit(f"bench, stream and corpus disagree on the corpus hash: {sorted(hashes)}")

    print(f"# DSv4 sparse attention baseline: `{bench['label']}`\n")
    print("Caveat: the corpus is synthetic. Its CSA picks come from a random-weight Lightning Indexer and are")
    print("near-uniform over the readable entries, while a trained indexer favors recent and neighboring entries,")
    print("so CSA gather locality here is pessimistic. Sliding and HCA indices do not depend on weights.\n")
    print_provenance(bench)
    print()
    print_headline(bench)
    print_arms(bench)
    print_host_overhead(bench)
    print_correctness(bench)
    print_stream(stream)
    print_histograms(args.corpus, manifest, list(bench["items"]) + stream["stream"])


if __name__ == "__main__":
    main()
