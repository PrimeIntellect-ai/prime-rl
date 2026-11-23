"""Compare e2e runs against main: per-step metrics against the floor of all main runs, received weight hashes, checkpoint exports and compile logs."""

import argparse
import hashlib
import itertools
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from safetensors_hash import hash_safetensors_dir

ARMS = ("main1", "main2", "main3", "pr1", "pr2", "pr2r2")
ARM_NAME = re.compile(r"^(main|pr)\d")
REFERENCE = "main1"
PRIMARY = re.compile(r"(^|[:/])(loss/mean|optim/grad_norm|mismatch_kl/all/mean|entropy/all/mean|max_vio/mean|reward/.*mean)$")
NOT_NUMERICS = re.compile(r"time|perf/|throughput|mfu|memory|mem/|tokens_per|per_s(ec)?$|latency|lag|inflight|queue|concurrency|progress/|_s$")
NOISE_FALLBACK = {"rl-dense-fs": "rl-dense-nccl", "rl-moe-fs": "rl-moe-nccl"}
RL_PAIRS = (("rl-dense-nccl", "rl-dense-fs"), ("rl-moe-nccl", "rl-moe-fs"))
EXACT_EXPORT_FILES = re.compile(r"\.safetensors$|^model\.safetensors\.index\.json$|^config\.json$")
CKPTS = ("ckpt-dense", "ckpt-moe")


def load_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "monitors" / "file" / "metrics.jsonl"
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        step = row.get("step")
        if step is None:
            continue
        producer = row.get("producer", "trainer")
        for key, value in row.items():
            if key in ("step", "producer") or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            by_step[step][f"{producer}:{key}"] = value
    return dict(by_step)


def step_times(metrics: dict[int, dict[str, float]]) -> list[float]:
    times = [(step, row["trainer:time"]) for step, row in sorted(metrics.items()) if "trainer:time" in row]
    return [later - earlier for (_, earlier), (step, later) in zip(times, times[1:]) if step >= 3]


def diff_series(left: dict, right: dict, key: str) -> dict | None:
    steps = sorted(step for step in set(left) & set(right) if key in left[step] and key in right[step])
    if not steps:
        return None
    deltas = [abs(left[step][key] - right[step][key]) for step in steps]
    relative = [delta / max(abs(left[step][key]), 1e-30) for delta, step in zip(deltas, steps)]
    first = next((step for step, delta in zip(steps, deltas) if delta != 0), None)
    third = max(1, len(deltas) // 3)
    early, late = statistics.fmean(deltas[:third]), statistics.fmean(deltas[-third:])
    return {
        "steps": len(steps),
        "first_diff": first,
        "max_abs": max(deltas),
        "max_rel": max(relative),
        "growth": late / early if early else (float("inf") if late else 1.0),
    }


def arm_runs(root: Path, cell: str) -> dict[str, dict]:
    return {
        path.name: load_metrics(path / cell)
        for path in sorted(root.iterdir())
        if ARM_NAME.match(path.name) and (path / cell / "monitors").is_dir()
    }


def floor_of(runs: dict[str, dict], key: str) -> float | None:
    mains = [name for name in runs if name.startswith("main")]
    gaps = [diff["max_abs"] for a, b in itertools.combinations(mains, 2) if (diff := diff_series(runs[a], runs[b], key))]
    return max(gaps) if gaps else None


def verdict(gap: float, floor: float | None) -> str:
    if gap == 0:
        return "bitwise"
    if floor is None:
        return "differs, no noise floor"
    if floor == 0:
        return "DIVERGES (main is bitwise)"
    ratio = gap / floor
    if ratio <= 1:
        return f"within floor ({ratio:.2f}x)"
    if ratio <= 2:
        return f"near floor ({ratio:.2f}x)"
    return f"ABOVE FLOOR ({ratio:.2f}x)"


def report_metrics(root: Path, cells: list[str]) -> None:
    print("## Per-step metrics: each non-main run against its nearest main run, floor = max gap among all main runs\n")
    print(f"{'cell':<18} {'run':<7} {'metric':<42} {'nearest':>8} {'max|d|':>10} {'floor':>10} {'mains':>5} "
          f"{'growth':>7}  verdict")
    for cell in cells:
        runs = arm_runs(root, cell)
        mains = [name for name in runs if name.startswith("main")]
        if not mains:
            continue
        floor_runs = runs if len(mains) > 1 or cell not in NOISE_FALLBACK else arm_runs(root, NOISE_FALLBACK[cell])
        floor_mains = sum(1 for name in floor_runs if name.startswith("main"))
        keys = sorted({key for row in runs[mains[0]].values() for key in row if PRIMARY.search(key)})
        for key in keys:
            floor = floor_of(floor_runs, key)
            floor_text = f"{floor:.2e}" if floor is not None else "-"
            for name, metrics in runs.items():
                if name.startswith("main"):
                    continue
                diffs = {main: diff_series(runs[main], metrics, key) for main in mains}
                diffs = {main: diff for main, diff in diffs.items() if diff}
                if not diffs:
                    print(f"{cell:<18} {name:<7} {key:<42} MISSING")
                    continue
                nearest = min(diffs, key=lambda main: diffs[main]["max_abs"])
                diff = diffs[nearest]
                print(f"{cell:<18} {name:<7} {key:<42} {nearest:>8} {diff['max_abs']:>10.2e} {floor_text:>10} "
                      f"{floor_mains:>5} {diff['growth']:>7.2f}  {verdict(diff['max_abs'], floor)}")
        for name in ("main1", "pr1", "pr2"):
            times = step_times(runs.get(name, {}))
            if times:
                print(f"{cell:<18} {name:<7} step wall time (steps>=3): median {statistics.median(times):.2f} s, "
                      f"max {max(times):.2f} s")
        print()


def worker_digest(worker: dict) -> str:
    items = sorted((name, tensor["shape"], tensor["dtype"], tensor["sha256"]) for name, tensor in worker.items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def tensor_diff(left: dict, right: dict, limit: int = 8) -> str:
    names = sorted(set(left) | set(right))
    changed = [name for name in names if left.get(name, {}).get("sha256") != right.get(name, {}).get("sha256")]
    shapes = [name for name in changed if name in left and name in right and
              (left[name]["shape"], left[name]["dtype"]) != (right[name]["shape"], right[name]["dtype"])]
    only = sorted(set(left) ^ set(right))
    return f"{len(changed)}/{len(names)} tensors differ (layout changes {len(shapes)}, unmatched names {only[:4]}): {changed[:limit]}"


def load_probe(root: Path, arm: str, cell: str) -> dict:
    path = root / arm / f"{cell}.probe" / "fingerprints.json"
    return json.loads(path.read_text()) if path.exists() else {}


def report_weights(root: Path) -> None:
    print("## Weights received by vLLM (worker fingerprints), filesystem broadcast files, conversion cache\n")
    for pair in RL_PAIRS:
        for cell in pair:
            reference = load_probe(root, REFERENCE, cell)
            for arm in ARMS:
                probe = load_probe(root, arm, cell)
                if not probe:
                    continue
                v0 = sorted({worker_digest(worker) for worker in probe.get("v0", [])})
                pre = sorted({worker_digest(worker) for worker in probe.get("pre", [])})
                line = (f"{cell:<14} {arm:<6} v0 workers={len(probe.get('v0', []))} digests={v0} valid={probe.get('v0_valid')}"
                        f" | pre digests={pre} valid={probe.get('pre_valid')} pre==v0={pre == v0}")
                if arm != REFERENCE and reference.get("v0") and probe.get("v0"):
                    same = v0 == sorted({worker_digest(worker) for worker in reference["v0"]})
                    line += f" | v0==main1: {same}"
                    if not same:
                        line += "\n    " + tensor_diff(reference["v0"][0], probe["v0"][0])
                print(line)
                sender = root / arm / f"{cell}.probe" / "fs_step0_tensors.json"
                reference_sender = root / REFERENCE / f"{cell}.probe" / "fs_step0_tensors.json"
                if sender.exists() and arm != REFERENCE and reference_sender.exists():
                    left, right = json.loads(reference_sender.read_text()), json.loads(sender.read_text())
                    print(f"{cell:<14} {arm:<6} filesystem step_0 files == main1: {left == right}"
                          + ("" if left == right else "\n    " + tensor_diff(left, right)))
                conversion = root / arm / "conv" / cell / "prime"
                reference_conversion = root / REFERENCE / "conv" / cell / "prime"
                if conversion.is_dir() and arm != REFERENCE and reference_conversion.is_dir():
                    left, right = hash_safetensors_dir(reference_conversion), hash_safetensors_dir(conversion)
                    print(f"{cell:<14} {arm:<6} HF->prime conversion cache == main1: {left == right}"
                          + ("" if left == right else "\n    " + tensor_diff(left, right)))
        nccl, fs = pair
        for arm in ARMS:
            left, right = load_probe(root, arm, nccl).get("v0"), load_probe(root, arm, fs).get("v0")
            if left and right:
                same = {worker_digest(worker) for worker in left} == {worker_digest(worker) for worker in right}
                print(f"{nccl} vs {fs} {arm:<6} v0 identical across transports: {same}")
        print()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def json_diff(left: dict, right: dict) -> list[str]:
    return [f"{key}: {left.get(key, '(missing)')!r} -> {right.get(key, '(missing)')!r}"
            for key in sorted(set(left) | set(right)) if left.get(key, "(missing)") != right.get(key, "(missing)")]


def report_exports(root: Path) -> None:
    print("## Checkpoint exports (tools/convert_dcp_to_bf16.py of main1's step_10)\n")
    for ckpt in CKPTS:
        name = f"{ckpt}-main-step10"
        reference = root / REFERENCE / "export" / name
        source = root / "source" / ckpt
        source_gen = source / "generation_config.json"
        if not reference.is_dir():
            continue
        model_name = (source / "model_name").read_text().strip() if (source / "model_name").exists() else "?"
        print(f"{ckpt}: source model {model_name}, source generation_config.json "
              f"{'present' if source_gen.exists() else 'absent'}")
        main_gen = reference / "generation_config.json"
        if main_gen.exists():
            print(f"  main1 derived generation_config.json: {json.loads(main_gen.read_text())}")
            if source_gen.exists():
                print(f"  main1 vs source: {json_diff(json.loads(source_gen.read_text()), json.loads(main_gen.read_text()))}")
        for arm in ARMS:
            export = root / arm / "export" / name
            if not export.is_dir():
                continue
            status = (root / arm / "export" / f"{name}.hfload").read_text().strip() if (root / arm / "export" / f"{name}.hfload").exists() else "?"
            own = root / arm / "export" / f"{ckpt}-own-step20-by-main.hfload"
            own_status = own.read_text().strip() if own.exists() else "-"
            print(f"  {arm:<6} check_hf_load: {status} | own step_20 exported by main's tool, check_hf_load: {own_status}")
            if arm == REFERENCE:
                continue
            files = sorted({path.name for path in reference.iterdir()} | {path.name for path in export.iterdir()})
            for file in files:
                left, right = reference / file, export / file
                if file == "generation_config.json":
                    continue
                if not left.exists() or not right.exists():
                    print(f"    {file}: only in {'main1' if left.exists() else arm}")
                    continue
                if file_sha(left) == file_sha(right):
                    continue
                kind = "MUST MATCH" if EXACT_EXPORT_FILES.search(file) else "info"
                detail = ""
                if file.endswith(".json"):
                    detail = str(json_diff(json.loads(left.read_text()), json.loads(right.read_text()))[:6])
                print(f"    {file}: bytes differ [{kind}] {detail}")
            exact = all(file_sha(reference / file) == file_sha(export / file)
                        for file in files if EXACT_EXPORT_FILES.search(file) and (reference / file).exists() and (export / file).exists())
            print(f"    safetensors + index + config.json byte-identical to main1: {exact}")
            arm_gen = export / "generation_config.json"
            if arm.startswith("pr"):
                if source_gen.exists():
                    if not arm_gen.exists():
                        print("    generation_config.json: MISSING (source has one)")
                    else:
                        same_bytes = file_sha(arm_gen) == file_sha(source_gen)
                        same_json = json.loads(arm_gen.read_text()) == json.loads(source_gen.read_text())
                        print(f"    generation_config.json vs source: bytes equal {same_bytes}, JSON equal {same_json}"
                              + ("" if same_json else f" {json_diff(json.loads(source_gen.read_text()), json.loads(arm_gen.read_text()))}"))
                else:
                    print(f"    generation_config.json: {'UNEXPECTED (source has none)' if arm_gen.exists() else 'absent, as the source'}")
            elif arm_gen.exists() and main_gen.exists():
                print(f"    generation_config.json == main1: {file_sha(arm_gen) == file_sha(main_gen)}")
        print()


TRACING = re.compile(r"torchdynamo start tracing (\S+) (\S+):\d+")
RECOMPILE = re.compile(r"Recompiling function (\S+) in (\S+):\d+")
GUARD = re.compile(r"^\s*-\s+\d+/\d+:\s+(.*)$")
GRAPH_BREAK = re.compile(r"Graph break", re.IGNORECASE)


def normalize(text: str) -> str:
    text = re.sub(r"/\S*/([^/\s]+)", r"\1", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "H", text)
    text = re.sub(r":\d+", ":L", text)
    return re.sub(r"\d{4,}", "N", text)


def parse_rank_log(path: Path) -> dict:
    tracing, recompiles, reasons, breaks = Counter(), Counter(), Counter(), 0
    in_guards = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("] ", 1)[-1] if "] " in raw else raw
        if match := TRACING.search(line):
            tracing[(match.group(1), Path(match.group(2)).name)] += 1
        if match := RECOMPILE.search(line):
            recompiles[(match.group(1), Path(match.group(2)).name)] += 1
            in_guards = False
        if "triggered by the following guard failure" in line:
            in_guards = True
            continue
        if in_guards:
            if match := GUARD.match(line.split("[__recompiles]")[-1]):
                reasons[normalize(match.group(1))] += 1
                continue
            in_guards = False
        if GRAPH_BREAK.search(line):
            breaks += 1
    return {"tracing": tracing, "recompiles": recompiles, "reasons": reasons, "breaks": breaks}


def load_compile(run_dir: Path) -> dict[str, dict]:
    logs = {}
    for path in sorted(run_dir.glob("logs/**/torchrun/**/stderr.log")):
        logs[path.parent.name] = parse_rank_log(path)
    return logs


def span(values: list[int]) -> str:
    return f"{min(values)}" if min(values) == max(values) else f"{min(values)}-{max(values)}"


def report_compile(root: Path, cells: list[str]) -> None:
    print("## torch.compile (TORCH_LOGS=recompiles,graph_breaks,dynamo), per trainer rank\n")
    for cell in cells:
        reference = load_compile(root / REFERENCE / cell)
        if not reference:
            continue
        for arm in ARMS:
            logs = reference if arm == REFERENCE else load_compile(root / arm / cell)
            if not logs:
                continue
            ranks = sorted(logs)
            line = (f"{cell:<18} {arm:<6} ranks={len(ranks)} traced frames={span([sum(logs[r]['tracing'].values()) for r in ranks])}"
                    f" recompiles={span([sum(logs[r]['recompiles'].values()) for r in ranks])}"
                    f" graph-break lines={span([logs[r]['breaks'] for r in ranks])}")
            if arm != REFERENCE:
                same = all(logs.get(r, {}).get(key) == reference[r][key] for r in reference for key in ("tracing", "recompiles", "reasons"))
                line += f" | same compile events as main1 on every rank: {same}"
            print(line)
            if arm == REFERENCE or not same:
                rank0 = logs[ranks[0]]
                models = {key: count for key, count in rank0["tracing"].items() if not key[1].startswith(("torch", "_"))}
                print(f"    rank {ranks[0]} traced frames: {dict(sorted(models.items()))}")
                print(f"    rank {ranks[0]} recompiles: {dict(rank0['recompiles'])}")
                for reason, count in rank0["reasons"].most_common(6):
                    print(f"      {count}x {reason[:160]}")
            if arm != REFERENCE and not same:
                for r in ranks[:1]:
                    print(f"    rank {r} traced diff: +{dict(logs[r]['tracing'] - reference[r]['tracing'])} "
                          f"-{dict(reference[r]['tracing'] - logs[r]['tracing'])}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="E2E_OUT, holding one directory per arm")
    parser.add_argument("--sections", default="metrics,weights,exports,compile")
    args = parser.parse_args()
    cells = sorted({path.name for arm in ARMS if (args.root / arm).is_dir() for path in (args.root / arm).iterdir()
                    if path.is_dir() and (path / "monitors").is_dir()})
    sections = args.sections.split(",")
    for arm in ARMS:
        status = args.root / arm / "status.tsv"
        if status.exists():
            print(f"{arm} status:\n{status.read_text()}")
    if "metrics" in sections:
        report_metrics(args.root, cells)
    if "weights" in sections:
        report_weights(args.root)
    if "exports" in sections:
        report_exports(args.root)
    if "compile" in sections:
        report_compile(args.root, cells)


if __name__ == "__main__":
    main()
