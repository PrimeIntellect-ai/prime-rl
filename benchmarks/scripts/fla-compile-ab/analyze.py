"""Summarize one A/B run dir: Dynamo compiles per frame and recompile reasons (TORCH_TRACE), compile seconds per rank,
time to first step, and per-step loss, grad norm and step time.

Usage: python analyze.py RUN_DIR [--json OUT]
Reads RUN_DIR/torch_trace/*rank_<R>_*.log, RUN_DIR/start_epoch.txt, the trainer log and the file monitor's metrics.jsonl.
"""

import argparse
import datetime
import glob
import json
import os
import re
from collections import Counter, defaultdict

RANK_RE = re.compile(r"rank_(\d+)_")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LOOP_RE = re.compile(r"(\d\d:\d\d:\d\d)\s+INFO Starting training loop")
STEP_RE = re.compile(r"(\d\d:\d\d:\d\d)\s+\w+\s+(?:\x1b\[[0-9;]*m)?Step (\d+) \|")


def read_compile_records(path):
    records = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith("\t"):
                continue
            brace = line.find("] {")
            if brace < 0 or "compilation_metrics" not in line:
                continue
            try:
                meta = json.loads(line[brace + 2 :])
            except json.JSONDecodeError:
                continue
            record = meta.get("compilation_metrics")
            if record is not None:
                records.append(record)
    return records


def us(value):
    return (value or 0) / 1e6


def short_reason(reason):
    reason = (reason or "").strip()
    reason = re.sub(r"0x[0-9a-f]+", "0x?", reason)
    return " ".join(reason.split())[:1500]


OPTIMIZER_FILES = ("muon.py", "scalar_opts.py")


def is_layer_frame(label):
    return not any(name in label for name in OPTIMIZER_FILES)


def frame_label(record):
    return f"{record.get('co_name')} @ {os.path.basename(record.get('co_filename') or '?')}:{record.get('co_firstlineno')}"


def compile_stats(trace_dir):
    by_rank = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.log"))):
        match = RANK_RE.search(os.path.basename(path))
        if match:
            by_rank.setdefault(int(match.group(1)), []).extend(read_compile_records(path))
    per_rank = {}
    for rank, records in sorted(by_rank.items()):
        forward = [r for r in records if r.get("is_forward") is not False]
        frames = defaultdict(list)
        for r in forward:
            frames[str(r.get("compile_id")).split("/")[0]].append(r)
        reasons = Counter(short_reason(r.get("recompile_reason")) for r in forward if r.get("recompile_reason"))
        per_rank[rank] = {
            "records": len(records),
            "forward_compiles": len(forward),
            "backward_compiles": len(records) - len(forward),
            "recompiles": sum(reasons.values()),
            "compile_wall_s": sum(us(r.get("duration_us")) for r in records),
            "dynamo_cumulative_s": sum(us(r.get("dynamo_cumulative_compile_time_us")) for r in records),
            "aot_autograd_s": sum(us(r.get("aot_autograd_cumulative_compile_time_us")) for r in records),
            "inductor_s": sum(us(r.get("inductor_cumulative_compile_time_us")) for r in records),
            "frames": {
                frame_id: {
                    "label": frame_label(rs[0]),
                    "compiles": len(rs),
                    "compile_wall_s": sum(us(r.get("duration_us")) for r in rs),
                    "graph_ops": [r.get("graph_op_count") for r in rs],
                    "entries": [
                        {
                            "compile_id": r.get("compile_id"),
                            "wall_s": us(r.get("duration_us")),
                            "inductor_s": us(r.get("inductor_cumulative_compile_time_us")),
                            "graph_ops": r.get("graph_op_count"),
                            "reason": short_reason(r.get("recompile_reason")),
                        }
                        for r in rs
                    ],
                }
                for frame_id, rs in sorted(frames.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
            },
            "recompile_reasons": dict(reasons.most_common()),
            "graph_breaks": dict(
                Counter(
                    short_reason(reason).split(" Hint:")[0][:300]
                    for r in forward
                    for reason in (r.get("restart_reasons") or [])
                ).most_common()
            ),
        }
    return per_rank


def metrics(run_dir):
    paths = glob.glob(os.path.join(run_dir, "out", "**", "metrics.jsonl"), recursive=True) or [
        path for path in glob.glob(os.path.join(run_dir, "**", "metrics.jsonl"), recursive=True) if "torch_trace" not in path
    ]
    rows = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = row.get("step")
                if step is None:
                    continue
                merged = rows.setdefault(int(step), {})
                if "time" in row:
                    merged["first_record_time"] = min(merged.get("first_record_time", row["time"]), row["time"])
                merged.update(row)
    return rows, paths


def pick(row, *candidates):
    for key in candidates:
        if key in row:
            return row[key]
    return None


def step_end_times(log_path, start_epoch):
    start = datetime.datetime.fromtimestamp(start_epoch, datetime.timezone.utc)
    ends = {}
    with open(log_path, errors="replace") as handle:
        for line in handle:
            loop = LOOP_RE.search(ANSI_RE.sub("", line))
            if loop and ("[default0]" in line or "[trainer0]" in line) and 0 not in ends:
                hh, mm, ss = map(int, loop.group(1).split(":"))
                stamp = start.replace(hour=hh, minute=mm, second=ss, microsecond=0)
                if stamp < start - datetime.timedelta(seconds=1):
                    stamp += datetime.timedelta(days=1)
                ends[0] = stamp.timestamp()
            match = STEP_RE.search(ANSI_RE.sub("", line))
            if not match or not ("[default0]" in line or "[trainer0]" in line):
                continue
            hh, mm, ss = map(int, match.group(1).split(":"))
            stamp = start.replace(hour=hh, minute=mm, second=ss, microsecond=0)
            if stamp < start - datetime.timedelta(seconds=1):
                stamp += datetime.timedelta(days=1)
            ends.setdefault(int(match.group(2)), stamp.timestamp())
    return ends


def find_trainer_log(run_dir):
    for candidate in ("logs/trainer.log", "logs/attempt_1/trainer.log"):
        path = os.path.join(run_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def analyze(run_dir):
    start_epoch = float(open(os.path.join(run_dir, "start_epoch.txt")).read())
    end_path = os.path.join(run_dir, "end_epoch.txt")
    out = {
        "run_dir": run_dir,
        "rc": open(os.path.join(run_dir, "rc.txt")).read().strip() if os.path.exists(os.path.join(run_dir, "rc.txt")) else None,
        "wall_s": float(open(end_path).read()) - start_epoch if os.path.exists(end_path) else None,
    }
    for name in ("sha.txt", "worktree_head.txt", "host.txt", "arm.txt"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            out[name.removesuffix(".txt")] = open(path).read().strip()
    out["compile"] = compile_stats(os.path.join(run_dir, "torch_trace"))
    rows, paths = metrics(run_dir)
    out["metrics_files"] = paths
    out["steps"] = {
        step: {
            "loss": pick(row, "loss/mean", "loss", "train/loss"),
            "grad_norm": pick(row, "optim/grad_norm", "grad_norm", "train/grad_norm"),
            "time_step": pick(row, "time/step"),
            "end_s_from_launch": row["first_record_time"] - start_epoch if "first_record_time" in row else None,
        }
        for step, row in sorted(rows.items())
    }
    out["metric_keys"] = sorted({k for row in rows.values() for k in row})
    log_path = find_trainer_log(run_dir)
    if log_path:
        ends = step_end_times(log_path, start_epoch)
        out["step_log_end_s_from_launch"] = {step: t - start_epoch for step, t in sorted(ends.items())}
    return out


def render(result):
    lines = [f"# {result['run_dir']}", f"rc={result['rc']} wall_s={result['wall_s']} sha={result.get('sha')} host={result.get('host')}"]
    compile_ = result["compile"]
    if compile_:
        rank0 = compile_.get(0, {})
        lines.append(
            f"rank0: fwd compiles={rank0.get('forward_compiles')} bwd={rank0.get('backward_compiles')} recompiles={rank0.get('recompiles')} "
            f"compile wall s={rank0.get('compile_wall_s', 0):.1f} (dynamo cum {rank0.get('dynamo_cumulative_s', 0):.1f}, "
            f"aot {rank0.get('aot_autograd_s', 0):.1f}, inductor {rank0.get('inductor_s', 0):.1f})"
        )
        worst = max(compile_.items(), key=lambda kv: kv[1]["compile_wall_s"])
        lines.append(f"max over ranks: compile wall s={worst[1]['compile_wall_s']:.1f} (rank {worst[0]})")
        lines.append("| frame | label | compiles | wall s |")
        lines.append("|---|---|---|---|")
        for frame_id, frame in rank0.get("frames", {}).items():
            lines.append(f"| {frame_id} | {frame['label']} | {frame['compiles']} | {frame['compile_wall_s']:.1f} |")
        lines.append("layer-frame entries (rank 0), with the recompile reason that triggered each:")
        for frame in rank0.get("frames", {}).values():
            if not is_layer_frame(frame["label"]):
                continue
            for entry in frame["entries"]:
                lines.append(
                    f"- {frame['label']} {entry['compile_id']} ({entry['wall_s']:.1f} s, inductor {entry['inductor_s']:.1f} s, "
                    f"{entry['graph_ops']} ops): {(entry['reason'] or 'first compile')[:600]}"
                )
        lines.append("graph breaks (rank 0, restart reasons):")
        for reason, count in rank0.get("graph_breaks", {}).items():
            lines.append(f"- {count}x {reason}")
        lines.append("recompile reasons (rank 0):")
        for reason, count in rank0.get("recompile_reasons", {}).items():
            lines.append(f"- {count}x {reason}")
    lines.append(f"step log end s from launch: {result.get('step_log_end_s_from_launch')}")
    for step, row in result["steps"].items():
        lines.append(
            f"step {step}: loss={row['loss']} grad_norm={row['grad_norm']} time/step={row['time_step']} "
            f"end s from launch={row['end_s_from_launch']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    result = analyze(os.path.abspath(args.run_dir))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
    print(render(result))


if __name__ == "__main__":
    main()
