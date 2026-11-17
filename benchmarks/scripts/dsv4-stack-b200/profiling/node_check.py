"""GPU node health check: can this node (or node pair) be trusted for timings and multi-node jobs?

Run under torchrun, one process per GPU. Checks run on all GPUs of a node at once (except host copies), so power,
thermals and shared links are loaded as in training. Bandwidths are GB/s, 1 GB = 1e9 bytes. Checks (`--checks`, comma
separated, default all):
- `gpu`: per GPU, BF16 GEMM TFLOP/s (M=N=K=8192; `bf16_short` is the median of 50 calls after warmup,
  `bf16_sustained` the median of 50-call blocks over the second half of a `--sustain-s` run, default 15 s, kept
  continuously busy, which is what a power cap lets a long GEMM run at), FP8 GEMM TFLOP/s (`torch._scaled_mm`,
  e4m3, tensorwise scales, short only), HBM bandwidth on 2 GiB buffers (`hbm_copy` counts read plus write, `hbm_fill`
  write only, `hbm_read` a float32 sum, read only; median of 30), and pinned host copies (`h2d`, `d2h`, 1 GiB, one
  GPU at a time, since eight at once share host memory bandwidth and halve each GPU's figure). During
  the sustained run, nvidia-smi samples each GPU's SM clock, power, temperature and throttle reasons
  (`sm_mhz`, `power_w`, `temp_c`, `throttle`: OR of the active reason bits, 0x4 is the SW power cap).
- `p2p`: NCCL send/recv shift tests, each rank sending to rank+shift while receiving from rank-shift (256 MiB,
  GB/s per direction). On one node, shifts 1..7 cover every GPU pair; per GPU, `p2p_median` over shifts is low only
  for a GPU with a bad link (a healthy GPU meets the bad one in just two shifts), and `p2p_min` names the worst
  shift. On two nodes, a shift of a node's GPU count sends every GPU to the same local rank on the other node, which
  tests each GPU's own NIC rail (`rail_p2p`). Skipped on more nodes: NCCL gives each peer fewer p2p channels as the
  rank count grows, so the same shift over 15 nodes read 11-13 GB/s on links that carry 45 GB/s in pairs.
- `comm`: all_reduce (float32 sum), all_gather, reduce_scatter and all_to_all over all ranks at 64 MiB, 256 MiB and
  1 GiB, timed with CUDA events around back-to-back calls after warmup, taking the slowest rank's mean. Sizes and
  bus bandwidth follow nccl-tests: the size is the full buffer (all_gather: output, reduce_scatter: input, all_to_all:
  one rank's send buffer, rounded so each rank's share is a multiple of 64 elements: unaligned shares at GPU counts
  that are not a power of two read all_gather and reduce_scatter about 2x low), `algbw = size / time`, and
  `busbw = algbw * 2(n-1)/n` for all_reduce and `algbw * (n-1)/n` for the others, where n is the rank count. busbw
  is comparable to link bandwidth across n. On more than one node it also times a 1 GiB all_gather in
  one-GPU-per-node groups, one group per local GPU index (e.g. expert-weight FSDP under expert parallelism): all groups at
  once (`rail_all_gather`) and local GPU 0's group alone (`rail_all_gather_solo`). Concurrent well below solo means
  groups share NICs (`tools/nccl.md`). Each GPU record also holds the NCCL version actually loaded (`nccl`).
- `health`: nvidia-smi power limit, max SM clock, uncorrectable ECC (volatile, aggregate), remapped rows
  (uncorrectable, pending, failure), PCIe width (current vs max), active NVLinks per GPU, kernel XID lines when
  `dmesg` is readable without root (`xid: null` otherwise), and InfiniBand ports from /sys/class/infiniband (state,
  rate). Run after the load checks, so ECC errors raised during them count.

Rank 0 writes one JSON line per GPU (`kind: gpu`), per node (`node`), per shift rank (`p2p`) and per collective size
(`comm`, with its host list) to `--out`, then prints a PASS/WARN/FAIL verdict per node against EXPECT (INFO lines, such as a lifetime-only ECC count, do not change it), the table of
normal bands and fail floors keyed by GPU name (each with its source and date). `--summarize FILE...` reads the files
of several runs and adds fleet checks: a node more than 5% below the fleet median on any metric (the slowest GPU's
value for per-GPU metrics, plus its active IB port count and slowest IB rate) is WARN, and each 2-node run is a pair
result, flagged when below EXPECT or 5% below the pair median. A node whose every pair is low FAILs; a node with only
some low pairs is WARN, since the partner may be the cause. A later file's GPU and node results (e.g. a rerun) replace
an earlier file's for the same node.

Recipe on SLURM, one torchrun per node. NCCL reads its usual environment: set NCCL_IB_HCA to the InfiniBand HCAs
when a node also has an Ethernet RDMA port, and check NCCL_DEBUG=INFO's `NET/IB : Using` line once. Use c10d
rendezvous for multi-node runs: SLURM_NODEID need not follow the -w order, so a static --node-rank can deadlock.
  single node, all nodes in parallel (about a minute):
    srun -N <n> --ntasks-per-node=1 bash -c 'torchrun --standalone --nproc-per-node 8 node_check.py \
      --out DIR/$(hostname).jsonl'
  node pair (repeat over a pairing with a known-good reference node and a second disjoint pairing; pairs that share
  no node may run concurrently):
    srun -N 2 -w A,B --ntasks-per-node=1 torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d \
      --rdzv-endpoint A:29500 --rdzv-id pair-A-B node_check.py --checks p2p,comm --out DIR/pair-A-B.jsonl
  all nodes: the same with -N <n>, --nnodes <n> and --checks comm
  fleet table (no torch needed): python node_check.py --summarize DIR/*.jsonl

Usage: torchrun ... node_check.py --out FILE [--checks gpu,p2p,comm,health] [--sustain-s 15]
       python node_check.py --summarize FILE [FILE ...]
"""

import argparse
import ctypes
import glob
import json
import os
import socket
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime

EXPECT = {
    "NVIDIA H100 80GB HBM3": {
        "source": (
            "node_check.py sweep of 15 H100 SXM nodes (700 W cap, 8x 400 Gb/s NDR IB), SLURM job 56108, 2026-11-11: "
            "normal is about 3% and floor about 10% below the fleet median (wider for the noisy short GEMMs)"
        ),
        "nvlinks": 18,
        "power_limit_w": 700,
        "gpu": {
            "bf16_short": (650, 550),
            "bf16_sustained": (640, 600),
            "fp8_short": (1250, 1100),
            "hbm_copy": (2950, 2850),
            "hbm_fill": (3200, 3050),
            "hbm_read": (3000, 2900),
            "h2d": (50, 40),
            "d2h": (50, 40),
            "p2p_median": (290, 260),
        },
        "comm": {
            1: {"all_reduce": (455, 420), "all_gather": (335, 310), "reduce_scatter": (335, 310), "all_to_all": (315, 290)},
            2: {"all_reduce": (420, 380), "all_gather": (300, 270), "reduce_scatter": (345, 310), "all_to_all": (82, 72)},
        },
        "rail_p2p": (42, 35),
    },
}
SIZES = (64 << 20, 256 << 20, 1 << 30)
FLEET_TOLERANCE = 0.05
RAIL_SHARE_RATIO = 0.8


def timed_ms(fn, iters, warmup):
    import torch

    for _ in range(warmup):
        fn()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events)


def sustained_tflops(fn, flops, seconds, block=50):
    import torch

    blocks = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(block):
            fn()
        end.record()
        blocks.append((time.time() - t0, start, end))
        if len(blocks) > 1:
            blocks[-2][2].synchronize()
    torch.cuda.synchronize()
    late = [s.elapsed_time(e) for t, s, e in blocks if t >= seconds / 2] or [s.elapsed_time(e) for _, s, e in blocks]
    return flops * block / (statistics.median(late) * 1e-3) / 1e12


class SmiSampler:
    """Samples nvidia-smi clocks, power, temperature and throttle reasons in the background."""

    FIELDS = "timestamp,uuid,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"

    def __init__(self):
        cmd = ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-lms", "250"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        self.lines = []
        self.thread = threading.Thread(target=lambda: self.lines.extend(self.proc.stdout), daemon=True)
        self.thread.start()

    def stop(self, since):
        self.proc.terminate()
        self.thread.join(timeout=5)
        per_gpu = defaultdict(list)
        for line in self.lines:
            fields = [f.strip() for f in line.split(",")]
            try:
                stamp = datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                values = [float(f) for f in fields[2:5]] + [int(fields[5], 16)]
            except (ValueError, IndexError):
                continue
            if stamp >= since:
                per_gpu[fields[1].removeprefix("GPU-")].append(values)
        out = {}
        for uuid, rows in per_gpu.items():
            throttle = 0
            for row in rows:
                throttle |= row[3]
            out[uuid] = {
                "sm_mhz": statistics.median(r[0] for r in rows),
                "power_w": statistics.median(r[1] for r in rows),
                "temp_c": max(r[2] for r in rows),
                "throttle": hex(throttle),
            }
        return out


def gpu_checks(local_rank, sustain_s):
    import torch
    import torch.distributed as dist

    n = 8192
    flops = 2 * n**3
    a, b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16), torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    out = {"bf16_short": flops / (timed_ms(lambda: a @ b, 50, 20) * 1e-3) / 1e12}
    sampler = SmiSampler() if local_rank == 0 else None
    dist.barrier()
    t0 = time.time()
    out["bf16_sustained"] = sustained_tflops(lambda: a @ b, flops, sustain_s)
    dist.barrier()
    clocks = sampler.stop(t0 + sustain_s / 2) if sampler else None
    a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    fp8 = lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    out["fp8_short"] = flops / (timed_ms(fp8, 50, 20) * 1e-3) / 1e12
    nbytes = 2 << 30
    src, dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda"), torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    src.random_(0, 255)
    out["hbm_copy"] = 2 * nbytes / (timed_ms(lambda: dst.copy_(src), 30, 5) * 1e-3) / 1e9
    out["hbm_fill"] = nbytes / (timed_ms(lambda: dst.fill_(1), 30, 5) * 1e-3) / 1e9
    floats = src.view(torch.float32).uniform_()
    out["hbm_read"] = nbytes / (timed_ms(lambda: floats.sum(), 30, 5) * 1e-3) / 1e9
    host = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    for turn in range(int(os.environ["LOCAL_WORLD_SIZE"])):
        dist.barrier()
        if turn == local_rank:
            out["h2d"] = (1 << 30) / (timed_ms(lambda: dev.copy_(host, non_blocking=True), 10, 3) * 1e-3) / 1e9
            out["d2h"] = (1 << 30) / (timed_ms(lambda: host.copy_(dev, non_blocking=True), 10, 3) * 1e-3) / 1e9
    return out, clocks


def agreed_iters(fn, target_s=0.3):
    import torch
    import torch.distributed as dist

    fn()
    estimate = torch.tensor(timed_ms(fn, 3, 0), device="cuda")
    dist.all_reduce(estimate, op=dist.ReduceOp.MAX)
    return max(5, min(100, int(target_s * 1e3 / estimate.item())))


def mean_ms_slowest_rank(fn, iters):
    import torch
    import torch.distributed as dist

    dist.barrier()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    per_rank = torch.tensor(start.elapsed_time(end) / iters, device="cuda")
    mine = per_rank.item()
    dist.all_reduce(per_rank, op=dist.ReduceOp.MAX)
    return per_rank.item(), mine


def shift_gbs(shift, nbytes=256 << 20):
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    send, recv = torch.ones(nbytes, dtype=torch.uint8, device="cuda"), torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def exchange():
        ops = [dist.P2POp(dist.isend, send, (rank + shift) % world), dist.P2POp(dist.irecv, recv, (rank - shift) % world)]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    _, mine = mean_ms_slowest_rank(exchange, agreed_iters(exchange))
    return nbytes / (mine * 1e-3) / 1e9


def loaded_nccl_version():
    version = ctypes.c_int()
    ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(version))
    major, rest = divmod(version.value, 10000)
    return f"{major}.{rest // 100}.{rest % 100}"


def rail_collectives(local_rank, local_world, nnodes):
    import torch
    import torch.distributed as dist

    groups = [dist.new_group([lr + local_world * n for n in range(nnodes)]) for lr in range(local_world)]
    elems = SIZES[-1] // 4 // (nnodes * 64) * (nnodes * 64)
    full, part = torch.empty(elems, device="cuda"), torch.ones(elems // nnodes, device="cuda")
    rows = []
    for op, active in (("rail_all_gather", True), ("rail_all_gather_solo", local_rank == 0)):
        def gather():
            if active:
                dist.all_gather_into_tensor(full, part, group=groups[local_rank])

        ms, _ = mean_ms_slowest_rank(gather, agreed_iters(gather))
        algbw = elems * 4 / (ms * 1e-3) / 1e9
        rows.append({"op": op, "bytes": elems * 4, "ms": ms, "algbw": algbw, "busbw": algbw * (nnodes - 1) / nnodes})
    return rows


def collectives(world):
    import torch
    import torch.distributed as dist

    all_gather = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
    reduce_scatter = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    rows = []
    for size in SIZES:
        elems = size // 4 // (world * 64) * (world * 64)
        size = elems * 4
        full, part = torch.ones(elems, device="cuda"), torch.ones(elems // world, device="cuda")
        a2a_out = torch.empty(elems, device="cuda")
        ops = {
            "all_reduce": (lambda: dist.all_reduce(full), 2 * (world - 1) / world),
            "all_gather": (lambda: all_gather(full, part), (world - 1) / world),
            "reduce_scatter": (lambda: reduce_scatter(part, full), (world - 1) / world),
            "all_to_all": (lambda: dist.all_to_all_single(a2a_out, full), (world - 1) / world),
        }
        for name, (fn, factor) in ops.items():
            ms, _ = mean_ms_slowest_rank(fn, agreed_iters(fn))
            algbw = size / (ms * 1e-3) / 1e9
            rows.append({"op": name, "bytes": size, "ms": ms, "algbw": algbw, "busbw": algbw * factor})
    return rows


def run_text(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def number(text):
    try:
        return float(text)
    except ValueError:
        return text


def health():
    fields = (
        "uuid,name,power.limit,clocks.max.sm,ecc.errors.uncorrected.volatile.total,"
        "ecc.errors.uncorrected.aggregate.total,remapped_rows.uncorrectable,remapped_rows.pending,"
        "remapped_rows.failure,pcie.link.width.current,pcie.link.width.max"
    ).split(",")
    keys = ["uuid", "name", "power_limit_w", "max_sm_mhz", "ecc_unc_volatile", "ecc_unc_aggregate", "remap_unc",
            "remap_pending", "remap_failure", "pcie_width", "pcie_width_max"]
    text = run_text(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]) or ""
    gpus = {}
    for line in text.strip().splitlines():
        row = dict(zip(keys, (number(f.strip()) for f in line.split(","))))
        row["uuid"] = row["uuid"].removeprefix("GPU-")
        gpus[row["uuid"]] = row
    links, uuid = defaultdict(int), None
    for line in (run_text(["nvidia-smi", "nvlink", "-s"]) or "").splitlines():
        if "UUID:" in line:
            uuid = line.split("UUID:")[1].strip(" )").removeprefix("GPU-")
        elif line.strip().startswith("Link") and "GB/s" in line:
            links[uuid] += 1
    for uuid, row in gpus.items():
        row["nvlinks"] = links[uuid]
    try:
        dmesg = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=60)
        xid = [line for line in dmesg.stdout.splitlines() if "Xid" in line] if dmesg.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        xid = None
    ports = []
    for port in sorted(glob.glob("/sys/class/infiniband/*/ports/*")):
        read = lambda name: open(os.path.join(port, name)).read().strip()
        ports.append({"dev": port.split("/")[4], "port": port.split("/")[-1], "state": read("state"),
                      "rate": read("rate"), "link_layer": read("link_layer")})
    return gpus, {"xid": xid, "ib_ports": ports}


def run(args):
    import torch
    import torch.distributed as dist

    local_rank, local_world = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    nnodes = world // local_world
    host = socket.gethostname()
    props = torch.cuda.get_device_properties(device)
    checks = set(args.checks.split(","))
    mine = {"kind": "gpu", "host": host, "gpu": local_rank, "uuid": str(props.uuid), "name": props.name}
    clocks = None
    if "gpu" in checks:
        metrics, clocks = gpu_checks(local_rank, args.sustain_s)
        mine.update(metrics)
    p2p = []
    if "p2p" in checks:
        shifts = range(1, world) if nnodes == 1 else [local_world] if nnodes == 2 else []
        for shift in shifts:
            p2p.append({"kind": "p2p", "host": host, "gpu": local_rank, "rank": rank, "shift": shift,
                        "gbs": shift_gbs(shift)})
        if nnodes == 1:
            mine["p2p_median"] = statistics.median(r["gbs"] for r in p2p)
            mine["p2p_min"] = min(r["gbs"] for r in p2p)
    comm = collectives(world) if "comm" in checks else []
    if "comm" in checks and nnodes > 1:
        comm += rail_collectives(local_rank, local_world, nnodes)
    mine["nccl"] = loaded_nccl_version()
    gpus, node = health() if "health" in checks and local_rank == 0 else ({}, None)
    gathered = [None] * world
    dist.all_gather_object(gathered, (mine, p2p, clocks, gpus, node))
    if rank == 0:
        hosts = sorted({m["host"] for m, *_ in gathered})
        per_uuid = defaultdict(dict)
        for _, _, g_clocks, g_gpus, _ in gathered:
            for uuid, info in list((g_clocks or {}).items()) + list(g_gpus.items()):
                per_uuid[uuid].update({k: v for k, v in info.items() if k not in ("uuid", "name")})
        records = [{**m, **per_uuid[m["uuid"]]} for m, *_ in gathered]
        records += [{"kind": "node", "host": m["host"], **n} for m, _, _, _, n in gathered if n is not None]
        records += [{"hosts": hosts, "nnodes": nnodes, **r} for _, rows, _, _, _ in gathered for r in rows]
        records += [{"kind": "comm", "hosts": hosts, "nnodes": nnodes, "world": world, **row} for row in comm]
        with open(args.out, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        summarize(records)
    dist.barrier()
    dist.destroy_process_group()


def fmt(value):
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def judge(value, band, label, problems):
    normal, floor = band
    if value < floor:
        problems.append(("FAIL", f"{label} {fmt(value)} < floor {floor}"))
    elif value < normal:
        problems.append(("WARN", f"{label} {fmt(value)} < normal {normal}"))


def largest(comm_rows, nnodes):
    by_op = {}
    for row in comm_rows:
        if row["nnodes"] == nnodes and row["bytes"] >= by_op.get(row["op"], {}).get("bytes", 0):
            by_op[row["op"]] = row
    return by_op


def node_problems(host, gpus, node, comm_rows, expect):
    problems = []
    for g in gpus:
        tag = f"gpu{g['gpu']}"
        for metric, band in expect.get("gpu", {}).items():
            if metric in g:
                judge(g[metric], band, f"{tag} {metric}", problems)
        if g.get("nvlinks", expect.get("nvlinks", 0)) < expect.get("nvlinks", 0):
            problems.append(("FAIL", f"{tag} {g['nvlinks']} NVLinks active, expect {expect['nvlinks']}"))
        if isinstance(g.get("power_limit_w"), float) and g["power_limit_w"] < expect.get("power_limit_w", 0):
            problems.append(("WARN", f"{tag} power limit {g['power_limit_w']:.0f} W"))
        if g.get("ecc_unc_volatile", 0) not in (0, "[N/A]"):
            problems.append(("FAIL", f"{tag} {g['ecc_unc_volatile']} volatile uncorrectable ECC errors"))
        if g.get("ecc_unc_aggregate", 0) not in (0, "[N/A]"):
            live = g.get("ecc_unc_volatile", 0) not in (0, "[N/A]") or "Yes" in (g.get("remap_pending"), g.get("remap_failure"))
            problems.append(("WARN" if live else "INFO", f"{tag} {g['ecc_unc_aggregate']} aggregate uncorrectable ECC errors"
                             + ("" if live else " (lifetime count only: volatile 0, no pending or failed row remaps)")))
        if g.get("remap_failure") == "Yes":
            problems.append(("FAIL", f"{tag} row remapping failed"))
        if g.get("remap_pending") == "Yes":
            problems.append(("WARN", f"{tag} row remap pending (needs GPU reset)"))
        if isinstance(g.get("pcie_width"), float) and g["pcie_width"] < g["pcie_width_max"]:
            problems.append(("FAIL", f"{tag} PCIe x{g['pcie_width']:.0f} of x{g['pcie_width_max']:.0f}"))
    if node:
        if node["xid"]:
            problems.append(("WARN", f"{len(node['xid'])} XID lines in dmesg, last: {node['xid'][-1]}"))
        ib = [p for p in node["ib_ports"] if p["link_layer"] == "InfiniBand"]
        top = max((float(p["rate"].split()[0]) for p in ib), default=0)
        for p in ib:
            if p["state"] != "4: ACTIVE" or float(p["rate"].split()[0]) < top:
                problems.append(("FAIL", f"IB {p['dev']}/{p['port']} {p['state']} {p['rate']}"))
    for op, row in largest([r for r in comm_rows if r["hosts"] == [host]], 1).items():
        if op in expect.get("comm", {}).get(1, {}):
            judge(row["busbw"], expect["comm"][1][op], f"intra {op} busbw", problems)
    return problems


def summarize(records):
    by_kind = defaultdict(list)
    latest = {}
    for r in records:
        if r["kind"] in ("gpu", "node"):
            key = (r["kind"], r["host"], r.get("gpu"))
            latest[key] = {**latest.get(key, {}), **r}
        else:
            by_kind[r["kind"]].append(r)
    for r in latest.values():
        by_kind[r["kind"]].append(r)
    gpus_by_host = defaultdict(list)
    for g in by_kind["gpu"]:
        gpus_by_host[g["host"]].append(g)
    nodes = {n["host"]: n for n in by_kind["node"]}
    hosts = sorted(set(gpus_by_host) | set(nodes))
    name = next((g["name"] for g in by_kind["gpu"]), "")
    expect = EXPECT.get(name, {})
    print(f"GPU model: {name}; expectations: {expect.get('source', 'none for this model, fleet checks only')}")
    versions = sorted({g["nccl"] for g in by_kind["gpu"] if "nccl" in g})
    if versions:
        print(f"NCCL loaded: {', '.join(versions)}")
    per_host = {h: node_problems(h, gpus_by_host[h], nodes.get(h), by_kind["comm"], expect) for h in hosts}
    metrics = {}
    for h in hosts:
        row = {}
        for metric in list(expect.get("gpu", {})) or ["bf16_short", "bf16_sustained", "fp8_short", "hbm_copy"]:
            values = [g[metric] for g in gpus_by_host[h] if metric in g]
            if values:
                row[metric] = min(values)
        for op, r in largest([r for r in by_kind["comm"] if r["hosts"] == [h]], 1).items():
            row[f"intra_{op}"] = r["busbw"]
        ib = [p for p in nodes[h]["ib_ports"] if p["link_layer"] == "InfiniBand"] if h in nodes else []
        if ib:
            row["ib_active"] = sum(p["state"] == "4: ACTIVE" for p in ib)
            row["ib_rate_min"] = min(float(p["rate"].split()[0]) for p in ib)
        metrics[h] = row
    if len(hosts) > 2:
        for metric in {m for row in metrics.values() for m in row}:
            values = {h: row[metric] for h, row in metrics.items() if metric in row}
            median = statistics.median(values.values())
            for h, v in values.items():
                if v < (1 - FLEET_TOLERANCE) * median:
                    per_host[h].append(("WARN", f"{metric} {fmt(v)} is {100 * (1 - v / median):.0f}% below fleet median {fmt(median)}"))
    pairs = pair_results(by_kind, expect)
    low_pairs = defaultdict(list)
    for p in pairs:
        for h in p["hosts"]:
            per_host.setdefault(h, [])
            low_pairs[h].append(p["problems"])
    for h, results in low_pairs.items():
        bad = [p for p in results if p]
        if bad:
            level = "FAIL" if len(bad) == len(results) and len(results) > 1 else "WARN"
            per_host[h].append((level, f"{len(bad)}/{len(results)} pairs low: " + "; ".join(p[0] for p in bad)))
    columns = ["bf16_short", "bf16_sustained", "fp8_short", "hbm_copy", "h2d", "p2p_median", "intra_all_reduce",
               "intra_all_to_all"]
    print("| node | " + " | ".join(columns) + " | IB active | pairs low | verdict |")
    print("|---" * (len(columns) + 4) + "|")
    for h in sorted(per_host):
        cells = [f"{metrics.get(h, {})[c]:.0f}" if c in metrics.get(h, {}) else "-" for c in columns]
        low = sum(1 for p in low_pairs.get(h, []) if p)
        levels = {l for l, _ in per_host[h]}
        verdict = "FAIL" if "FAIL" in levels else "WARN" if "WARN" in levels else "PASS"
        ib = [p for p in nodes[h]["ib_ports"] if p["link_layer"] == "InfiniBand"] if h in nodes else []
        active = f"{sum(p['state'] == '4: ACTIVE' for p in ib)}/{len(ib)}" if h in nodes else "-"
        print(f"| {h} | " + " | ".join(cells) + f" | {active} | {low}/{len(low_pairs.get(h, []))} | {verdict} |")
    print("(GPU columns are the node's slowest GPU; TFLOP/s for GEMMs, GB/s otherwise; comm columns are busbw at the largest size)")
    for h in sorted(per_host):
        for level, text in per_host[h]:
            print(f"{level} {h}: {text}")
    shared = [p for p in pairs if p["metrics"].get("rail_all_gather_solo")
              and p["metrics"].get("rail_all_gather", 0) < RAIL_SHARE_RATIO * p["metrics"]["rail_all_gather_solo"]]
    if shared:
        print(f"WARN shared NICs on {len(shared)}/{len(pairs)} pairs: concurrent one-GPU-per-node all_gather below "
              f"{RAIL_SHARE_RATIO}x the solo rate; an NCCL version or topology effect, not a node fault (tools/nccl.md)")
    for p in pairs:
        print(f"pair {'+'.join(p['hosts'])}: " + ", ".join(f"{k} {fmt(v)}" for k, v in p["metrics"].items())
              + (f"  LOW: {'; '.join(p['problems'])}" if p["problems"] else ""))
    for nnodes in sorted({r["nnodes"] for r in by_kind["comm"] if r["nnodes"] > 2}):
        rows = [r for r in by_kind["comm"] if r["nnodes"] == nnodes]
        print(f"{nnodes}-node collectives (busbw GB/s by size MiB):")
        for r in rows:
            print(f"  {r['op']} {r['bytes'] >> 20} MiB: {r['busbw']:.0f} ({r['ms']:.2f} ms)")


def pair_results(by_kind, expect):
    pairs = defaultdict(dict)
    for r in by_kind["comm"]:
        if r["nnodes"] == 2:
            pairs[tuple(r["hosts"])].update(largest([x for x in by_kind["comm"] if x["hosts"] == r["hosts"]], 2))
    results = []
    for hosts, ops in pairs.items():
        metrics = {op: row["busbw"] for op, row in ops.items()}
        rail = [r["gbs"] for r in by_kind["p2p"] if r["hosts"] == list(hosts)]
        if rail:
            metrics["rail_p2p_min"] = min(rail)
        results.append({"hosts": list(hosts), "metrics": metrics, "problems": []})
    for p in results:
        for metric, value in p["metrics"].items():
            band = expect.get("rail_p2p") if metric == "rail_p2p_min" else expect.get("comm", {}).get(2, {}).get(metric)
            found = []
            if band:
                judge(value, band, metric, found)
            median = statistics.median(q["metrics"][metric] for q in results if metric in q["metrics"])
            if len(results) > 2 and value < (1 - FLEET_TOLERANCE) * median:
                found.append(("WARN", f"{metric} {fmt(value)} vs pair median {fmt(median)}"))
            p["problems"] += [text for _, text in found]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="JSONL results file (rank 0 writes it)")
    parser.add_argument("--checks", default="gpu,p2p,comm,health")
    parser.add_argument("--sustain-s", type=float, default=15.0)
    parser.add_argument("--summarize", nargs="+", metavar="FILE", help="print the fleet table over result files")
    args = parser.parse_args()
    if args.summarize:
        records = [json.loads(line) for path in args.summarize for line in open(path) if line.strip()]
        summarize(records)
    else:
        run(args)


if __name__ == "__main__":
    main()
