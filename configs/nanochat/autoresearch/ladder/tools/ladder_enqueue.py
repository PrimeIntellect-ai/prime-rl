"""Write ladder_queue.jsonl: every new run of the full #420 ladder (isoflop 1e18/3e18/6e18 + miniseries d10-d20)
for both arms, longest estimated job first. Runs already in the ledger with identical config+math are reused
(see REUSE) and not queued.

python ladder_enqueue.py [--dry]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ladder_plan import ARMS, ISO, MINI, counts  # noqa: E402

B = 2**19
# measured effective training FLOP/s on 8xH200 (fpt * tokens/s), loop runs; gated = s4r8+Muon
PLAIN_FLOPS_S = {8: 1.05e15, 10: 1.42e15, 12: 1.96e15, 14: 2.29e15, 16: 2.61e15, 18: 2.74e15}
GATED_RATIO = 0.63  # gated s4r8 FLOP/s / plain FLOP/s at d12 (0.65) and d16 (0.63)

# (tag, depth, arm) -> run_id reused from the loop (seed 0, identical model config, optimizer and schedule math)
REUSE = {
    ("iso1e18", 8, "plain"): "d0znccr8yygv35zayt1ffuiz",
    ("iso1e18", 10, "plain"): "cvdd3ss4n4n2v6n0glnv63dw",
    ("iso1e18", 12, "plain"): "zy80o126t0me24flosobi60a",  # code-eval twin of wdajij5avg (0.899745 vs 0.899706)
    ("iso1e18", 14, "plain"): "tmh0bgqpo00e1lpfa9wzcc2p",
    ("iso1e18", 16, "plain"): "mqj84lajctvex7uj65jzi169",
    ("iso1e18", 12, "gated"): "g4ui5f0126kaf2f2gzfb2hmi",  # code-eval twin of z19hkgkbmz (0.895246 vs 0.895294)
    ("iso3e18", 14, "plain"): "ejf66wr5rjulamc84kbnflv3",
    ("iso3e18", 16, "plain"): "m6n0nosm2hcfccbbdufungor",
    ("iso3e18", 18, "plain"): "kxl2bienkozpry2id7skuluf",
    ("iso3e18", 16, "gated"): "ith3zo19hf31jpmonls3mfuc",
    ("mini", 12, "plain"): "rxtdjmw5ftkkrrmlfveciztc",  # h420c: 2060 steps (1.080B tokens; exact 8x params = 1.082B)
}


def est_flops_s(depth, arm):
    ks = sorted(PLAIN_FLOPS_S)
    if depth in PLAIN_FLOPS_S:
        f = PLAIN_FLOPS_S[depth]
    elif depth > ks[-1]:
        f = PLAIN_FLOPS_S[18] + (depth - 18) * 0.05e15
    else:
        lo = max(k for k in ks if k < depth)
        hi = min(k for k in ks if k > depth)
        f = PLAIN_FLOPS_S[lo] + (PLAIN_FLOPS_S[hi] - PLAIN_FLOPS_S[lo]) * (depth - lo) / (hi - lo)
    return f * (GATED_RATIO if arm == "gated" else 1.0)


def jobs():
    out = []
    for C, depths in ISO.items():
        tag = f"iso{C:.0e}".replace("+", "").replace("e0", "e")
        for d in depths:
            for arm in ARMS:
                out.append(dict(tag=tag, depth=d, arm=arm, flops=C))
    for d in MINI:
        for arm in ARMS:
            c = counts(d, arm)
            out.append(dict(tag="mini", depth=d, arm=arm, tokens=8 * c["all_v1"]))
    return out


def job_args(j):
    arm = ARMS[j["arm"]]
    args = ["--depth", str(j["depth"]), "--residual", json.dumps(arm["residual"], separators=(",", ":")),
            "--variant", arm["variant"], "--tag", j["tag"], "--seed", "0", "--code-eval", "--val-points", "20",
            "--ladder", j["tag"]]
    if "flops" in j:
        args += ["--flops", f"{j['flops']:.0e}"]
    else:
        args += ["--tokens", str(int(j["tokens"]))]
    for o in arm["optim"]:
        args += ["--optim", o]
    if j.get("micro_batch"):
        args += ["--micro-batch", str(j["micro_batch"])]
    note = f"full #420 ladder: {j['tag']} d{j['depth']} {j['arm']}" + (
        " (8 x all params)" if j["tag"] == "mini" else "")
    args += ["--note", note]
    return args


def main():
    dry = "--dry" in sys.argv
    q = []
    for j in jobs():
        if (j["tag"], j["depth"], j["arm"]) in REUSE:
            continue
        c = counts(j["depth"], j["arm"])
        flops = j.get("flops") or c["flops_per_token"] * round(j["tokens"] / B) * B
        est_h = flops / est_flops_s(j["depth"], j["arm"]) / 3600
        jid = f"L-{j['tag']}-d{j['depth']}-{j['arm']}"
        q.append(dict(id=jid, args=job_args(j), priority=round(est_h * 1000), est_h=est_h, flops=flops))
    q.sort(key=lambda r: -r["priority"])
    tot = sum(r["est_h"] for r in q)
    for r in q:
        print(f"{r['id']:28s} {r['flops']:.3e} est {r['est_h']*60:6.1f} min")
    print(f"{len(q)} runs, est {tot:.1f} node-hours training (+~8 min/run startup+CORE = "
          f"{tot + len(q) * 8 / 60:.1f})")
    if not dry:
        with open(ROOT / "ladder_queue.jsonl", "w") as f:
            for r in q:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
