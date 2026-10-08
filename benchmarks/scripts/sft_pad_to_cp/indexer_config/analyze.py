"""Evaluate config-selection policies offline from grid.json per-config timings."""

import itertools
import json
import sys
from collections import defaultdict

def _np2(n):
    return 1 << (n - 1).bit_length()


rows = json.load(open(sys.argv[1]))
CONFIGS = list(rows[0]["ms"])


def best(r):
    return min(r["ms"].values())


def argbest(r):
    return min(r["ms"], key=r["ms"].get)


def report(pad):
    rs = [r for r in rows if r["k_pad_to"] == pad]
    print(f"\n### k_pad_to={pad}: {len(rs)} shapes")
    print("fixed config: worst ratio, mean ratio, worst shape, extra ms at worst")
    for c in CONFIGS:
        ratios = [(r["ms"][c] / best(r), r) for r in rs]
        worst, wr = max(ratios, key=lambda x: x[0])
        mean = sum(x for x, _ in ratios) / len(ratios)
        print(f"  {c:16s} worst={worst:.3f} mean={mean:.3f} at {wr['layout']},{wr['total']},r{wr['rank']}"
              f" (+{wr['ms'][c] - best(wr):.2f} ms)")

    for keyname, keyfn in [
        ("pow2(S_Q),pow2(S_K)", lambda r: (_np2(r["s_q"]), _np2(r["s_k"]))),
        ("pow2 + layout(af>0.25)", lambda r: (_np2(r["s_q"]), _np2(r["s_k"]),
                                              r["active_frac"] > 0.25)),
        ("pow2 + layout(af 3 bins)", lambda r: (_np2(r["s_q"]), _np2(r["s_k"]),
                                                0 if r["active_frac"] < 0.05 else 1 if r["active_frac"] < 0.4 else 2)),
        ("exact S_Q (old)", lambda r: (r["s_q"], _np2(r["s_k"]))),
    ]:
        groups = defaultdict(list)
        for r in rs:
            groups[keyfn(r)].append(r)
        worst, pair, losses = 1.0, None, []
        for g in groups.values():
            for a, b in itertools.product(g, g):
                ratio = b["ms"][argbest(a)] / best(b)
                losses.append(ratio)
                if ratio > worst:
                    worst, pair = ratio, (a, b)
        mean = sum(losses) / len(losses)
        desc = "" if pair is None else (f" first={pair[0]['layout']},{pair[0]['total']},r{pair[0]['rank']}"
                                        f" later={pair[1]['layout']},{pair[1]['total']},r{pair[1]['rank']}"
                                        f" (+{pair[1]['ms'][argbest(pair[0])] - best(pair[1]):.2f} ms)")
        print(f"lock-in {keyname:26s} keys={len(groups):3d} worst={worst:.3f} mean={mean:.3f}{desc}")

    print("winners by active_frac:")
    for r in sorted(rs, key=lambda r: r["active_frac"]):
        top = sorted(r["ms"], key=r["ms"].get)
        print(f"  af={r['active_frac']:.3f} {r['layout']:10s} {r['total']:6d} r{r['rank']} sq%16={r['s_q'] % 16:2d}"
              f" sk%16={r['s_k'] % 16:2d} best={best(r):7.2f} {top[0]} {top[1]}:{r['ms'][top[1]] / best(r):.2f}"
              f" w8s3N128:{r['ms']['M64_N128_w8_s3'] / best(r):.2f} w4s2N128:{r['ms']['M64_N128_w4_s2'] / best(r):.2f}")


for pad in sorted({r["k_pad_to"] for r in rows}):
    report(pad)

if {1, 16} <= {r["k_pad_to"] for r in rows}:
    print("\n### padded best vs stock best, and padded fixed config vs stock per-shape best")
    by = {(r["layout"], r["total"], r["rank"], r["k_pad_to"]): r for r in rows}
    for c in ["M64_N128_w4_s2", "M64_N128_w8_s3", "M64_N64_w4_s2"]:
        speed = []
        for (l, t, k, p), r in by.items():
            if p != 1:
                continue
            rp = by[(l, t, k, 16)]
            speed.append((best(r) / rp["ms"][c], l, t, k, best(r), rp["ms"][c]))
        lo, hi = min(speed), max(speed)
        print(f"  padded {c}: stock-best/padded-fixed min={lo[0]:.3f} ({lo[1]},{lo[2]},r{lo[3]} {lo[4]:.2f}->{lo[5]:.2f})"
              f" max={hi[0]:.3f} ({hi[1]},{hi[2]},r{hi[3]} {hi[4]:.2f}->{hi[5]:.2f})")
    tot_s = sum(best(r) for r in rows if r["k_pad_to"] == 1)
    tot_p = sum(best(r) for r in rows if r["k_pad_to"] == 16)
    print(f"  sum of per-shape best: stock={tot_s:.1f} ms padded={tot_p:.1f} ms")
