#!/usr/bin/env python
"""Full #420 ladder, two arms (plain = Karpathy's ff46300 architecture; gated s4 r8 + Muon gates).

Reads ledger.jsonl, selects the ladder runs (new runs tagged iso1e18/iso3e18/iso6e18/mini with a `ladder` field, plus
the reused loop runs listed in ladder_enqueue.REUSE), fits everything and writes
  plots/ladder/*.png            the six #420-style figures
  ladder_results.json           every point and every fit (schema documented in SCHEMA below)
  ladder_tables.md              markdown tables used by ladder_report.md

python make_ladder.py
"""
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ledger as L  # noqa: E402
from ladder_enqueue import REUSE  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

B = 2**19
SEQ = 2048
VOCAB = 32768
PLOTS = ROOT / "plots" / "ladder"
OUT_JSON = ROOT / "ladder_results.json"
COUNTS_CACHE = ROOT / "ladder_counts.json"
GPUS = 8
BUDGETS = {"iso1e18": 1e18, "iso3e18": 3e18, "iso6e18": 6e18}
ARM_OF_VARIANT = {"plain": "plain", "gated-s4r8-muon": "gated"}
ARM_LABEL = {"plain": "Plain (Karpathy's architecture)", "gated": "Gated s4 r8, Muon gates"}
ARM_SHORT = {"plain": "plain", "gated": "gated s4r8+Muon"}
ARMS = ("plain", "gated")

# ---------------------------------------------------------------- #420 reference numbers (karpathy/nanochat #420)
REF_ISO_420 = [  # Jan 7 isoflop optima: FLOPs, all params (#420 v1 convention), tokens, val bpb
    dict(flops=1e18, params=136_706_248, tokens=1_116_751_173, bpb=0.9781),
    dict(flops=3e18, params=240_236_324, tokens=1_880_909_903, bpb=0.9202),
    dict(flops=6e18, params=330_186_832, tokens=2_711_097_322, bpb=0.8874),
]
REF_MINI_V1 = [  # depth, model_dim, params_M (all), tokens_B, val_bpb, CORE, train_time_min (8xH100)
    (10, 640, 91, 0.73, 1.0312, 0.071, 5.0), (11, 704, 112, 0.89, 1.0096, 0.0918, 6.6),
    (12, 768, 135, 1.08, 0.9825, 0.1059, 7.8), (13, 832, 163, 1.30, 0.9644, 0.1015, 11.2),
    (14, 896, 194, 1.55, 0.9437, 0.1185, 13.2), (15, 960, 229, 1.83, 0.9257, 0.1158, 17.4),
    (16, 1024, 268, 2.15, 0.9101, 0.1332, 21.4), (17, 1088, 313, 2.50, 0.8952, 0.1518, 33.9),
    (18, 1152, 362, 2.90, 0.8817, 0.1611, 37.5), (19, 1216, 417, 3.33, 0.8687, 0.1659, 53.0),
    (20, 1280, 477, 3.82, 0.8572, 0.1708, 59.5)]
GPT2_CORE = [("GPT-2 124M", 0.114), ("GPT-2 355M", 0.185), ("GPT-2 774M", 0.215), ("GPT-2 1.6B", 0.257)]
GPT3_CORE = [("GPT-3 125M", 0.148), ("GPT-3 350M", 0.216), ("GPT-3 760M", 0.266), ("GPT-3 1.3B", 0.291)]
CORE_FIT_420 = dict(a=3.7555, b=0.0344, form="CORE = 1 - a * FLOPs^(-b)", fit_on="#420 v1 miniseries d>=12")

# ---------------------------------------------------------------- style (dataviz reference palette, light mode)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ARM_COLOR = {"plain": "#2a78d6", "gated": "#eb6834"}
ARM_CMAP = {"plain": cm.Blues, "gated": cm.Oranges}
REF_COLOR = "#52514e"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6, "lines.markersize": 6,
    "font.family": ["DejaVu Sans"],
})


def shade(arm, i, n):
    """Ordinal ramp in the arm's hue: light (first) -> dark (last)."""
    t = 0.38 + 0.57 * (i / max(1, n - 1))
    return ARM_CMAP[arm](t)


# ---------------------------------------------------------------- counts
def v1_counts(depth):
    """#420 v1 architecture (Jan 7): dim = 64*depth, vocab 32768, untied, no value embeds, full attention."""
    dim = 64 * depth
    wte = VOCAB * dim
    n_all = 2 * wte + 12 * depth * dim * dim + 2 * depth
    fpt = 6 * (n_all - wte) + 12 * depth * dim * SEQ  # nanochat estimate_flops
    return n_all, fpt


def load_counts():
    cache = json.loads(COUNTS_CACHE.read_text()) if COUNTS_CACHE.exists() else {}
    return cache


def counts_for(depth, arm, cache):
    key = f"{arm}-{depth}"
    if key not in cache:
        from ladder_plan import counts
        c = counts(depth, arm)
        cache[key] = {k: int(v) for k, v in c.items()}
        COUNTS_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache[key]


# ---------------------------------------------------------------- run selection
def select_runs():
    ledger = L.load()
    reuse_ids = {v: k for k, v in REUSE.items()}
    cands = defaultdict(list)
    for rid, r in ledger.items():
        arm = ARM_OF_VARIANT.get(r.get("variant"))
        if arm is None or r.get("seed") != 0:
            continue
        if rid in reuse_ids:
            tag, depth, _ = reuse_ids[rid]
            cands[(tag, arm, depth)].append((r, True))
        elif r.get("ladder") in ("iso1e18", "iso3e18", "iso6e18", "mini"):
            cands[(r["ladder"], arm, r["depth"])].append((r, False))
    chosen = {}
    for key, lst in cands.items():
        done = [x for x in lst if x[0].get("complete")]
        pick = max(done or lst, key=lambda x: x[0].get("launched_at", 0))
        chosen[key] = dict(rec=pick[0], reused=pick[1], attempts=len(lst))
    return chosen


def tokens_per_s(r):
    return r.get("tokens_per_s")


def build_points(chosen, cache):
    pts = []
    for (tag, arm, depth), x in sorted(chosen.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        r = x["rec"]
        c = counts_for(depth, arm, cache)
        fpt = r["flops_per_token"]
        tps = tokens_per_s(r)
        bpb = r.get("final_val_bpb") if r.get("final_done") else None
        curve = [dict(step=int(s), tokens=int(s) * B, flops=int(s) * B * fpt, bpb=float(v))
                 for s, v in (r.get("val_curve") or [])]
        th = r["tokens"] / tps / 3600 if tps else None
        pts.append(dict(
            run_id=r["run_id"], url=r.get("url"), sha=r.get("sha"), image_tag=r.get("image_tag"), tag=tag, arm=arm,
            depth=depth,
            seed=r.get("seed"), init_seed=r.get("init_seed"), reused=x["reused"], attempts=x["attempts"],
            status=r.get("status"), complete=bool(r.get("complete")), eval_mode=r.get("eval_mode"),
            micro_batch=r.get("micro_batch", 16), model_dim=c["wte"] // VOCAB,
            params_scaling=c["scaling"], params_all=c["all_v1"], params_total=c["total"],
            params_value_embeds=c["value_embeds"], flops_per_token=fpt, steps=r["steps"], tokens=r["tokens"],
            flops=fpt * r["tokens"], budget=BUDGETS.get(tag), weight_decay=r.get("weight_decay"),
            val_interval=r.get("val_interval"), final_val_bpb=bpb if r.get("complete") else None,
            last_val_bpb=r.get("last_val_bpb"), core=r.get("core_score"), tokens_per_s=tps,
            flops_per_s=tps * fpt if tps else None, train_hours=th, gpu_hours=th * GPUS if th else None,
            peak_mem_gib=r.get("peak_mem_gib"), val_curve=curve, note=r.get("note", "")))
    return pts


# ---------------------------------------------------------------- fits
def quad_fit(xs_n, ys):
    """Quadratic fit of y in ln N. Returns dict with coeffs, N*, y*, or None (needs >= 3 points, convex)."""
    if len(xs_n) < 3:
        return None
    x = np.log(np.asarray(xs_n, float))
    y = np.asarray(ys, float)
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return dict(coeffs=[a, b, c], convex=False)
    xs = -b / (2 * a)
    return dict(coeffs=[float(a), float(b), float(c)], convex=True, n_star=float(math.exp(xs)),
                y_star=float(a * xs * xs + b * xs + c), extrapolated=bool(xs < x.min() or xs > x.max()))


def interp_log(xs, ys, x0):
    """ln y vs ln x: quadratic (>=3 pts) / linear fit evaluated at x0."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    deg = 2 if len(xs) >= 3 else 1
    p = np.polyfit(lx, ly, deg)
    return float(math.exp(np.polyval(p, math.log(x0))))


def power_fit(xs, ys):
    """y = k * x^e in log-log; returns k, e (needs >= 2 points)."""
    if len(xs) < 2:
        return None
    e, lk = np.polyfit(np.log(xs), np.log(ys), 1)
    return dict(k=float(math.exp(lk)), exponent=float(e), n_points=len(xs))


def flops_s_at(points, arm, n_scaling):
    """Training FLOP/s (8xH200) of an arm at scaling-param count n (log-log interpolation across measured depths)."""
    by_d = defaultdict(list)
    for p in points:
        if p["arm"] == arm and p["flops_per_s"]:
            by_d[p["depth"]].append((p["params_scaling"], p["flops_per_s"]))
    if not by_d:
        return None
    xs, ys = [], []
    for d, lst in sorted(by_d.items()):
        xs.append(lst[0][0])
        ys.append(float(np.median([v for _, v in lst])))
    if len(xs) == 1:
        return ys[0]
    lx, ly = np.log(xs), np.log(ys)
    x0 = math.log(n_scaling)
    if x0 <= lx[0]:
        return ys[0]
    if x0 >= lx[-1]:
        return ys[-1]
    return float(math.exp(np.interp(x0, lx, ly)))


def isoflop_fits(points):
    fits = []
    for tag, C in BUDGETS.items():
        for arm in ARMS:
            ps = sorted([p for p in points if p["tag"] == tag and p["arm"] == arm and p["final_val_bpb"]],
                        key=lambda p: p["params_all"])
            if not ps:
                continue
            best = min(ps, key=lambda p: p["final_val_bpb"])
            ends = best is ps[0] or best is ps[-1]
            rec = dict(arm=arm, tag=tag, budget=C, depths=[p["depth"] for p in ps],
                       best_observed=dict(depth=best["depth"], bpb=best["final_val_bpb"], run_id=best["run_id"]),
                       min_at_end_rung=bool(ends and len(ps) >= 3))
            for conv, key in (("all", "params_all"), ("scaling", "params_scaling")):
                f = quad_fit([p[key] for p in ps], [p["final_val_bpb"] for p in ps])
                if not f or not f.get("convex"):
                    rec[conv] = f
                    continue
                n = f["n_star"]
                d_star = interp_log([p[key] for p in ps], [p["tokens"] for p in ps], n)
                f.update(N_star=n, D_star=d_star, D_over_N=d_star / n, bpb_star=f["y_star"])
                rec[conv] = f
            def gpu_h(n_all):
                n_s = interp_log([p["params_all"] for p in ps], [p["params_scaling"] for p in ps], n_all)
                fs = flops_s_at(points, arm, n_s)
                return (fs, C / fs * GPUS / 3600) if fs else (None, None)

            if rec.get("all") and rec["all"].get("convex"):
                rec["flops_per_s_at_opt"], rec["gpu_hours_at_opt"] = gpu_h(rec["all"]["N_star"])
            # robustness: exact parabola through the best rung and its two neighbours (None at an end rung)
            i = ps.index(best)
            if 0 < i < len(ps) - 1:
                f3 = quad_fit([p["params_all"] for p in ps[i - 1:i + 2]], [p["final_val_bpb"] for p in ps[i - 1:i + 2]])
                if f3 and f3.get("convex"):
                    f3.update(N_star=f3["n_star"], bpb_star=f3["y_star"], gpu_hours_at_opt=gpu_h(f3["n_star"])[1],
                              depths=[p["depth"] for p in ps[i - 1:i + 2]])
                rec["local3"] = f3
            rec["best_observed"]["gpu_hours"] = (C / best["flops_per_s"] * GPUS / 3600
                                                 if best.get("flops_per_s") else None)
            fits.append(rec)
    return fits


ESTIMATORS = {  # name -> (description, getter(fit_record) -> (bpb, gpu_hours) or None)
    "quadfit": ("quadratic fit of bpb in ln N over all rungs (#420 method; primary)",
                lambda f: (f["all"]["bpb_star"], f.get("gpu_hours_at_opt"))
                if f.get("all") and f["all"].get("convex") else None),
    "local3": ("exact parabola through the best rung and its two neighbours",
               lambda f: (f["local3"]["bpb_star"], f["local3"].get("gpu_hours_at_opt"))
               if f.get("local3") and f["local3"].get("convex") else None),
    "best": ("best observed rung (no fit); GPU-hours of that run",
             lambda f: (f["best_observed"]["bpb"], f["best_observed"].get("gpu_hours"))),
}


def frontier_for(fits, get):
    """Power-law frontier bpb* = A * X^-alpha per arm (X = FLOPs and X = GPU-hours) + multipliers of gated."""
    out = {"fits": {}, "multipliers": []}
    vals = {}
    for f in fits:
        v = get(f)
        if v:
            vals[(f["arm"], f["tag"])] = v
    for arm in ARMS:
        rows = [(BUDGETS[t], vals[(arm, t)][0], vals[(arm, t)][1]) for t in BUDGETS if (arm, t) in vals]
        if len(rows) >= 2:
            pf = power_fit([r[0] for r in rows], [r[1] for r in rows])
            ph = power_fit([r[2] for r in rows], [r[1] for r in rows]) if all(r[2] for r in rows) else None
            out["fits"][arm] = dict(flops=dict(A=pf["k"], alpha=-pf["exponent"], n_points=len(rows)),
                                    gpu_hours=dict(A=ph["k"], alpha=-ph["exponent"]) if ph else None,
                                    points=[dict(flops=r[0], bpb_star=r[1], gpu_hours=r[2]) for r in rows])
    pfit = out["fits"].get("plain")
    if pfit:
        A, al = pfit["flops"]["A"], pfit["flops"]["alpha"]
        for t, C in BUDGETS.items():
            if ("gated", t) not in vals or ("plain", t) not in vals:
                continue
            (bg, hg), (bp, hp) = vals[("gated", t)], vals[("plain", t)]
            c_eq = (bg / A) ** (-1 / al)
            m = dict(tag=t, budget=C, bpb_star_plain=bp, bpb_star_gated=bg, delta_bpb=bp - bg,
                     flops_plain_to_match=c_eq, flops_multiplier=c_eq / C)
            if pfit.get("gpu_hours") and hg:
                Ah, alh = pfit["gpu_hours"]["A"], pfit["gpu_hours"]["alpha"]
                h_eq = (bg / Ah) ** (-1 / alh)
                m.update(gpu_hours_gated=hg, gpu_hours_plain=hp, gpu_hours_plain_to_match=h_eq,
                         gpu_hours_multiplier=h_eq / hg)
            out["multipliers"].append(m)
    return out


def frontier(fits):
    out = {"primary_estimator": "quadfit", "estimators": {}}
    for name, (desc, get) in ESTIMATORS.items():
        out["estimators"][name] = dict(description=desc, **frontier_for(fits, get))
    out["fits"] = out["estimators"]["quadfit"]["fits"]
    out["multipliers"] = out["estimators"]["quadfit"]["multipliers"]
    return out


def optimal_scaling(fits):
    out = {}
    for arm in ARMS:
        for conv in ("all", "scaling"):
            rows = [f for f in fits if f["arm"] == arm and f.get(conv) and f[conv].get("convex")]
            if len(rows) < 2:
                continue
            C = [f["budget"] for f in rows]
            out[f"{arm}/{conv}"] = dict(arm=arm, convention=conv,
                                        N_star=power_fit(C, [f[conv]["N_star"] for f in rows]),
                                        D_star=power_fit(C, [f[conv]["D_star"] for f in rows]))
    C = [r["flops"] for r in REF_ISO_420]
    out["420/all"] = dict(arm="420", convention="all", N_star=power_fit(C, [r["params"] for r in REF_ISO_420]),
                          D_star=power_fit(C, [r["tokens"] for r in REF_ISO_420]))
    return out


def same_depth_gains(points):
    """plain - gated final bpb for every (tag, depth) both arms ran (isoflop rungs are FLOPs-matched; miniseries
    rungs are 8 x params per arm, so gated also gets slightly more FLOPs there)."""
    by = {(p["tag"], p["arm"], p["depth"]): p for p in points if p["final_val_bpb"]}
    out = []
    for (tag, arm, d), g in sorted(by.items()):
        if arm != "gated" or (tag, "plain", d) not in by:
            continue
        q = by[(tag, "plain", d)]
        out.append(dict(tag=tag, depth=d, params_all_gated=g["params_all"], params_all_plain=q["params_all"],
                        tokens_per_param_gated=g["tokens"] / g["params_all"], flops_gated=g["flops"],
                        flops_plain=q["flops"], bpb_plain=q["final_val_bpb"], bpb_gated=g["final_val_bpb"],
                        delta_bpb=q["final_val_bpb"] - g["final_val_bpb"],
                        core_plain=q["core"], core_gated=g["core"], tokens_per_s_ratio=(
                            g["tokens_per_s"] / q["tokens_per_s"] if g["tokens_per_s"] and q["tokens_per_s"] else None)))
    return out


def mini_multipliers(points):
    """Effective multipliers along the miniseries (tokens = 8 x params, close to compute-optimal here: D*/N* ~ 8-9):
    plain FLOPs (GPU-hours) needed to reach each gated point's bpb, by log-log interpolation along plain's
    miniseries, divided by that gated point's FLOPs (GPU-hours). None outside plain's range."""
    P = [p for p in points if p["tag"] == "mini" and p["arm"] == "plain" and p["final_val_bpb"]]
    G = [p for p in points if p["tag"] == "mini" and p["arm"] == "gated" and p["final_val_bpb"]]
    if len(P) < 2:
        return []
    lb = np.log([p["final_val_bpb"] for p in P])
    o = np.argsort(lb)
    lc = np.log([p["flops"] for p in P])[o]
    lh = np.log([p["gpu_hours"] for p in P])[o]
    lb = lb[o]
    out = []
    for g in sorted(G, key=lambda p: p["depth"]):
        x = math.log(g["final_val_bpb"])
        inside = lb[0] <= x <= lb[-1]
        c_eq = math.exp(np.interp(x, lb, lc)) if inside else None
        h_eq = math.exp(np.interp(x, lb, lh)) if inside else None
        out.append(dict(depth=g["depth"], flops_gated=g["flops"], bpb_gated=g["final_val_bpb"],
                        flops_plain_to_match=c_eq, flops_multiplier=c_eq / g["flops"] if c_eq else None,
                        gpu_hours_gated=g["gpu_hours"], gpu_hours_plain_to_match=h_eq,
                        gpu_hours_multiplier=h_eq / g["gpu_hours"] if h_eq and g["gpu_hours"] else None))
    return out


def core_fit(xs, ys):
    """CORE = 1 - a * FLOPs^(-b), fit as ln(1-CORE) = ln a - b ln FLOPs."""
    if len(xs) < 2:
        return None
    e, la = np.polyfit(np.log(xs), np.log(1 - np.asarray(ys)), 1)
    return dict(a=float(math.exp(la)), b=float(-e), n_points=len(xs))


def core_fits(points):
    out = {}
    for arm in ARMS:
        ps = [p for p in points if p["tag"] == "mini" and p["arm"] == arm and p["core"] is not None
              and p["depth"] >= 12 and p["complete"]]
        f = core_fit([p["flops"] for p in ps], [p["core"] for p in ps])
        if f:
            f["depths"] = sorted(p["depth"] for p in ps)
            out[arm] = f
    v1 = [(v1_counts(d)[1] * tok * 1e9, core) for d, _, _, tok, _, core, _ in REF_MINI_V1 if d >= 12]
    f = core_fit([x for x, _ in v1], [y for _, y in v1])
    out["420_v1_refit"] = dict(f, note="refit on #420 v1 table with FLOPs = tokens x nanochat estimate_flops of the "
                                       "v1 architecture (our estimate); published: a=3.7555, b=0.0344")
    return out


# ---------------------------------------------------------------- figures
def fmt_c(c):
    return f"{c:.0e}".replace("e+", "e")


def fmt_params(n):
    return f"{n / 1e6:.0f}M" if n < 1e9 else f"{n / 1e9:.2f}B"


def param_fmt():
    return matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M" if v < 1e9 else f"{v / 1e9:.1f}B")


def flops_fmt():
    return matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e+", "e"))


def set_param_ticks(ax, ticks=(50e6, 70e6, 100e6, 150e6, 200e6, 300e6, 500e6, 700e6)):
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(param_fmt())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def repel_labels(ax, items, dx=5, min_px=10, color=INK2, fontsize=7.5):
    """items: [(x, y, text)] -> right-of-point labels spread vertically so they never overlap."""
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    tr = ax.transData
    pix = sorted(((tr.transform((x, y)), x, y, t) for x, y, t in items), key=lambda r: r[0][1])
    ys = [r[0][1] for r in pix]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_px)
    shift = (sum(r[0][1] for r in pix) - sum(ys)) / len(ys)  # recentre the stack on the points
    for (pt, x, y, t), yy in zip(pix, ys):
        ax.annotate(t, (x, y), xytext=(dx, (yy + shift - pt[1]) * 72 / fig.dpi), textcoords="offset points",
                    va="center", ha="left", fontsize=fontsize, color=color)


def end_label(ax, x, y, text, color=INK2, dx=4):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left", fontsize=8,
                color=color)


def fig_isoflop(points, fits):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    tags = list(BUDGETS)
    for ax, arm in zip(axes, ARMS):
        for i, tag in enumerate(tags):
            col = shade(arm, i, len(tags))
            ps = sorted([p for p in points if p["tag"] == tag and p["arm"] == arm and p["final_val_bpb"]],
                        key=lambda p: p["params_all"])
            if not ps:
                continue
            x = [p["params_all"] for p in ps]
            y = [p["final_val_bpb"] for p in ps]
            ax.plot(x, y, "o", color=col, ms=6, mec=SURFACE, mew=1.0, zorder=3)
            for p in ps:
                ax.annotate(f"d{p['depth']}", (p["params_all"], p["final_val_bpb"]), xytext=(0, -11),
                            textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
            f = next((f for f in fits if f["arm"] == arm and f["tag"] == tag), None)
            if f and f.get("all") and f["all"].get("convex"):
                a, b, c = f["all"]["coeffs"]
                xx = np.linspace(math.log(min(x)) - 0.08, math.log(max(x)) + 0.08, 200)
                ax.plot(np.exp(xx), a * xx * xx + b * xx + c, "-", color=col, lw=1.6, zorder=2)
                ax.plot(f["all"]["N_star"], f["all"]["bpb_star"], "*", color=col, ms=15, mec=INK, mew=0.6, zorder=5)
                ax.annotate(f"{tag[3:]} FLOPs", (max(x), y[-1]), xytext=(4, 9), textcoords="offset points",
                            ha="left", va="bottom", fontsize=8, color=INK2)
            else:
                ax.plot(x, y, "-", color=col, lw=1.2, alpha=0.6)
        if arm == "plain":
            for r in REF_ISO_420:
                ax.plot(r["params"], r["bpb"], "o", mfc="none", mec=REF_COLOR, mew=1.4, ms=9, zorder=4)
                ax.annotate(f"#420 {r['flops']:.0e}".replace("e+", "e"), (r["params"], r["bpb"]), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=7, color=INK2)
            ax.plot([], [], "o", mfc="none", mec=REF_COLOR, mew=1.4, ms=9, label="#420 published optima")
            ax.plot([], [], "*", color=MUTED, ms=12, mec=INK, mew=0.6, label="quadratic-fit optimum (ours)")
            ax.legend(loc="upper right")
        ax.set_title(ARM_LABEL[arm])
        ax.set_xscale("log")
        set_param_ticks(ax)
        ax.set_xlabel("parameters (all, #420 convention: embeddings + matrices + scalars, no value embeds)",
                      fontsize=8.5)
    axes[0].set_ylabel("final val bpb")
    fig.suptitle("Isoflop curves: val bpb vs model size at fixed FLOPs (#420 'scaling3')", color=INK, fontsize=12)
    fig.text(0.5, 0.005, "Hollow markers: #420 Jan 7 optima (v1 architecture, Karpathy's val shard). Our val split "
             "(fineweb100b-packed/val) and architecture (ff46300: value embeds, SSSL windows) differ, so compare "
             "shapes and locations, not absolute bpb.", ha="center", fontsize=7.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, "1_isoflop_ucurves.png")


def fig_optimal(fits, opt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    C = np.logspace(17.8, 19.0, 50)
    for arm in ARMS:
        rows = [f for f in fits if f["arm"] == arm and f.get("all") and f["all"].get("convex")]
        if not rows:
            continue
        col = ARM_COLOR[arm]
        bc = [f["budget"] for f in rows]
        for ax, key in ((axes[0], "N_star"), (axes[1], "D_star")):
            ys = [f["all"][key] for f in rows]
            fit = opt.get(f"{arm}/all", {}).get(key)
            lab = ARM_SHORT[arm] + (f"  ∝ C^{fit['exponent']:.2f}" if fit else "")
            ax.plot(bc, ys, "o", color=col, ms=7, mec=SURFACE, mew=1, label=lab, zorder=3)
            if fit:
                ax.plot(C, fit["k"] * C ** fit["exponent"], "-", color=col, lw=1.4, zorder=2)
        for conv, ls in (("all", "-"), ("scaling", "--")):
            rr = [f for f in fits if f["arm"] == arm and f.get(conv) and f[conv].get("convex")]
            if rr:
                axes[2].plot([f["budget"] for f in rr], [f[conv]["D_over_N"] for f in rr], ls, marker="o",
                             color=col, ms=6, mec=SURFACE, label=f"{ARM_SHORT[arm]}, {conv} params")
    ref = opt["420/all"]
    rc = [r["flops"] for r in REF_ISO_420]
    for ax, key, k2 in ((axes[0], "N_star", "params"), (axes[1], "D_star", "tokens")):
        fit = ref[key]
        ax.plot(rc, [r[k2] for r in REF_ISO_420], "o", mfc="none", mec=REF_COLOR, mew=1.4, ms=9,
                label=f"#420 optima  ∝ C^{fit['exponent']:.2f}", zorder=4)
        ax.plot(C, fit["k"] * C ** fit["exponent"], ":", color=REF_COLOR, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training FLOPs")
        ax.legend(loc="upper left")
    axes[2].plot(rc, [r["tokens"] / r["params"] for r in REF_ISO_420], "o", mfc="none", mec=REF_COLOR, mew=1.4,
                 ms=9, label="#420 optima (all params)")
    axes[2].axhline(8, color=MUTED, lw=0.8, ls=":")
    axes[2].annotate("#420: ~8", (1.05e18, 8), xytext=(0, 3), textcoords="offset points", fontsize=7.5,
                     color=MUTED)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("training FLOPs")
    axes[2].set_ylim(5.5, 11.5)
    axes[2].legend(loc="lower right", fontsize=7, handlelength=3.2, frameon=True, facecolor=SURFACE,
                   edgecolor=GRID, framealpha=1)
    tok_fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e9:.1f}B")
    for ax, fmt in ((axes[0], param_fmt()), (axes[1], tok_fmt)):
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.tick_params(axis="y", which="minor", labelsize=7)
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e18, 2e18, 3e18, 6e18, 1e19]))
        ax.xaxis.set_major_formatter(flops_fmt())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("N* (all params, #420 convention)")
    axes[1].set_ylabel("D* (tokens)")
    axes[2].set_ylabel("D* / N*")
    axes[0].set_title("Compute-optimal model size")
    axes[1].set_title("Compute-optimal tokens")
    axes[2].set_title("Tokens per parameter at the optimum")
    fig.suptitle("Optimal N* and D* vs FLOPs (quadratic fit in log N at each budget, power-law fit across budgets)",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "2_optimal_N_D_vs_flops.png")


def fig_runs_3e18(points):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    depths = sorted({p["depth"] for p in points if p["tag"] == "iso3e18"})
    ymin = min([p["final_val_bpb"] for p in points if p["tag"] == "iso3e18" and p["final_val_bpb"]] or [0.84])
    labels = {}
    for ax, arm in zip(axes, ARMS):
        labels[arm] = []
        for p in sorted([p for p in points if p["tag"] == "iso3e18" and p["arm"] == arm], key=lambda p: p["depth"]):
            col = shade(arm, depths.index(p["depth"]), len(depths))
            cv = p["val_curve"]
            if not cv:
                continue
            x = [c["flops"] for c in cv]
            y = [c["bpb"] for c in cv]
            ax.plot(x, y, "-", color=col, lw=1.6, marker="o" if len(cv) < 8 else None, ms=3.5)
            txt = f"d{p['depth']}  {y[-1]:.4f}" if p["complete"] else f"d{p['depth']} (running)"
            labels[arm].append((x[-1], y[-1], txt))
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("training FLOPs")
        ax.set_ylim(ymin - 0.01, ymin + 0.20)
        ax.set_xlim(0, 3.9e18)
    for ax, arm in zip(axes, ARMS):
        repel_labels(ax, labels[arm])
    axes[0].set_ylabel("val bpb")
    fig.suptitle("Individual runs at 3e18 FLOPs, one per depth (#420 'scaling_3e18')", color=INK, fontsize=12)
    fig.text(0.5, 0.005, "Plain d14/d16/d18 are reused loop runs with 5-13 val evals; the rest log ~20.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return save(fig, "3_runs_3e18.png")


def fig_miniseries(points):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    depths = list(range(10, 21))
    v1x = [v1_counts(d)[1] * tok * 1e9 for d, _, _, tok, _, _, _ in REF_MINI_V1]
    v1y = [r[4] for r in REF_MINI_V1]
    for ax, arm in zip(axes, ARMS):
        for p in sorted([p for p in points if p["tag"] == "mini" and p["arm"] == arm], key=lambda p: p["depth"]):
            col = shade(arm, depths.index(p["depth"]), len(depths))
            cv = p["val_curve"]
            if not cv:
                continue
            x = [c["flops"] for c in cv]
            y = [c["bpb"] for c in cv]
            ax.plot(x, y, "-", color=col, lw=1.4)
            if p["complete"]:
                ax.plot(x[-1], y[-1], "o", color=col, ms=5, mec=SURFACE, mew=0.8)
            if p["depth"] % 2 == 0 or p["depth"] in (11, 19):
                end_label(ax, x[-1], y[-1], f"d{p['depth']}", dx=5)
        ax.plot(v1x, v1y, "o", mfc="none", mec=REF_COLOR, mew=1.2, ms=7, label="#420 v1 final (d10-d20)")
        ax.set_xscale("log")
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("training FLOPs")
        ax.set_ylim(0.77, 1.12)
        ax.set_xlim(6e16, 2.2e19)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e17, 3e17, 1e18, 3e18, 1e19]))
        ax.xaxis.set_major_formatter(flops_fmt())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(loc="upper right")
    axes[0].set_ylabel("val bpb")
    fig.suptitle("Miniseries d10-d20, tokens = 8 x all params (#420 'jan7_runs2')", color=INK, fontsize=12)
    fig.text(0.5, 0.005, "#420 v1 points at FLOPs = v1 tokens x nanochat estimate_flops of the v1 architecture "
             "(our estimate); different val split and architecture, so the bpb gap is not a like-for-like comparison.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return save(fig, "4_miniseries_curves.png")


def fig_core(points, cfits):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for arm in ARMS:
        ps = sorted([p for p in points if p["tag"] == "mini" and p["arm"] == arm and p["core"] is not None
                     and p["complete"]], key=lambda p: p["depth"])
        if not ps:
            continue
        col = ARM_COLOR[arm]
        f = cfits.get(arm)
        lab = ARM_SHORT[arm] + (f"  (fit d≥12: 1 − {f['a']:.3f}·C^−{f['b']:.4f})" if f else "")
        axes[0].plot([p["flops"] for p in ps], [p["core"] for p in ps], "o-", color=col, ms=6, lw=1.2,
                     mec=SURFACE, label=lab, zorder=3)
        if f:
            C = np.logspace(17.5, 19.2, 60)
            axes[0].plot(C, 1 - f["a"] * C ** (-f["b"]), "--", color=col, lw=1.0, zorder=2)
        hrs = [(p["train_hours"], p["core"]) for p in ps if p["train_hours"]]
        axes[1].plot([h for h, _ in hrs], [c for _, c in hrs], "o-", color=col, ms=6, lw=1.2, mec=SURFACE,
                     label=f"{ARM_SHORT[arm]} (8xH200, training time)", zorder=3)
        for p in ps:
            if p["depth"] in (12, 16, 20):
                axes[0].annotate(f"d{p['depth']}", (p["flops"], p["core"]), xytext=(4, -9 if arm == "gated" else 5),
                                 textcoords="offset points", fontsize=7, color=MUTED)
    v1x = [v1_counts(d)[1] * tok * 1e9 for d, _, _, tok, _, _, _ in REF_MINI_V1]
    axes[0].plot(v1x, [r[5] for r in REF_MINI_V1], "o", mfc="none", mec=REF_COLOR, mew=1.2, ms=7,
                 label="#420 v1 miniseries (FLOPs: our estimate)")
    axes[1].plot([r[6] / 60 for r in REF_MINI_V1], [r[5] for r in REF_MINI_V1], "o", mfc="none", mec=REF_COLOR,
                 mew=1.2, ms=7, label="#420 v1 miniseries (8xH100)")
    for ax in axes:
        for lab, v in GPT2_CORE:
            ax.axhline(v, color=MUTED, lw=0.7, ls=":")
        for lab, v in GPT3_CORE:
            ax.axhline(v, color=MUTED, lw=0.7, ls="--", dashes=(4, 3))
        ax.set_xscale("log")
        ax.set_ylim(0.04, 0.30)
    # reference labels on the right panel only (merge the two that coincide)
    labs = {}
    for lab, v in GPT2_CORE + GPT3_CORE:
        labs.setdefault(round(v, 2), []).append(f"{lab} {v:.3f}")
    xr = axes[1].get_xlim()[1] if axes[1].lines else 3
    for v, ls in labs.items():
        axes[1].annotate(" / ".join(ls), (1.0, v), xycoords=("axes fraction", "data"), xytext=(3, 0),
                         textcoords="offset points", va="center", fontsize=6.8, color=INK2)
    axes[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([3e17, 1e18, 3e18, 1e19]))
    axes[0].xaxis.set_major_formatter(flops_fmt())
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2]))
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xlabel("training FLOPs")
    axes[1].set_xlabel("training hours on one 8-GPU node")
    axes[0].set_ylabel("CORE")
    axes[0].set_title("CORE vs FLOPs")
    axes[1].set_title("CORE vs wall-clock")
    for ax in axes:
        ax.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1)
    fig.suptitle("CORE of the miniseries (#420 'jan7'); dotted = GPT-2, dashed = GPT-3 reference CORE",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    return save(fig, "5_core_vs_flops_and_hours.png")


def fig_frontier(fits, front):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for arm in ARMS:
        ff = front["fits"].get(arm)
        rows = [f for f in fits if f["arm"] == arm and f.get("all") and f["all"].get("convex")]
        if not rows:
            continue
        col = ARM_COLOR[arm]
        xs = [f["budget"] for f in rows]
        ys = [f["all"]["bpb_star"] for f in rows]
        hs = [f.get("gpu_hours_at_opt") for f in rows]
        lab = ARM_SHORT[arm] + (f"  bpb* = {ff['flops']['A']:.3f}·C^−{ff['flops']['alpha']:.4f}" if ff else "")
        axes[0].plot(xs, ys, "*", color=col, ms=13, mec=INK, mew=0.5, label=lab, zorder=4)
        bo = [f["best_observed"]["bpb"] for f in rows]
        axes[0].plot(xs, bo, "o", mfc="none", mec=col, ms=6, mew=1.0, zorder=3)
        if ff:
            C = np.logspace(17.9, 18.85, 40)
            axes[0].plot(C, ff["flops"]["A"] * C ** (-ff["flops"]["alpha"]), "-", color=col, lw=1.3)
        if all(hs):
            labh = ARM_SHORT[arm] + (f"  bpb* = {ff['gpu_hours']['A']:.3f}·H^−{ff['gpu_hours']['alpha']:.4f}"
                                     if ff and ff.get("gpu_hours") else "")
            axes[1].plot(hs, ys, "*", color=col, ms=13, mec=INK, mew=0.5, label=labh, zorder=4)
            if ff and ff.get("gpu_hours"):
                H = np.logspace(math.log10(min(hs)) - 0.08, math.log10(max(hs)) + 0.08, 40)
                axes[1].plot(H, ff["gpu_hours"]["A"] * H ** (-ff["gpu_hours"]["alpha"]), "-", color=col, lw=1.3)
            for f, h, y in zip(rows, hs, ys):
                axes[1].annotate(f"{f['tag'][3:]}", (h, y), xytext=(5, 4), textcoords="offset points",
                                 fontsize=7, color=MUTED)
    for m in front["multipliers"]:
        txt = f"×{m['flops_multiplier']:.2f}"
        axes[0].annotate(txt, (m["budget"], m["bpb_star_gated"]), xytext=(0, -16), textcoords="offset points",
                         ha="center", fontsize=8, color=INK2)
        if m.get("gpu_hours_multiplier"):
            axes[1].annotate(f"×{m['gpu_hours_multiplier']:.2f}", (m["gpu_hours_gated"], m["bpb_star_gated"]),
                             xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    axes[0].plot([], [], "o", mfc="none", mec=MUTED, ms=6, label="best observed rung")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e18, 2e18, 3e18, 6e18]))
    axes[0].xaxis.set_major_formatter(flops_fmt())
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 3, 5, 8, 12]))
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xlabel("training FLOPs")
    axes[1].set_xlabel("GPU-hours (H200) at the optimum, training only")
    axes[0].set_ylabel("compute-optimal val bpb (bpb*)")
    axes[0].set_title("Frontier vs FLOPs  (×k = effective compute multiplier of gated)")
    axes[1].set_title("Frontier vs GPU-hours  (×k = GPU-hour multiplier of gated)")
    fig.suptitle("Headline frontier: compute-optimal bpb, 1e18-6e18 FLOPs", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "6_frontier.png")


def fig_gain(gains, front, mm):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    tags = list(BUDGETS)
    for i, tag in enumerate(tags):
        rows = [g for g in gains if g["tag"] == tag]
        col = cm.Purples(0.45 + 0.5 * i / (len(tags) - 1))
        ax.plot([g["params_all_gated"] for g in rows], [g["delta_bpb"] for g in rows], "o-", color=col, ms=6,
                mec=SURFACE, lw=1.4, label=f"isoflop {tag[3:]} (FLOPs-matched)")
    rows = [g for g in gains if g["tag"] == "mini"]
    ax.plot([g["params_all_gated"] for g in rows], [g["delta_bpb"] for g in rows], "s--", color="#1baf7a", ms=5,
            mec=SURFACE, lw=1.2, label="miniseries (8 x params per arm)")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xscale("log")
    set_param_ticks(ax, (50e6, 100e6, 200e6, 300e6, 500e6, 700e6))
    ax.set_xlabel("parameters (all, #420 convention; gated model)")
    ax.set_ylabel("val bpb gain of gated (plain − gated)")
    ax.set_title("Same-depth gain shrinks with model size")
    ax.legend(loc="upper right", fontsize=7.5)
    ax = axes[1]
    mk = {"quadfit": "o", "local3": "D", "best": "^"}
    for name, est in front["estimators"].items():
        ms = est["multipliers"]
        ax.plot([m["budget"] for m in ms], [m["flops_multiplier"] for m in ms], mk[name], color="#6250d6", ms=7,
                mfc="#6250d6" if name == "quadfit" else "none", mew=1.3, ls="-" if name == "quadfit" else "none",
                lw=1.3, label=f"FLOPs, isoflop optima ({name})")
        ax.plot([m["budget"] for m in ms], [m.get("gpu_hours_multiplier") for m in ms], mk[name], color="#1baf7a",
                ms=7, mfc="#1baf7a" if name == "quadfit" else "none", mew=1.3,
                ls="-" if name == "quadfit" else "none", lw=1.3, label=f"GPU-hours, isoflop optima ({name})")
    mmf = [m for m in mm if m["flops_multiplier"]]
    ax.plot([m["flops_gated"] for m in mmf], [m["flops_multiplier"] for m in mmf], ":", color="#6250d6", lw=1.2,
            marker=".", ms=6, label="FLOPs, along the miniseries")
    ax.plot([m["flops_gated"] for m in mmf], [m["gpu_hours_multiplier"] for m in mmf], ":", color="#1baf7a",
            lw=1.2, marker=".", ms=6, label="GPU-hours, along the miniseries")
    ax.axhline(1.0, color=INK2, lw=0.9)
    ax.annotate("break-even", (3.5e17, 1.0), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([3e17, 1e18, 3e18, 6e18, 1e19]))
    ax.xaxis.set_major_formatter(flops_fmt())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.5, 1.25)
    ax.set_xlabel("training FLOPs (gated)")
    ax.set_ylabel("gated multiplier (plain needs ×k to match)")
    ax.set_title("Effective multiplier: compute (violet) vs GPU-hours (aqua)")
    ax.legend(loc="center", fontsize=6.8, ncol=2, bbox_to_anchor=(0.5, 0.56))
    fig.suptitle("Where the gated advantage comes from, and what it costs", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "7_gain_vs_scale.png")


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


# ---------------------------------------------------------------- tables
def tables(points, fits, front, cfits, opt, gains, mm):
    lines = []
    lines.append("### Compute-optimal points (quadratic fit of val bpb in log N per budget)\n")
    lines.append("| budget | arm | N* all params | N* scaling params | D* tokens | D*/N* (all) | D*/N* (scaling) "
                 "| bpb* | best rung | GPU-h at opt |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for f in fits:
        a, s = f.get("all") or {}, f.get("scaling") or {}
        if not a.get("convex"):
            lines.append(f"| {f['tag'][3:]} | {f['arm']} | n/a | n/a | n/a | | | | d{f['best_observed']['depth']} "
                         f"{f['best_observed']['bpb']:.4f} | |")
            continue
        gh = f.get("gpu_hours_at_opt")
        flag = " (end rung)" if f["min_at_end_rung"] else ""
        lines.append(f"| {f['tag'][3:]} | {f['arm']} | {fmt_params(a['N_star'])} | {fmt_params(s['N_star'])} | "
                     f"{a['D_star'] / 1e9:.2f}B | {a['D_over_N']:.1f} | {s['D_over_N']:.1f} | {a['bpb_star']:.4f} | "
                     f"d{f['best_observed']['depth']} {f['best_observed']['bpb']:.4f}{flag} | "
                     + (f"{gh:.2f} |" if gh else " |"))
    for r in REF_ISO_420:
        lines.append(f"| {fmt_c(r['flops'])} | #420 | {fmt_params(r['params'])} | | {r['tokens'] / 1e9:.2f}B | "
                     f"{r['tokens'] / r['params']:.1f} | | {r['bpb']:.4f} (their val) | | |")
    lines.append("\n### Power-law fits of the optima (N* = k·C^a, D* = k·C^b)\n")
    lines.append("| arm | convention | N* exponent | D* exponent | points |")
    lines.append("|---|---|---:|---:|---:|")
    for k, v in opt.items():
        if v.get("N_star"):
            lines.append(f"| {v['arm']} | {v['convention']} | {v['N_star']['exponent']:.3f} | "
                         f"{v['D_star']['exponent']:.3f} | {v['N_star']['n_points']} |")
    for name, est in front["estimators"].items():
        lines.append(f"\n### Gated vs plain multipliers — estimator `{name}`: {est['description']}\n")
        lines.append("Plain frontier bpb* = A·X^-α through the plain points (X = FLOPs, or GPU-hours at the optimum); "
                     "multiplier = X plain needs to reach gated's bpb* / X gated used.\n")
        lines.append("| budget | bpb* plain | bpb* gated | Δ | FLOPs multiplier | GPU-h gated | GPU-h plain needs "
                     "| GPU-hour multiplier |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in est["multipliers"]:
            lines.append(f"| {fmt_c(m['budget'])} | {m['bpb_star_plain']:.4f} | {m['bpb_star_gated']:.4f} | "
                         f"{m['delta_bpb']:+.4f} | {m['flops_multiplier']:.3f}× | {m.get('gpu_hours_gated', 0):.2f} | "
                         f"{m.get('gpu_hours_plain_to_match', 0):.2f} | "
                         f"{m.get('gpu_hours_multiplier', float('nan')):.3f}× |")
        for arm, ff in est["fits"].items():
            gh = ff.get("gpu_hours")
            lines.append(f"\n- {arm}: bpb* = {ff['flops']['A']:.4f}·C^-{ff['flops']['alpha']:.4f}"
                         + (f"; vs GPU-hours bpb* = {gh['A']:.4f}·H^-{gh['alpha']:.4f}" if gh else ""))
    lines.append("\n### Miniseries (tokens = 8 × all params) vs #420 v1\n")
    lines.append("| depth | plain bpb | gated bpb | Δ (plain−gated) | #420 v1 bpb | plain CORE | gated CORE "
                 "| #420 v1 CORE | plain h | gated h |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    mini = {(p["arm"], p["depth"]): p for p in points if p["tag"] == "mini"}
    for d, _, _, _, vb, vc, _ in REF_MINI_V1:
        p, g = mini.get(("plain", d)), mini.get(("gated", d))

        def f4(x, k):
            return f"{x[k]:.4f}" if x and x.get(k) is not None else "–"

        dl = (f"{p['final_val_bpb'] - g['final_val_bpb']:+.4f}" if p and g and p["final_val_bpb"]
              and g["final_val_bpb"] else "–")
        hp = f"{p['train_hours']:.2f}" if p and p["complete"] and p.get("train_hours") else "–"
        hg = f"{g['train_hours']:.2f}" if g and g["complete"] and g.get("train_hours") else "–"
        lines.append(f"| d{d} | {f4(p, 'final_val_bpb')} | {f4(g, 'final_val_bpb')} | {dl} | {vb:.4f} | "
                     f"{f4(p, 'core')} | {f4(g, 'core')} | {vc:.4f} | {hp} | {hg} |")
    lines.append("\n### Miniseries effective multipliers (plain FLOPs / GPU-hours to match gated, interpolated along "
                 "plain's miniseries)\n")
    lines.append("| depth | gated FLOPs | gated bpb | FLOPs multiplier | GPU-hour multiplier |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in mm:
        fm = f"{m['flops_multiplier']:.3f}×" if m["flops_multiplier"] else "– (beyond plain's range)"
        gm = f"{m['gpu_hours_multiplier']:.3f}×" if m["gpu_hours_multiplier"] else "–"
        lines.append(f"| d{m['depth']} | {m['flops_gated']:.2e} | {m['bpb_gated']:.4f} | {fm} | {gm} |")
    lines.append("\n### Same-depth gains (plain − gated val bpb)\n")
    lines.append("| set | depth | gated tokens/param (all) | plain bpb | gated bpb | gain | gated/plain tok/s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for g in gains:
        tr = f"{g['tokens_per_s_ratio']:.2f}" if g["tokens_per_s_ratio"] else "–"
        lines.append(f"| {g['tag']} | d{g['depth']} | {g['tokens_per_param_gated']:.1f} | {g['bpb_plain']:.4f} | "
                     f"{g['bpb_gated']:.4f} | {g['delta_bpb']:+.4f} | {tr} |")
    lines.append("\n### CORE fits (miniseries d ≥ 12): CORE = 1 − a·FLOPs^(−b)\n")
    for k, v in cfits.items():
        lines.append(f"- {k}: a = {v['a']:.4f}, b = {v['b']:.4f} ({v['n_points']} points)")
    lines.append(f"- #420 published (Jan 7): a = {CORE_FIT_420['a']}, b = {CORE_FIT_420['b']}")
    return "\n".join(lines) + "\n"


SCHEMA = {
    "schema_version": "1",
    "top_level": {
        "meta": "generation time, conventions, units, caveats",
        "arms": "arm id -> label, residual config, optimizer overrides",
        "reference_420": "Karpathy #420 numbers used in the figures (isoflop optima, v1 miniseries, GPT-2/3 CORE, CORE fit)",
        "runs": "one entry per ladder point (see run_fields); val_curve is the in-run val bpb series",
        "isoflop_fits": "per (arm, budget): quadratic fit of final val bpb in ln(N) for N in both param conventions",
        "optimal_scaling_fits": "power laws N* = k*C^exponent and D* = k*C^exponent per arm and convention (+ #420)",
        "frontier": "estimators.{quadfit,local3,best}: per arm bpb* = A*C^-alpha (C = FLOPs) and bpb* = A*H^-alpha "
                    "(H = GPU-hours at the optimum), plus per-budget multipliers of gated vs plain (FLOPs and "
                    "GPU-hours plain needs to reach gated's bpb*, over gated's own). Top-level fits/multipliers = "
                    "the primary estimator (quadfit, the #420 method)",
        "core_fits": "CORE = 1 - a*FLOPs^-b per arm on miniseries d>=12 (+ refit of #420 v1 with our FLOPs estimate)",
        "same_depth_gains": "plain - gated final bpb for every (tag, depth) both arms ran, with tokens/param and CORE",
        "miniseries_multipliers": "per gated miniseries depth: plain FLOPs / GPU-hours needed to match it (log-log "
                                  "interpolation along plain's miniseries) over gated's own",
        "figures": "relative paths of the PNGs",
    },
    "run_fields": {
        "run_id/url/sha": "Model Factory run id, dashboard URL, prime-rl commit",
        "image_tag": "prime-rl image pinned at launch (null = backend default of 2026-11-11 before 22:40 UTC, same "
                     "heavy deps as v0.10.1.dev176)",
        "tag": "iso1e18 | iso3e18 | iso6e18 | mini", "arm": "plain | gated", "depth": "n_layer",
        "reused": "true if reused from the earlier autoresearch loop (identical config and math)",
        "params_all": "#420 convention: wte + lm_head + transformer matrices (incl. residual-connection params) + "
                      "scalars; excludes value embeddings", "params_scaling": "ff46300 convention: matrices + lm_head",
        "params_total": "every parameter incl. value embeddings", "flops_per_token": "nanochat estimate_flops + "
        "connection extras", "tokens/steps/flops": "training horizon; flops = flops_per_token * tokens",
        "final_val_bpb": "val bpb after the last step (null if not complete)", "core": "final CORE (null if none)",
        "tokens_per_s": "median training throughput on 8xH200", "train_hours": "tokens / tokens_per_s / 3600",
        "gpu_hours": "8 * train_hours", "val_curve": "[{step, tokens, flops, bpb}]",
    },
}


def main():
    cache = load_counts()
    chosen = select_runs()
    points = build_points(chosen, cache)
    fits = isoflop_fits(points)
    opt = optimal_scaling(fits)
    front = frontier(fits)
    cfits = core_fits(points)
    gains = same_depth_gains(points)
    mm = mini_multipliers(points)
    figs = [fig_isoflop(points, fits), fig_optimal(fits, opt), fig_runs_3e18(points), fig_miniseries(points),
            fig_core(points, cfits), fig_frontier(fits, front), fig_gain(gains, front, mm)]
    res = dict(
        schema=SCHEMA,
        meta=dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  batch_tokens=B, seq_len=SEQ, gpus_per_run=GPUS, gpu_type="H200_141GB",
                  val="in-trainer val bpb on the first 10240 rows (40 x 2^19 tokens) of fineweb100b-packed/val",
                  hyperparams="seed 0 (init_seed 42, fixed data order); wd = 0.2*D_REF/D, D_REF = 10.5 * d12 scaling "
                              "params; warmdown 0.5; AdamW emb/unemb LR x (d_model/768)^-0.5; batch 2^19",
                  isoflop_steps="steps = round(C / (flops_per_token * 2^19)), nudged <= 3 so the val interval divides",
                  miniseries_tokens="8 x params_all of that arm",
                  caveats=["#420 numbers use Karpathy's val shard and the v1 architecture; absolute bpb differs",
                           "#420 v1 FLOPs are our estimate (v1 tokens x nanochat estimate_flops of the v1 arch)",
                           "#420 hours are 8xH100, ours 8xH200 training time (startup/eval excluded)",
                           "single seed per point; loop noise: sigma 0.0003 bpb at 1e18 d12, 0.0013 at 3e18 d16"]),
        arms={"plain": dict(label=ARM_LABEL["plain"], residual={"type": "plain"}, optimizer_overrides={}),
              "gated": dict(label=ARM_LABEL["gated"],
                            residual={"type": "gated", "streams": 4, "low_rank": 8, "read_scale": 2.0},
                            optimizer_overrides={"connection_matrix_optimizer": "muon"})},
        reference_420=dict(isoflop_optima=REF_ISO_420,
                           miniseries_v1=[dict(depth=d, model_dim=md, params_all=pm * 1e6, tokens=tb * 1e9,
                                               val_bpb=vb, core=vc, train_minutes_8xH100=tm,
                                               flops_estimate=v1_counts(d)[1] * tb * 1e9)
                                          for d, md, pm, tb, vb, vc, tm in REF_MINI_V1],
                           gpt2_core=dict(GPT2_CORE), gpt3_core=dict(GPT3_CORE), core_fit=CORE_FIT_420),
        runs=points, isoflop_fits=fits, optimal_scaling_fits=opt, frontier=front, core_fits=cfits,
        same_depth_gains=gains, miniseries_multipliers=mm, figures=figs)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "ladder_tables.md").write_text(tables(points, fits, front, cfits, opt, gains, mm))
    n_done = sum(1 for p in points if p["final_val_bpb"])
    print(f"{len(points)} ladder points ({n_done} complete); figures: {figs}")


if __name__ == "__main__":
    main()
