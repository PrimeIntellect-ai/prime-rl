"""Per-level table of stress_driver.py runs, with each arm's knee and thrash onset.

    analyze_stress.py stress/<run>.jsonl [...]

Throughput is also given per GPU of the arm (16), so arms compare on equal compute: out = generated tokens,
prefill = prompt tokens actually computed, prompt = all prompt tokens served (computed + cache hits).
recompute = prefill tokens computed / prompt tokens (prompt tokens not served from GPU or external cache): it rises
when sessions' KV is evicted between turns and has to be prefilled again (thrash). The knee is the level with the
most generated tokens/s. Thrash onset is the first level whose recompute share is at least twice the lowest level's
and above 0.2, or whose generated tokens/s drop more than 10% below the best lower level.
"""

import json
import sys
from pathlib import Path

GPUS = 16  # every arm: 2 nodes x 8 B200


def levels(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    windows = [r for r in rows if r["kind"] == "window" and r.get("phase", "hold") == "hold"]
    out = []
    done = {r["level"] for r in rows if r["kind"] == "level"}
    live = [w for w in windows if w["level"] not in done]
    if live:  # the running level: mean of its windows after the first five minutes (or all of them before that)
        lvl = [w for w in live if w["level_min"] > 5] or live
        roles = {}
        for role in lvl[0]["roles"]:
            keys = {k for w in lvl for k, v in w["roles"][role].items() if v is not None}
            roles[role] = {
                k: sum(w["roles"][role][k] for w in lvl if w["roles"][role].get(k) is not None)
                / max(1, sum(w["roles"][role].get(k) is not None for w in lvl))
                for k in keys
            }
        rows.append({"kind": "level", "level": lvl[0]["level"], "roles": roles, "live_min": lvl[-1]["level_min"]})
    for r in rows:
        if r["kind"] != "level":
            continue
        a = r["roles"]["all"]
        lvl = [w for w in windows if w["level"] == r["level"]]
        half = lvl[len(lvl) // 2 :] if "live_min" not in r else [w for w in lvl if w["level_min"] > 5] or lvl
        inflight = [w["inflight"] for w in half if w.get("inflight") is not None]
        roles = r["roles"]
        hit_role = roles.get("prefill") or roles.get("agg") or a
        out.append(
            {
                "level": r["level"] if "live_min" not in r else f"{r['level']} (live, {r['live_min']:.0f} min)",
                # P/D counts a prompt on both roles (decode receives it over NIXL) and prefill emits one token per
                # call, so served prompt tokens come from prefill and generated tokens from decode.
                "prompt": hit_role.get("prompt_tok_s", 0.0),
                "inflight": sum(inflight) / len(inflight) if inflight else None,
                "gen": (roles.get("decode") or a).get("gen_tok_s", 0.0),
                "calls": a.get("req_done_s", 0.0) / (2 if "decode" in roles else 1),
                "prefill": a.get("prefill_computed_tok_s", 0.0),
                "recompute": a.get("prefill_computed_tok_s", 0.0) / hit_role["prompt_tok_s"]
                if hit_role.get("prompt_tok_s")
                else None,
                "gpu_hit": hit_role.get("gpu_hit_rate"),
                "ext_hit": (roles.get("prefill") or {}).get("ext_hit_rate"),
                "waiting": a.get("waiting", 0.0),
                "running": a.get("running", 0.0),
                "preempt": a.get("preempt_s", 0.0),
                "itl": (roles.get("decode") or a).get("itl_mean"),
                "ttft": a.get("ttft_mean"),
                "e2e": (roles.get("decode") or a).get("e2e_mean"),
            }
        )
    return out


def fmt(v, spec):
    return format(v, spec) if v is not None else "-"


def main() -> None:
    for path in map(Path, sys.argv[1:]):
        rows = levels(path)
        if not rows:
            print(f"{path.stem}: no finished levels yet")
            continue
        print(f"\n## {path.stem}")
        print("| level | inflight | out tok/s/GPU | prefill tok/s/GPU | prompt tok/s/GPU | calls/s | recompute | GPU hit | ext hit | running | waiting | preempt/s | ITL s | TTFT s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(
                f"| {r['level']} | {fmt(r['inflight'], '.0f')} | {r['gen'] / GPUS:.0f} | {r['prefill'] / GPUS:.0f} | "
                f"{r['prompt'] / GPUS:.0f} | {r['calls']:.1f} | {fmt(r['recompute'], '.3f')} | {fmt(r['gpu_hit'], '.3f')} | "
                f"{fmt(r['ext_hit'], '.3f')} | {r['running']:.0f} | {r['waiting']:.0f} | {r['preempt']:.2f} | "
                f"{fmt(r['itl'], '.3f')} | {fmt(r['ttft'], '.2f')} |"
            )
        rows = [r for r in rows if isinstance(r["level"], int)] or rows
        knee = max(rows, key=lambda r: r["gen"])
        base = min((r["recompute"] for r in rows if r["recompute"] is not None), default=None)
        onset, best = None, 0.0
        for r in rows:
            thrash = r["recompute"] is not None and base is not None and r["recompute"] >= max(2 * base, 0.2)
            if thrash or (best and r["gen"] < 0.9 * best):
                onset = r["level"]
                break
            best = max(best, r["gen"])
        print(f"knee: {knee['level']} ({knee['gen'] / GPUS:.0f} out tok/s/GPU); thrash onset: {onset or 'not reached'}")


if __name__ == "__main__":
    main()
