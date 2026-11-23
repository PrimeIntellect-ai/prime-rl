"""Print per-source eval summary from a run's monitors/file/metrics.jsonl."""
import json, sys
rows = {}
for line in open(sys.argv[1]):
    if '"eval/' in line:
        rows.update({k: v for k, v in json.loads(line).items() if k.startswith("eval/")})
srcs = sorted({k.split("/")[1] for k in rows})
def g(s, m, scope="all", stat="mean"):
    return rows.get(f"eval/{s}/{scope}/agent/{m}" + (f"/{stat}" if stat else ""))
def f(x):
    return "-" if x is None else (f"{x:.2f}" if abs(x) < 10 else f"{x:,.0f}")
cols = [("reward", "reward"), ("pass@4", ("pass@4", "effective", None)), ("has_sub", "metrics/has_sub_rlm"),
        ("sub calls", "metrics/sub_rlm_num_calls"), ("timeout", "is_timeout"), ("err", "has_error"),
        ("out tok", "num_output_tokens"), ("turns", "num_turns"), ("wall s", "timing/total")]
print(" | ".join(["source"] + [c for c, _ in cols]))
for s in srcs:
    print(" | ".join([s] + [f(g(s, *m) if isinstance(m, tuple) else g(s, m)) for _, m in cols]))
