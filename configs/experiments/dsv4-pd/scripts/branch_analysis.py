"""Per-call prefix continuity from eval trace streams: does each call's prompt extend the same agent session's
previous call (prompt + completion)? A call that does not is a router prefix mismatch under bidirectional KV.

    branch_analysis.py <traces/live dir> [max files]
"""
import collections
import glob
import json
import sys

files = sorted(glob.glob(sys.argv[1] + "/*.jsonl"))[: int(sys.argv[2]) if len(sys.argv) > 2 else None]
by_min = collections.defaultdict(lambda: [0, 0])
kinds = collections.Counter()
mismatch_ctx = []
t0 = None
for path in files:
    parent, last_node, calls = {}, {}, []
    for line in open(path):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for node in rec.get("nodes", []):
            parent[len(parent)] = node.get("parent")
        for call in rec.get("calls", []):
            calls.append(call)
    for call in calls:
        if call.get("error") or call.get("node") is None:
            continue
        sess = (call.get("acp") or {}).get("agent_session_id") or "root"
        start = call["time"]["start"] if isinstance(call.get("time"), dict) and "start" in call["time"] else None
        ancestors, n = set(), parent.get(call["node"])
        while n is not None and n not in ancestors:
            ancestors.add(n)
            n = parent.get(n)
        prev = last_node.get(sess)
        last_node[sess] = call["node"]
        if prev is None:
            kinds["first call of agent session"] += 1
            continue
        ok = prev in ancestors
        kinds["extends previous" if ok else "MISMATCH"] += 1
        if start:
            t0 = min(t0 or start, start)
            by_min[start] = by_min.get(start, [0, 0])
        if not ok:
            mismatch_ctx.append((call["usage"]["prompt_tokens"], call.get("finish_reason"), sess != "root", start))
        if start:
            by_min[start][0 if ok else 1] += 1
print(len(files), "traces;", dict(kinds))
buckets = collections.defaultdict(lambda: [0, 0])
for start, (ok, bad) in by_min.items():
    b = int((start - t0) // 600) * 10
    buckets[b][0] += ok
    buckets[b][1] += bad
for b in sorted(buckets):
    ok, bad = buckets[b]
    print(f"min {b:3d}-{b + 10:3d}: {ok + bad:6d} follow-up calls, mismatch {bad / max(ok + bad, 1):.3f}")
sub = sum(1 for m in mismatch_ctx if m[2])
print("mismatches in sub-agent sessions:", sub, "of", len(mismatch_ctx))
ctx = sorted(m[0] for m in mismatch_ctx)
if ctx:
    print("mismatch prompt tokens: p10 %d p50 %d p90 %d" % (ctx[len(ctx) // 10], ctx[len(ctx) // 2], ctx[9 * len(ctx) // 10]))
