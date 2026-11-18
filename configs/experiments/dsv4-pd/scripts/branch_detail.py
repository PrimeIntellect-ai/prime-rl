"""Where do mismatched calls diverge from the agent's previous call? branch_detail.py <traces/live dir> [max files]"""
import collections
import glob
import json
import sys

files = sorted(glob.glob(sys.argv[1] + "/*.jsonl"))[: int(sys.argv[2]) if len(sys.argv) > 2 else None]
rows = []
for path in files:
    nodes, calls = [], []
    for line in open(path):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        nodes += rec.get("nodes", [])
        calls += rec.get("calls", [])

    def chain(n):
        out = []
        while n is not None:
            out.append(n)
            n = nodes[n].get("parent")
        return out[::-1]

    prev = None
    for call in calls:
        if call.get("error") or call.get("node") is None:
            continue
        cur = chain(call["node"])
        if prev is not None and prev[0][-1] not in cur:
            p, pcall = prev
            common = 0
            for a, b in zip(p, cur):
                if a != b:
                    break
                common += 1
            tok = lambda ns: sum(len(nodes[n].get("token_ids") or []) for n in ns)
            rows.append(dict(prev_finish=pcall.get("finish_reason"), prompt=call["usage"]["prompt_tokens"],
                             prev_path=tok(p), common=tok(p[:common]),
                             new_roles=[(nodes[n].get("message") or {}).get("role") for n in cur[common:]][:4],
                             dropped_roles=[(nodes[n].get("message") or {}).get("role") for n in p[common:]][:4]))
        prev = (cur, call)
print(len(rows), "mismatches")
print("previous finish_reason:", collections.Counter(r["prev_finish"] for r in rows).most_common())
print("kept share of previous path:", collections.Counter(round(r["common"] / max(r["prev_path"], 1), 1) for r in rows).most_common(12))
print("new prompt / previous path:", collections.Counter(round(r["prompt"] / max(r["prev_path"], 1), 1) for r in rows).most_common(8))
print("first dropped roles:", collections.Counter(tuple(r["dropped_roles"][:2]) for r in rows).most_common(6))
print("first new roles:", collections.Counter(tuple(r["new_roles"][:2]) for r in rows).most_common(6))
