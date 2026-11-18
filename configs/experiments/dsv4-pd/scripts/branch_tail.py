"""For same-length mismatches: old path length and what the new prompt appends. branch_tail.py <dir> <max files>"""
import collections
import glob
import json
import sys

lens, tails = [], collections.Counter()
for path in sorted(glob.glob(sys.argv[1] + "/*.jsonl"))[: int(sys.argv[2])]:
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
        if prev is not None and prev[-1] not in cur:
            old = sum(len(nodes[n].get("token_ids") or []) for n in prev)
            new = sum(len(nodes[n].get("token_ids") or []) for n in cur[:-1])
            if new > 0.5 * old:
                lens.append(old)
                msg = nodes[cur[-2]].get("message") or {}
                content = msg.get("content")
                text = content if isinstance(content, str) else json.dumps(content)[:200]
                tails[(msg.get("role"), (text or "")[:70])] += 1
        prev = cur
lens.sort()
print(len(lens), "same-length mismatches; old path p10/p50/p90:", lens[len(lens) // 10], lens[len(lens) // 2], lens[9 * len(lens) // 10])
print("old path >= 110K:", sum(x >= 110_000 for x in lens))
for (role, text), n in tails.most_common(6):
    print(n, role, repr(text))
