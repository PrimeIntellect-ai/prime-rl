"""Decode the token divergence of same-length mismatches: branch_diff.py <traces/live dir> <max files> <examples>"""
import glob
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("PrimeIntellect/DeepSeek-V4-Flash-0731-bf16")
shown = 0
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
            old = [t for n in prev for t in nodes[n].get("token_ids") or []]
            new = [t for n in cur[:-1] for t in nodes[n].get("token_ids") or []]
            if len(new) > 0.5 * len(old):
                i = next((k for k, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
                print(f"=== {path.split('/')[-1][:8]} old path {len(old)} new prompt {len(new)} diverge at {i}")
                print("OLD:", repr(tok.decode(old[max(0, i - 40): i + 60])))
                print("NEW:", repr(tok.decode(new[max(0, i - 40): i + 60])))
                shown += 1
                if shown >= int(sys.argv[3]):
                    sys.exit()
        prev = cur
