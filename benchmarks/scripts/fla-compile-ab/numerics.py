"""Per step, the relative loss and grad norm difference of RUN_DIR against BASE_DIR. Usage: python numerics.py BASE_DIR RUN_DIR."""

import glob
import json
import sys


def load(run):
    rows = {}
    for p in glob.glob(f"{run}/out/**/metrics.jsonl", recursive=True):
        for l in open(p):
            r = json.loads(l)
            rows.setdefault(r["step"], {}).update(r)
    return {s: (r["loss/mean"], r["optim/grad_norm"]) for s, r in rows.items()}


b, o = load(sys.argv[1]), load(sys.argv[2])
print(
    " ".join(
        f"{s}:{abs(o[s][0] - b[s][0]) / abs(b[s][0]):.1e}/{abs(o[s][1] - b[s][1]) / abs(b[s][1]):.1e}"
        for s in sorted(set(b) & set(o))
    )
)
