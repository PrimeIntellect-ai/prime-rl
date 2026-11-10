import json
import sys


def load(path):
    steps = {}
    for line in open(path):
        record = json.loads(line)
        step = record.get("step")
        if step is None:
            continue
        steps.setdefault(step, {}).update(record)
    return steps


base = sys.argv[1]
for pair in sys.argv[2:]:
    a_name, b_name = pair.split(":")
    a = load(f"{base}/{a_name}/monitors/file/metrics.jsonl")
    b = load(f"{base}/{b_name}/monitors/file/metrics.jsonl")
    steps = sorted(set(a) & set(b))
    print(f"### {a_name} vs {b_name}: {len(steps)} steps")
    worst = {"loss/mean": 0.0, "optim/grad_norm": 0.0}
    for step in steps:
        rel = {k: abs(b[step][k] - a[step][k]) / abs(a[step][k]) for k in worst}
        for k in worst:
            worst[k] = max(worst[k], rel[k])
        print(
            f"step {step:2d} loss {a[step]['loss/mean']:.6f} {b[step]['loss/mean']:.6f} rel {rel['loss/mean']:.2e}"
            f" | gnorm {a[step]['optim/grad_norm']:.6f} {b[step]['optim/grad_norm']:.6f} rel {rel['optim/grad_norm']:.2e}"
        )
    print(f"max rel diff: loss {worst['loss/mean']:.2e}, grad_norm {worst['optim/grad_norm']:.2e}")
