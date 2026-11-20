import collections
import json
import re
import sys

out = sys.argv[1]
res = {}
for a in ["p129", "p133", "noslp"]:
    for fn in __import__("glob").glob(f"{out}/res-{a}-*.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            res.setdefault(r["file"], {})[a] = r
rows = []
for f, arms in res.items():
    src = open(f).read() if __import__("os").path.exists(f) else None
    sizes = re.findall(r"rand_strided\(\((.*?)\)", src) if src else []
    shapes = [tuple(int(x) for x in s.split(",") if x.strip()) for s in sizes]
    q, k = shapes[0], shapes[1]
    key = (q[0], k[0], q[2], q[1])
    p, o = arms["p129"], arms["p133"]
    status = {}
    if "crash" in p:
        status = "crash(p129 only)" if "crash" not in o else "crash(all)"
    else:
        for cfg, c in p["configs"].items():
            c3 = o["configs"].get(cfg)
            kw = json.loads(cfg)
            name = f"X{kw['kw'].get('XBLOCK')}/w{kw['w']}"
            status[name] = "ok" if c["det"] and c3["det"] and c["d"] == c3["d"] else ("BAD" if c3["det"] else "nondet133")
    rows.append((key, status))
for key, status in sorted(rows):
    print("Hq=%d Hkv=%d D=%d S=%d" % key, status)
