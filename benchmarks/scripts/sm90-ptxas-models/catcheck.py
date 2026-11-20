import glob, json, sys

out = sys.argv[1]
for arm in ("p129", "p133"):
    for fn in sorted(glob.glob(f"{out}/res-{arm}-*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            if "crash" in r or "error" in r:
                print(arm, "CRASH/ERROR", r.get("name"), r.get("crash"), r.get("error"), r["file"].split("/")[-1][:12])
                continue
            if "cat" not in r["name"] or ("select_backward" not in r["name"] and "split_with_sizes" not in r["name"]):
                continue
            digests = {k: tuple(c["d"]) for k, c in r["configs"].items()}
            dets = {k: c["det"] for k, c in r["configs"].items()}
            agree = len(set(digests.values())) == 1
            print(arm, r["name"], r["file"].split("/")[-1][:12], "configs_agree" if agree else "CONFIGS_DISAGREE", {json.loads(k)["kw"].get("XBLOCK"): (json.loads(k)["w"], d) for k, d in dets.items()})
