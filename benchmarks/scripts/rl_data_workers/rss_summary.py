import sys, collections
for path in sys.argv[1:]:
    by = collections.defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[2].isdigit():
            step = int(p[3]) // 4
            by[p[1]][step] = int(p[2]) / 1024**2
    print(path.split("/")[-1])
    for pid, steps in sorted(by.items()):
        ks = sorted(steps)
        pick = [k for k in ks if k in (2, 5, 10, 15, 20, 25, 29, 30)] or ks
        print(f"  rank pid {pid}: " + "  ".join(f"step {k}: {steps[k]:.2f}" for k in pick) + " GiB")
