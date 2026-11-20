import glob, os, sys
import torch
out = sys.argv[1]
for f in sorted(glob.glob(f'{out}/dump-p129/*.pt')):
    b = os.path.basename(f)
    a = torch.load(f); o = torch.load(f.replace('dump-p129', 'dump-p133'))
    n = torch.load(f.replace('dump-p129', 'dump-noslp')) if os.path.exists(f.replace('dump-p129', 'dump-noslp')) else None
    for i, (x, y, m) in enumerate(zip(a['bytes'], o['bytes'], a['meta'])):
        if torch.equal(x, y): continue
        dt = getattr(torch, m[0].split('.')[1])
        xs = x.view(dt); ys = y.view(dt)
        wrong = (xs != ys) & ~(xs.isnan() & ys.isnan()) if dt.is_floating_point else (xs != ys)
        diff = (xs.float() - ys.float()).abs()[wrong]
        nslp = '' if n is None else f" noslp==133: {torch.equal(n['bytes'][i], y)}"
        print(f"{b} arg{i} {m[0]} shape={m[1]} wrong={int(wrong.sum())}/{xs.numel()} maxabs={float(diff.max()) if diff.numel() else 0:.3g}{nslp}")
