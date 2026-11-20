import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

arm, outdir = sys.argv[1], Path(sys.argv[2])
files = sys.argv[3:]
refmod_spec = importlib.util.spec_from_file_location("refs", Path(__file__).with_name("refs.py"))
refs = importlib.util.module_from_spec(refmod_spec)
refmod_spec.loader.exec_module(refs)
outdir.mkdir(parents=True, exist_ok=True)
stream = torch.cuda.current_stream().cuda_stream
for f in files:
    src = re.sub(r"cuda:\d+", "cuda:0", open(f).read())
    src = re.sub(r"(DeviceProperties\(.*?index=)\d+", r"\g<1>0", src)
    local = outdir / Path(f).name
    local.write_text(src)
    spec = importlib.util.spec_from_file_location("k_" + Path(f).stem, local)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kern = [v for v in vars(mod).values() if isinstance(v, CachingAutotuner)][0]
    name = kern.inductor_meta.get("kernel_name") or kern.fn.__name__
    kern.precompile()
    for launcher in list(kern.launchers):
        kern.launchers = [launcher]
        kern._cached_launcher = None
        c = launcher.config
        for rep in range(2):
            torch.manual_seed(0)
            args = mod.get_args()
            inputs = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
            out_idx, ref = refs.reference(name, inputs)
            kern.run(*args, stream=stream)
            torch.cuda.synchronize()
            out = args[out_idx]
            wrong = (out != ref).sum().item()
            maxabs = (out.float() - ref.float()).abs().max().item()
            print(json.dumps({"arm": arm, "file": Path(f).stem[:12], "name": name, "cfg": {**c.kwargs, "w": c.num_warps, "s": c.num_stages}, "rep": rep, "out_shape": list(out.shape), "wrong": wrong, "numel": out.numel(), "maxabs": maxabs}), flush=True)
