"""Differential scan of Inductor kernel files (TORCHINDUCTOR_BENCHMARK_KERNEL=1) across ptxas arms.

drive <arm> <listfile> <outdir> <shard> <nshards> [dumpfile]
work  <arm> <listfile> <outdir> <shard> <nshards> [dumpfile]

arm: p129 (bundled 12.9), p133 (bundled ptxas-blackwell 13.3), noslp (12.9 with LLVM SLP off). The arm's ptxas is
chosen by TRITON_PTXAS_PATH set by the caller; noslp is applied here.
Each (kernel, config) runs twice on identical seeded inputs; the storage digest of every tensor arg is recorded.
dumpfile: optional json list of [file, config_key]; for those, every tensor arg after the run is saved.
"""

import hashlib
import importlib.util
import json
import re
import os
import subprocess
import sys
import time
from pathlib import Path


def config_key(launcher):
    c = launcher.config
    return json.dumps({"kw": c.kwargs, "w": c.num_warps, "s": c.num_stages}, sort_keys=True)


def storage_bytes(t):
    import torch

    s = t.untyped_storage()
    u8 = torch.empty(0, dtype=torch.uint8, device=t.device)
    u8.set_(s, 0, (s.nbytes(),))
    return u8


def digest(args):
    import torch

    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(hashlib.blake2b(storage_bytes(a).cpu().numpy().tobytes(), digest_size=12).hexdigest())
    return out


def work(arm, files, outdir, shard, dump):
    import torch

    if arm == "noslp":
        import triton.backends.nvidia.compiler as nvc

        orig = nvc.llvm.optimize_module

        def optimize_module_noslp(*a, **k):
            k["disable_slp_vectorizer"] = True
            return orig(*a, **k)

        nvc.llvm.optimize_module = optimize_module_noslp
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    stream = torch.cuda.current_stream().cuda_stream
    res = open(outdir / f"res-{arm}-{shard}.jsonl", "a")
    prog = outdir / f"prog-{arm}-{shard}.txt"
    dumpdir = outdir / f"dump-{arm}"
    srcdir = outdir / f"src-{arm}-{shard}"
    srcdir.mkdir(exist_ok=True)
    for f in files:
        prog.write_text(f)
        rec = {"file": f}
        t0 = time.time()
        try:
            src = re.sub(r"cuda:\d+", "cuda:0", open(f).read())
            src = re.sub(r"(DeviceProperties\(.*?index=)\d+", r"\g<1>0", src)
            local = srcdir / Path(f).name
            local.write_text(src)
            spec = importlib.util.spec_from_file_location("k_" + Path(f).stem, local)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            kerns = [v for v in vars(mod).values() if isinstance(v, CachingAutotuner)]
            kern = kerns[0]
            rec["name"] = kern.inductor_meta.get("kernel_name") or kern.fn.__name__
            kern.precompile()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            res.write(json.dumps(rec) + "\n")
            res.flush()
            continue
        rec["configs"] = {}
        all_launchers = list(kern.launchers)
        for launcher in all_launchers:
            kern.launchers = [launcher]
            kern._cached_launcher = None
            key = config_key(launcher)
            ds = []
            for _ in range(2):
                torch.manual_seed(0)
                args = mod.get_args()
                kern.run(*args, stream=stream)
                torch.cuda.synchronize()
                ds.append(digest(args))
            rec["configs"][key] = {"d": ds[0], "det": ds[0] == ds[1], "spills": getattr(launcher, "n_spills", None)}
            if os.environ.get("DIFFSCAN_BENCH"):
                import triton.testing

                rec["configs"][key]["ms"] = triton.testing.do_bench(lambda: kern.run(*args, stream=stream), warmup=10, rep=50)
            if [f, key] in dump:
                dumpdir.mkdir(exist_ok=True)
                torch.manual_seed(0)
                args = mod.get_args()
                kern.run(*args, stream=stream)
                torch.cuda.synchronize()
                tens = [storage_bytes(a).cpu().clone() for a in args if isinstance(a, torch.Tensor)]
                meta = [(str(a.dtype), list(a.shape), list(a.stride()), a.storage_offset()) for a in args if isinstance(a, torch.Tensor)]
                torch.save({"bytes": tens, "meta": meta}, dumpdir / f"{Path(f).stem}__{hashlib.md5(key.encode()).hexdigest()[:8]}.pt")
        rec["secs"] = round(time.time() - t0, 2)
        res.write(json.dumps(rec) + "\n")
        res.flush()


def main():
    mode, arm, listfile, outdir, shard, nshards = sys.argv[1:7]
    dumpfile = sys.argv[7] if len(sys.argv) > 7 else None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shard, nshards = int(shard), int(nshards)
    files = [l.strip() for l in open(listfile) if l.strip()]
    files = files[shard::nshards]
    dump = json.load(open(dumpfile)) if dumpfile else []
    if mode == "work":
        skipf = os.environ.get("DIFFSCAN_SKIP_FILE_LIST")
        skip = set(json.load(open(skipf))) if skipf else set()
        work(arm, [f for f in files if f not in skip], outdir, shard, dump)
        return
    resf = outdir / f"res-{arm}-{shard}.jsonl"
    prog = outdir / f"prog-{arm}-{shard}.txt"
    skipf = outdir / f"skip-{arm}-{shard}.json"
    while True:
        done = set()
        if resf.exists():
            done = {json.loads(l)["file"] for l in open(resf)}
        if done >= set(files):
            break
        skipf.write_text(json.dumps(sorted(done)))
        env = dict(os.environ, DIFFSCAN_SKIP_FILE_LIST=str(skipf))
        if prog.exists():
            prog.unlink()
        rc = subprocess.run([sys.executable, __file__, "work", *sys.argv[2:]], env=env).returncode
        if rc != 0:
            bad = prog.read_text() if prog.exists() else None
            if bad is None or bad in done:
                print(f"worker rc={rc} with no progress, stopping", flush=True)
                break
            with open(resf, "a") as r:
                r.write(json.dumps({"file": bad, "crash": rc}) + "\n")


if __name__ == "__main__":
    main()
