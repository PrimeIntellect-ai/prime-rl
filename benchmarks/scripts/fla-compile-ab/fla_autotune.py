"""Per rank, the NB autotune keys of FLA's l2norm and gated-norm kernels and the seconds each took, from TRITON_PRINT_AUTOTUNING=1 output. Usage: python fla_autotune.py RUN_DIR."""

import glob
import re
import sys
from collections import defaultdict

run = sys.argv[1]
for path in sorted(
    glob.glob(f"{run}/torchrun/*/attempt_0/*/stdout.log") + glob.glob(f"{run}/torchrun/*/attempt_0/*/stderr.log")
):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    rank = path.split("/")[-2]
    tunes = re.findall(
        r"Triton autotuning for function (\w+),\s*with key as \((\d+), (\d+)[^\n]*\),\s*finished after ([\d.]+)s", text
    )
    tunes = [
        (f, int(n), float(s))
        for f, d, n, s in tunes
        if f in ("l2norm_fwd_kernel", "l2norm_bwd_kernel", "layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel")
    ]
    if not tunes:
        continue
    per_nb = defaultdict(float)
    for f, nb, s in tunes:
        per_nb[nb] += s
    print(
        f"rank {rank} ({path.split('/')[-1]}): NB values tuned in order {list(dict.fromkeys(nb for _, nb, _ in tunes))}, seconds per NB {dict((k, round(v, 1)) for k, v in per_nb.items())}"
    )
