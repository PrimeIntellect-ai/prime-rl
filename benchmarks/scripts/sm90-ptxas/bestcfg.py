import json
import os
import sys

from torch._inductor.runtime.autotune_cache import AutotuneCache

for f in sys.argv[1:]:
    p = AutotuneCache._make_local_cache_key(os.path.dirname(f), AutotuneCache._prepare_key(f))
    cfg = json.load(open(p)) if os.path.exists(p) else None
    print(f, json.dumps(cfg and {k: cfg[k] for k in ("XBLOCK", "R0_BLOCK", "num_warps") if k in cfg}))
