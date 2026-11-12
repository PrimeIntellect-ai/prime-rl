"""Plan the full #420 ladder for two arms (plain, gated s4r8 + Muon gates): params, FLOPs/token, steps, tokens.

python ladder_plan.py            # print the table
Used by ladder_launch.py; counts come from cpu_import (meta device, the branch's model code).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpu_import  # noqa: E402

B = 2**19
ARMS = {
    "plain": {"residual": {"type": "plain"}, "variant": "plain", "optim": []},
    "gated": {"residual": {"type": "gated", "streams": 4, "low_rank": 8, "read_scale": 2.0},
              "variant": "gated-s4r8-muon", "optim": ['connection_matrix_optimizer="muon"']},
}
ISO = {1e18: [8, 10, 12, 14, 16], 3e18: [10, 12, 14, 16, 18, 20], 6e18: [14, 16, 18, 20, 22]}
MINI = list(range(10, 21))
_cache = {}


def counts(depth, arm):
    key = (depth, arm)
    if key not in _cache:
        m = cpu_import.build(depth, ARMS[arm]["residual"])
        c = cpu_import.counts(m)
        # #420 v1 "all params": wte + lm_head + transformer matrices (incl. connection params) + scalars,
        # no value embeddings (v1 had none). This is the convention behind tokens = 8 x params (d12: 135.3M).
        c["all_v1"] = c["wte"] + c["lm_head"] + c["transformer_matrices"] + c["scalars"]
        _cache[key] = c
    return _cache[key]


if __name__ == "__main__":
    for arm in ARMS:
        for d in range(8, 25, 2) if False else range(8, 25):
            c = counts(d, arm)
            print(arm, d, c["total"], c["scaling"], c["all_v1"], c["flops_per_token"])
