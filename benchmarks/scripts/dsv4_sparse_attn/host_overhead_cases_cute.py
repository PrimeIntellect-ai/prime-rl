"""bench_ops.py cases splitting the CuTe forwards' per-call host cost into its layers.

The CuTe counterpart of `host_overhead_cases.py`, run the same way (the profiling skill's `scripts/bench_ops.py`,
`DSV4_HOST_ITEM` picks the corpus item, default `tiny-4096-hca-cp8r4`). `DSV4_HOST_BACKEND` picks `cute` or
`cute_ws` (default). `DSV4_HOST_QUERIES=N` keeps only the item's first N queries, so that the kernel is short
enough for do_bench's back-to-back times to read as host time per call. Each label peels one layer off the full
op:

- `op`: the custom op under `no_grad`, as the benchmark's forward times it.
- `op_body`: the backend's forward function called directly, without the dispatcher, autograd and dynamo-disable
  wrappers.
- `wrapper_prep`: the slot padding, `num_tiles_covering_valid_slots` and float32 sinks the body prepares.
- `fwd_call`: the CuTe module's entry point on prepared arguments: shape checks, output allocation and the call.
- `executor_call`: the compiled TVM-FFI executor alone on prepared arguments and preallocated outputs.
- `empty_floor`: two `torch.empty` calls the size of the outputs.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_CORPUS_DIR, SM_SCALE, load_indices, load_manifest, make_inputs, select_items  # noqa: E402

from prime_rl.trainer.models.kernels.deepseek_v4 import (  # noqa: E402
    dsv4_sparse_attn_fwd_cute,
    dsv4_sparse_attn_fwd_cute_ws,
)
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import (  # noqa: E402
    FORWARD_BACKENDS,
    SLOT_TILE,
    _pad_slots_to_tile,
    dsv4_sparse_attn,
    num_tiles_covering_valid_slots,
)

LOG2E = 1.44269504


def cases():
    item_id = os.environ.get("DSV4_HOST_ITEM", "tiny-4096-hca-cp8r4")
    backend = os.environ.get("DSV4_HOST_BACKEND", "cute_ws")
    module, slot_multiple = {
        "cute": (dsv4_sparse_attn_fwd_cute, SLOT_TILE),
        "cute_ws": (dsv4_sparse_attn_fwd_cute_ws, dsv4_sparse_attn_fwd_cute_ws.PAIR),
    }[backend]
    entry = getattr(module, f"dsv4_sparse_attn_fwd_{backend}")
    (item,) = select_items(load_manifest(DEFAULT_CORPUS_DIR), [item_id])
    n_queries = int(os.environ.get("DSV4_HOST_QUERIES", item.n_queries))
    indices = load_indices(DEFAULT_CORPUS_DIR, item)[:, :n_queries].contiguous()
    inputs = make_inputs(item, requires_grad=False)
    q, kv, sinks = inputs["q"][:, :n_queries].contiguous(), inputs["kv"], inputs["sinks"]

    def prep():
        padded = _pad_slots_to_tile(indices, slot_multiple)
        return padded, num_tiles_covering_valid_slots(padded, SLOT_TILE), sinks.float().contiguous()

    padded, tile_counts, sinks_f32 = prep()
    executor = module._compiled_fwd()
    out = torch.empty_like(q)
    lse = q.new_empty(q.shape[:-1], dtype=torch.float32)
    scale_args = (SM_SCALE, SM_SCALE * LOG2E) if backend == "cute" else (SM_SCALE * LOG2E,)

    def op():
        with torch.no_grad():
            dsv4_sparse_attn(q, kv, indices, sinks, SM_SCALE, backend=backend)

    def empty_floor():
        torch.empty_like(q)
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)

    return [
        (
            f"{item.id}[:{n_queries}] {backend}",
            {
                "op": op,
                "op_body": lambda: FORWARD_BACKENDS[backend](q, kv, indices, sinks, SM_SCALE, SLOT_TILE, 2, 256),
                "wrapper_prep": prep,
                "fwd_call": lambda: entry(q, kv, padded, sinks_f32, tile_counts, SM_SCALE),
                "executor_call": lambda: executor(q, kv, padded, sinks_f32, tile_counts, out, lse, *scale_args),
                "empty_floor": empty_floor,
            },
        )
    ]
