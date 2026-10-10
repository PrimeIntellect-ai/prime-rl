"""bench_ops.py cases splitting the TileLang forward's per-call host cost into its layers.

Run with the profiling skill's `scripts/bench_ops.py` from the repo root; `DSV4_HOST_ITEM` picks the corpus item
(default `tiny-4096-hca-cp8r4`, whose 85 µs of kernel time leaves every case host-bound, so do_bench's
back-to-back times read as host time per call). Each label peels one layer off the full op:

- `op`: the custom op under `no_grad`, as the benchmark's forward times it.
- `op_body`: `_tilelang_forward` called directly, without the dispatcher, autograd and dynamo-disable wrappers.
- `wrapper_prep`: the slot padding, tile view and `num_tiles_covering_valid_slots` the body runs before the kernel.
- `kernel_lookup`: the `tilelang.jit` call that returns the already compiled kernel.
- `kernel_call`: the compiled kernel on prepared arguments: TileLang's argument handling, output allocation and launch.
- `empty_floor`: two `torch.empty` calls the size of the outputs, the allocation floor inside `kernel_call`.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_CORPUS_DIR, SM_SCALE, load_indices, load_manifest, make_inputs, select_items  # noqa: E402

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import (  # noqa: E402
    _pad_slots_to_tile,
    _tilelang_forward,
    dsv4_sparse_attn,
    num_tiles_covering_valid_slots,
)
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd import dsv4_sparse_attn_fwd  # noqa: E402

BLOCK_I, NUM_STAGES, THREADS = 64, 2, 256


def cases():
    item_id = os.environ.get("DSV4_HOST_ITEM", "tiny-4096-hca-cp8r4")
    (item,) = select_items(load_manifest(DEFAULT_CORPUS_DIR), [item_id])
    indices = load_indices(DEFAULT_CORPUS_DIR, item)
    inputs = make_inputs(item, requires_grad=False)
    q, kv, sinks = inputs["q"], inputs["kv"], inputs["sinks"]
    batch, seq_len, heads, dim = q.shape

    def compiled_kernel():
        return dsv4_sparse_attn_fwd(
            heads, dim, 1, SM_SCALE, True, block_I=BLOCK_I, num_stages=NUM_STAGES, threads=THREADS
        )

    def prep():
        padded = _pad_slots_to_tile(indices)
        return padded.view(batch, seq_len, 1, -1, BLOCK_I), num_tiles_covering_valid_slots(padded, BLOCK_I)

    kernel = compiled_kernel()
    tiled, tile_counts = prep()
    sinks_f32 = sinks.float().contiguous()

    def op():
        with torch.no_grad():
            dsv4_sparse_attn(q, kv, indices, sinks, SM_SCALE)

    def empty_floor():
        torch.empty_like(q)
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)

    return [
        (
            item.id,
            {
                "op": op,
                "op_body": lambda: _tilelang_forward(q, kv, indices, sinks, SM_SCALE, BLOCK_I, NUM_STAGES, THREADS),
                "wrapper_prep": prep,
                "kernel_lookup": compiled_kernel,
                "kernel_call": lambda: kernel(q, kv, tiled, sinks_f32, tile_counts),
                "empty_floor": empty_floor,
            },
        )
    ]
