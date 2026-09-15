# FSDP-cached prepared operands

## Problem

During gradient accumulation the weights are constant, but low-precision ops re-derive their
quantized operands (qdata, scales, transposed layouts) from the high-precision weight on every
microbatch forward and backward, and again on activation-checkpoint recompute.

## Mechanism

FSDP2's tensor-subclass extension hooks turn the unshard/reshard lifecycle into a cache for
derived operands. Two wrapper subclasses, adapted from torchtitan's mxfp8 prototype:

- `ShardedPreparedTensor(op)`: the persistent parameter, wrapping the sharded master weight.
  Implements `fsdp_pre_all_gather` (cast the shard to param_dtype, all-gather in high precision)
  and `fsdp_post_all_gather` (call `op.prepare` on the gathered weight, hand the operand storage
  to FSDP, release the high-precision gather buffer). Generic, parameterized by the op.
- `UnshardedPreparedTensor`: what the module's forward sees between unshard and reshard. Holds no
  high-precision storage, only `.operands`. Generic, never subclassed.

An op plugs in by providing two things:

```python
@dataclass(frozen=True)
class MyOperands:
    qdata: torch.Tensor        # each field a distinct allocation, storage owned by FSDP
    scale_fwd: torch.Tensor
    scale_dgrad: torch.Tensor  # derived views (e.g. transposes) are @property, not fields


class MyOp:
    operands_cls = MyOperands

    def prepare(self, weight: torch.Tensor) -> MyOperands: ...

    def __call__(self, x, weight, ...):
        if isinstance(weight, UnshardedPreparedTensor):
            operands = weight.operands       # cache hit, no quantization kernels
        else:
            operands = self.prepare(weight)  # non-FSDP fallback
        ...
```

There is no cache-invalidation API. Reshard frees the operands; the next unshard rebuilds them
from the current master shards, so optimizer updates (including offloaded or overlapped ones) are
picked up automatically.

Cache lifetime is set by existing FSDP2 knobs, per `fully_shard` unit:

- default: operands live from unshard to reshard, covering one forward, its backward layouts, and
  any AC recompute in between
- `reshard_after_forward=False` plus `set_reshard_after_backward(False)` on non-final
  microbatches: operands live for the whole gradient-accumulation window, one `prepare` per
  optimizer step

## Pros

- Invalidation is free and always correct: the FSDP lifecycle is the cache lifecycle
- One `prepare` covers forward, backward layouts, and AC recompute; one per optimizer step in window mode
- Lowers unsharded memory: the bf16 gather buffer is released, only quantized bytes stay resident
- Adding an op is small: one frozen dataclass plus one `prepare` method

## Cons

- Dim-0 sharding only for now (an implementation limit, not fundamental): excludes
  `shard_fused_on_dim1` params and parts of the MoE path
- Window mode holds roughly 2 bytes/param of unsharded operands across the window and needs
  train-loop flag wiring
- Rides on private FSDP2 extension hooks and tensor-subclass machinery; torch.compile is the main risk
- All-gather stays high precision: no communication savings
