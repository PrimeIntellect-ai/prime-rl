# FSDP-cached prepared operands

## Problem

During gradient accumulation the weights are constant, but low-precision ops re-derive their
quantized operands (qdata, scales, transposed layouts) from the high-precision weight on every
microbatch forward and backward, and again on activation-checkpoint recompute. We would like a
mechanism which allows us to perform the quantization a minimal number of times (likely trading
memory for speed).

More generally, the design should allow for avoiding generic repeated weight-setup steps, not solely
restricted to quantization use cases.

## Mechanism

FSDP2's tensor-subclass extension hooks turn the unshard/reshard lifecycle into a cache for
derived operands. Two wrapper subclasses, adapted from torchtitan's mxfp8 prototype, serve every
recipe: a weight's preparation is set by the `prepare` callable attached when the parameter is
wrapped, so a new recipe never needs its own subclass.

- `ShardedPreparedTensor(shard, prepare_fn)`: the persistent parameter, wrapping the sharded
  master weight. Implements `fsdp_pre_all_gather` (cast the shard to param_dtype, all-gather in
  high precision) and `fsdp_post_all_gather` (run `prepare_fn` on the gathered weight, hand the
  prepared tensors' storage to FSDP, release the high-precision gather buffer).
- `UnshardedPreparedTensor`: what the module's forward sees between unshard and reshard. Holds no
  high-precision storage, only the named tensors its `prepare_fn` produced, exposed as
  `.prepared`, a read-only name -> tensor mapping. This attribute is how ops access the prepared
  tensors.

There is no cache-invalidation API. Reshard frees the prepared tensors; the next unshard rebuilds them
from the current master shards, so optimizer updates (including offloaded or overlapped ones) are
picked up automatically.

## Design criteria

- Zero-cost when unused: an unwrapped weight takes today's code path unchanged, unit tests included
- General: any weight preprocessing fits, quantization is only the first use case
- Preparation constraint: each processing step can only consume a single parameter. An op may
  consume several individually prepared weights
- No `nn.Module` changes (class swaps, model rewrites): swap in at the op level. Caching turns on
  only when the parameters are wrapped in the tensor subclasses
- Ops are stateless: no parameters, buffers, or mutable state, only frozen config. The module owns
  the weights, so swapping one op for another never moves state
- No invalidation surface: correctness rides the FSDP unshard/reshard lifecycle alone
- Strict: mis-installation and out-of-scope calls raise

## API

```python
PrepareFn = Callable[[Tensor], dict[str, Tensor]]


class Op(Protocol):
    def prepare(self, weight: Tensor) -> dict[str, Tensor]: ...  # keys become the wrapper's flat schema
    def __call__(self, *args, **kwargs): ...  # tensors and scalars only, never modules


def install_prepared_weights(module: nn.Module, prepare_fns: Mapping[str, PrepareFn]) -> None: ...
```

- `prepare` must be a pure function of the weight and frozen config, and every value it returns
  must be a tensor: the values become FSDP-owned unsharded storage, gathered at unshard and freed
  at reshard.
- `install_prepared_weights` wraps the named parameters in place. The caller supplies the mapping
  from parameter name to prepare callable. A missing or already-wrapped parameter raises. Run before
  `fully_shard`.
- Ops prepare all their weights the same way, so the caller passes `op.prepare` for every
  parameter. This is intentional, since no current op needs per-weight preparation, but it may
  become a limitation. The mapping can already hold a different callable per parameter; we would
  only need to decide how such an op exposes them.
- An op detects preparation itself: `isinstance(weight, UnshardedPreparedTensor)`, then read
  `weight.prepared`. Unwrapped weights take today's code path, and an op may prepare only some of
  its weights. A `ShardedPreparedTensor` at compute time means the op ran outside its weights'
  unshard scope and errors.

## Worked example: (hypothetical) multi-weight fused-MoE op

```python
def blockwise_fp8_prepare(weight: Tensor, block_size: int) -> dict[str, Tensor]:
    # the reusable recipe: a plain function any op's prepare can call
    # *_t entries: transposed layouts for the dx GEMM in backward
    return {"qdata": ..., "scales": ..., "qdata_t": ..., "scales_t": ...}


class FusedFp8ExpertCompute:
    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def prepare(self, weight):
        return blockwise_fp8_prepare(weight, self.block_size)

    def __call__(self, x, gate_up, down, num_tokens_per_expert):
        gu = gate_up.prepared if isinstance(gate_up, UnshardedPreparedTensor) else None
        dn = down.prepared if isinstance(down, UnshardedPreparedTensor) else None
        return _fused_fp8_experts(x, gate_up, down, gu, dn, num_tokens_per_expert)


# binding site (moe_runtime), setup time, before fully_shard
op = FusedFp8ExpertCompute()
moe.experts.compute = op
install_prepared_weights(moe.experts, {"gate_up_proj": op.prepare, "down_proj": op.prepare})


# the module names its own parameters; the op never sees the module
class GroupedExperts(nn.Module):
    def forward(self, x, num_tokens_per_expert):
        return self.compute(x, self.gate_up_proj, self.down_proj, num_tokens_per_expert)
```

`_fused_fp8_experts` is a single entry point that dispatches on `gu is None`. The None branch is
today's code path unchanged, so the mechanism is zero-cost and invisible when nothing is wrapped,
unit tests included. On the prepared branch the kernel reads the names it was written against
(`gu["qdata"]`, `dn["scales_t"]`, ...); flattening them into tensor arguments for the underlying
custom op is an internal detail of that branch.

A single-weight fp8 linear op reuses `blockwise_fp8_prepare` unchanged and reads only the entries
its kernel needs. An op preferring typed attribute access or derived views (e.g. a transpose as a
property) may declare a private frozen dataclass beside its kernel and build it from the mapping;
that is the op's own style, invisible to the machinery.

## Rules

- All weights consumed by one op call must live in the same `fully_shard` unit, so they are all
  realized at the call site.
- Check the raw parameters for the subclass, before any dtype cast, `to_local`, or layout
  transform.
- On the prepared branch, backward saves the wrapper and re-reads `.prepared` from it, never the
  prepared tensors themselves, so a reshard-then-refill between forward and backward stays
  correct.
- Joint preparation (one `prepare` over several weights) is out of scope; pack the weights into
  one parameter instead, as the existing QKV and gate_up fusions do.

## Cache lifetime

Set by existing FSDP2 knobs, per `fully_shard` unit:

- default: prepared tensors live from unshard to reshard, covering one forward, its backward
  layouts, and any AC recompute in between
- `reshard_after_forward=False` plus `set_reshard_after_backward(False)` on non-final
  microbatches: prepared tensors live for the whole gradient-accumulation window, one `prepare`
  per optimizer step

## Pros

- Invalidation is free and always correct: the FSDP lifecycle is the cache lifecycle
- One `prepare` covers forward, backward layouts, and AC recompute; one per optimizer step in window mode
- Lowers unsharded memory: the bf16 gather buffer is released, only quantized bytes stay resident
- Zero-cost when unwrapped, and a new preparation is one `prepare` method

## Cons

- Dim-0 sharding only for now (an implementation limit, not fundamental): excludes
  `shard_fused_on_dim1` params and parts of the MoE path
- Window mode holds roughly 2 bytes/param of unsharded prepared tensors across the window and
  needs train-loop flag wiring
- Rides on private FSDP2 extension hooks and tensor-subclass machinery; torch.compile is the main risk
- All-gather stays high precision: no communication savings
- One homogeneous `prepare` per op is baked into the protocol; per-weight heterogeneous
  preparation stays expressible through the install mapping but has no prescribed structure
  (intentional, revisit when a real op needs it)
