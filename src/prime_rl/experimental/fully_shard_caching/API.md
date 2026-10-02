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
recipe: a weight's preparation is set by the op attached when the parameter is wrapped, so a new
recipe never needs its own subclass.

Preparation happens at one of two points, and the op's type decides which.

- **After the all-gather.** The collective carries the high-precision weight and every rank prepares
  the whole gathered tensor. An op that implements `prepare` alone takes this path.
- **Before the all-gather.** Each rank prepares its own shard, the collective carries what that
  produced, and a completion step derives anything that can only be built once the shards are
  joined. An op that also implements `prepare_shard` and `complete_gathered` takes this path. The
  collective then moves prepared bytes rather than high-precision ones, and the preparation each
  rank performs shrinks with the number of shards.

The second path is not available to every recipe, because preparing a shard in isolation has to
agree with preparing the whole weight. `shard_blocking` is how an op declares when that holds; see
`## API`.

- `ShardedPreparedTensor(shard, op)`: the persistent parameter, wrapping the sharded master weight.
  Implements `fsdp_pre_all_gather` and `fsdp_post_all_gather`, and routes each to the op's matching
  method.
- `UnshardedPreparedTensor`: what the module's forward sees between unshard and reshard. Holds no
  high-precision storage, only the named tensors the op produced. An op reads each one as an
  attribute, `prepared_<name>`. The `.prepared` mapping, a read-only name to tensor view of the same
  set, is for code that has to iterate all of them, such as checkpointing.

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
Prepared = dict[str, Tensor]
Out = Prepared | None

# Per weight axis, the extent over which the recipe shares a scale.
# None means the whole axis, so the recipe reduces across it.
ShardBlocking = tuple[int | None, ...]


class PrepareOp(Protocol):
    def prepare(  # keys become the wrapper's flat schema
        self, weight: Tensor, *, out: Out = None
    ) -> Prepared: ...
    def __call__(self, *args, **kwargs): ...  # tensors and scalars only, never modules


class ShardedPrepareOp(Protocol):  # also prepares before the all-gather
    shard_blocking: ShardBlocking

    def prepare_shard(self, shard: Tensor, *, out: Out = None) -> Prepared: ...
    def complete_gathered(self, wire: Mapping[str, Tensor], *, out: Out = None) -> Prepared: ...


def install_prepared_weights(module: nn.Module, ops: Mapping[str, PrepareOp | ShardedPrepareOp]) -> None: ...
def unsharded_prepared_or_none(weight: torch.Tensor) -> UnshardedPreparedTensor | None: ...
```

- `prepare` must be a pure function of the weight and frozen config, and every value it returns
  must be a tensor: the values become FSDP-owned unsharded storage, gathered at unshard and freed
  at reshard. Each returned tensor must be contiguous and own its whole storage, since FSDP sizes
  that storage from `numel * itemsize` on every later unshard.
- With `out` supplied (every unshard after the first), a preparation method must write into exactly
  the tensors `out` holds and return the same mapping. Rebinding an entry to a fresh tensor raises:
  FSDP keeps the original objects, so the op would read stale data. `out=None` means allocate, which
  is the first unshard.
- `install_prepared_weights` wraps the named parameters in place and registers a `state_dict` post
  hook that replaces each wrapped entry with a plain alias of the master shard, so a checkpoint
  never contains the subclass. Installation has to reach checkpointing because the governing
  invariant is that a checkpoint must not encode whether caching was on: the wrapper is a runtime
  choice, and a checkpoint that carried it would refuse to load into a model that did not make the
  same choice. The caller supplies the mapping from parameter name to op, which may differ per
  parameter. A missing or already-wrapped parameter raises, as does an op matching neither protocol.
  Run before `fully_shard`.
- An op detects preparation itself with `unsharded_prepared_or_none(weight)`, then reads each
  prepared tensor as `weight.prepared_<name>`. Reading the `.prepared` mapping instead breaks the
  dynamo graph, because dynamo derives a source only for the names `__tensor_flatten__` reports, so
  the mapping's values reach a sourceless builder that cannot wrap a `FakeTensor`. Unwrapped weights
  take today's code path, and an op may prepare only some of its weights. A `ShardedPreparedTensor`
  at compute time means the op ran outside its weights' unshard scope and errors.

### Preparing before the all-gather

An op opts in by implementing `prepare_shard` and `complete_gathered`. Implementing neither is the
opt-out, so a recipe that cannot be sharded declares nothing and no flag selects the wrong path.
Such an op normally keeps `prepare` as well, which makes the same compute measurable both ways.

- **`prepare_shard` must be equivariant.** Preparing each shard and joining the results must equal
  preparing the joined weight, bitwise. `shard_blocking` declares when that holds, one entry per
  axis of the weight, giving the extent over which the recipe shares a scale: `1` for an axis whose
  indices are computed independently, `n` for an axis tiled at extent `n`, and `None` for an axis
  the recipe reduces across. Blockwise fp8 over experts declares `(1, 128, 128)`. Installation
  cannot check this, because the mesh does not exist until `fully_shard` runs, so the first unshard
  checks it and raises on every rank at once.
- **A pre-gather recipe cannot depend on a statistic reduced across the sharded axis**, because the
  data is already prepared by the time the shards meet. Per-tensor scaling is the usual example. A
  recipe that needs one can still have it, by computing it in a separate collective after the
  optimizer step and reading it as frozen config, but not from inside the hook.
- **`complete_gathered` receives what the collective carried**, keyed by the names `prepare_shard`
  returned, and returns the full prepared set. It must pass the gathered tensors through by identity
  rather than copying them, and every one of them must appear in what it returns, or it would be
  transported on every unshard and never read. Anything it derives is newly allocated storage that
  FSDP then owns on the same terms as `prepare`'s output.

## Lifetime and scope

`fully_shard` decides prepared-tensor lifetimes, and it decides them a whole unit at a time.

Liveness belongs to storage, not to objects: the wrapper, its `.prepared` mapping, and every tensor
in that mapping are the same objects for the whole run. Reshard frees the prepared tensors by
resizing them to zero bytes, leaving the objects in place; the next unshard re-allocates their
storage and `prepare` refills it. So an op re-reads its `prepared_<name>` attributes on every call
and never holds a prepared tensor across a reshard.

```
                          unshard                            reshard
master shard (sharded)  ############################################   always resident
gather buffer              ####                                        transient
prepared tensors               P###############################        the cache
                               ^ prepare
```

The gather buffer is released once `prepare` has consumed it, but FSDP gathers every parameter of a
unit before preparing any of them, so the unit's full unsharded `param_dtype` size stays a momentary
peak. Preparing before the all-gather removes that peak, because no high-precision gather buffer is
ever allocated: what the collective returns is already the cache.

Two existing FSDP2 knobs set the resident window, both per unit: `reshard_after_forward` (RAF), a
`fully_shard` argument, and `set_reshard_after_backward` (RAB), a method on the `FSDPModule`. `M` is
the number of microbatches.

```
P  prepare runs     #  prepared tensors resident     |  reshard frees them

                      mb0.fwd  mb0.bwd    mb1.fwd  mb1.bwd   prepares/step
RAF=True   RAB=True    P####|   P####|     P####|   P####|        2M
RAF=False  RAB=True    P###############|   P###############|       M
RAF=False  RAB=False   P##################################|       1
```

(NOTE: RAB=True on the final microbatch, or the optimizer's update never reaches `prepare`.)

Activation checkpointing adds nothing to any row: the recomputed forward runs inside backward, where
FSDP skips its post-forward reshard, so recompute rides the unshard backward already did.

Because the flags are per unit, holding a wrapped weight past its forward holds its unit's unwrapped
weights too, at full `param_dtype` size:

```
fully_shard(block)                        RAF/RAB here govern all three
  attn.qkv_proj, attn.o_proj, mlp.router  unwrapped: full param_dtype weight resident

fully_shard(block.mlp.experts)            its own unit, its own RAF/RAB
  gate_proj, up_proj, down_proj           wrapped: prepared tensors resident
```

`## Rules` already requires an op's weights to share a unit; the flags make that unit the lifetime
granularity too, so give wrapped weights a unit of their own and set RAF and RAB only there.

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
        gu = unsharded_prepared_or_none(gate_up)
        dn = unsharded_prepared_or_none(down)
        return _fused_fp8_experts(x, gate_up, down, gu, dn, num_tokens_per_expert)


# binding site (moe_runtime), setup time, before fully_shard
op = FusedFp8ExpertCompute()
moe.experts.compute = op
install_prepared_weights(moe.experts, {"gate_up_proj": op, "down_proj": op})


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
- On the prepared branch, backward saves the wrapper and re-reads its `prepared_<name>` attributes,
  never the prepared tensors themselves, so a reshard-then-refill between forward and backward stays
  correct.
- Joint preparation (one `prepare` over several weights) is out of scope; pack the weights into
  one parameter instead, as the existing QKV and gate_up fusions do.

## Pros

- Invalidation is free and always correct: the FSDP lifecycle is the cache lifecycle
- One `prepare` covers forward, backward layouts, and AC recompute; one per optimizer step at RAF=False, RAB=False
- Can lower resident unsharded memory: the gather buffer is released, so a weight costs only what
  its preparation returned. Whether that is a net saving is a property of the recipe, not of the
  mechanism. Blockwise fp8 is a wash, two one-byte layouts against a released two-byte buffer, and
  on sm90 that is forced rather than incidental: the fp8 matrix instruction reads only operands
  whose stride 1 runs along the contracted axis, so the forward and the dx GEMM cannot share one
  layout
- Zero-cost when unwrapped, and a new preparation is one `prepare` method

## Cons

- Dim-0 sharding only for now (an implementation limit, not fundamental): excludes
  `fusions.shard_fused_on_dim1` params and parts of the MoE path
- RAF=False with RAB=False holds every prepared tensor in the unit across the whole accumulation
  step, and needs train-loop wiring for the RAB toggle
- Rides on private FSDP2 extension hooks and tensor-subclass machinery; torch.compile is the main risk
- An op that prepares only after the all-gather keeps the collective at high precision and saves no
  communication
- Preparing before the all-gather rules out any recipe that reduces across the sharded axis, and
  requires shards even enough that no rank carries padding
