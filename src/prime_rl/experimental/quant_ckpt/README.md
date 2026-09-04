# Activation Quant Caching: Eager vs. Compile

This directory is an empirical prototype validating two claims that were originally derived from
reading PyTorch's dynamo/AOTAutograd source, not from actually running anything. `quant_cache_tensor.py`
implements the design below; `test_quant_cache_tensor_eager.py` and `test_quant_cache_tensor_compile.py`
check it (shared toy op/kernel definitions live in `fake_kernels.py`, the reset fixture in
`conftest.py`). Claim 1 (a live cache dict inside
`__torch_dispatch__` is unsound under `torch.compile`) is checked by `test_quant_cache_tensor_under_compile`.
Claim 2 (`torch.compile`'s CSE pass already dedupes redundant quantize calls for free, no subclass needed)
is checked by `test_cse_dedupes_sibling_calls_under_compile` and `test_eager_does_not_dedupe_for_contrast`.
A third, eager-only capability — the cache surviving from forward into the backward of two *independent*
`autograd.Function`s that share one input — is checked by
`test_fwd_and_wgrad_quantize_dedup_independently_across_sibling_calls`; see "Eager path across sibling
`autograd.Function`s" below. The compile equivalent of that same w1/w3 case — no subclass at all, just
compiling the sibling `autograd.Function` calls directly — is checked by
`test_cse_dedupes_fwd_and_wgrad_across_sibling_calls_under_compile`; see "Compiled path: CSE, not the
subclass" below.

## Motivation

Activations are fresh every microbatch, so there's no persistent-state event (like an optimizer step) to
hang an explicit invalidation on — the right lifetime for a cache here is the activation tensor's own
lifetime. No call site in this codebase needs this today: the obvious case (SwiGLU gate/up, QKV both
quantizing the same input) is better solved by fusing those projections into one Linear than by caching.
This documents the pattern for if/when a genuinely un-fusable sibling-consumer case shows up, and — more
importantly — the eager/compile split it requires, since that distinction is easy to get wrong.

## Eager path: a lifetime-bound tensor subclass

Wrap the *pre-quantized* tensor in a `torch.Tensor` wrapper subclass. The cache dict lives on the instance
and dies when it does — no `clear_cache()`, no version counters, unlike the weight cache (see
`quant_weight_cache.md`).

```python
class QuantCacheTensor(torch.Tensor):
    _cacheable_ops: ClassVar[dict[OpOverload, Callable]] = {}
    _REWRAP_OPS = {aten.reshape.default, aten.contiguous.default, aten.detach.default}

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls, data.shape, strides=data.stride(), dtype=data.dtype,
            device=data.device, requires_grad=data.requires_grad,
        )

    def __init__(self, data: torch.Tensor):
        self._data = data
        self._cache: dict = {}

    @classmethod
    def register_cacheable_op(cls, op, key_fn):
        cls._cacheable_ops[op] = key_fn

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unwrap = lambda a: a._data if isinstance(a, QuantCacheTensor) else a
        if func in cls._cacheable_ops:
            self_arg = next(a for a in args if isinstance(a, QuantCacheTensor))
            key = (func, cls._cacheable_ops[func](args, kwargs))
            if key in self_arg._cache:
                return self_arg._cache[key]
            result = func(*map(unwrap, args), **kwargs)
            self_arg._cache[key] = result
            return result
        result = func(*map(unwrap, args), **kwargs)
        if func in cls._REWRAP_OPS:
            out = QuantCacheTensor.__new__(QuantCacheTensor, result)
            out._data, out._cache = result, self_arg._cache  # same dict object, not a copy
            return out
        return result  # default: never rewrap
```

Cache key is `(op, non-tensor args)` — tensor identity is free, since the dict is already scoped to one
instance. Default policy for any op not in the registry is to unwrap and return a plain tensor ("never
rewrap"), *except* a small allowlist: `reshape`, `contiguous`, `detach`. These three are confirmed (by
tracing every real quantize call site in `fp8_linear.py` and `fp8_grouped_gemm.py`, forward and backward)
as the actual ops that sit between an activation and its quantize call — without rewrapping through them,
the wrapper is stripped before any quantize call is ever reached and the cache is dead code.
`torch.nn.functional.pad` and dropout are explicitly *not* on this list — both change values, so treating
them as identity-preserving would return silently-wrong cached results.

## Eager path across sibling `autograd.Function`s

The cache also survives across the forward/backward boundary of two *independent* `autograd.Function`
instances that share one wrapped input — not just within a single tensor's own dispatch chain. Concretely,
in an unfused SwiGLU MLP, `w1` (gate_proj) and `w3` (up_proj) both consume the same activation `h`. In
`fp8_linear.py`'s `Float8BlockwiseLinear`, `x` is quantized once per GEMM layout it's needed in: a row-major
cast for the forward matmul (`Y=X@W`), and a genuinely *different* op — a transposed-layout cast — for the
weight-gradient matmul (`dW=dY^T@X`). So `w1` and `w3` each create two independent dedup opportunities: their
forwards both want the row-major cast of `h`, and their backwards (wgrad) both want the transposed-layout
cast of `h`. These are two separate cache entries (different op key), not one merged count.

This works because `ctx.save_for_backward(h, ...)` in plain eager mode preserves `h`'s exact object identity
— confirmed empirically: `saved is h` and `saved._cache is h._cache` both hold. No `__tensor_flatten__`/
`__tensor_unflatten__` round-trip happens here, unlike under `torch.compile` (see "Why this breaks under
`torch.compile`" below) — that round-trip is specific to dynamo/functorch tracing, not to plain eager
autograd. So as long as both `Function.apply()` calls are handed the *same* wrapped `h` instance, their
independent backward calls share its cache correctly. `test_fwd_and_wgrad_quantize_dedup_independently_across_sibling_calls`
checks both the call-count dedup and that the resulting `w1`/`w3` weight gradients are numerically correct,
not just cheap.

## Why this breaks under `torch.compile`

`__torch_dispatch__`'s cache hit/miss branch is a Python-level check against mutable state that resolves
once at trace time and gets baked into the compiled graph permanently. If the cache is empty at trace time
(the normal case), the graph bakes in "always call the real op" — the cache silently never helps. If a hit
happens to occur at trace time, the graph bakes in "always return this specific frozen tensor" — a
correctness bug, since later invocations with different real data would get back the same stale value.

**Empirically confirmed** (`test_quant_cache_tensor_under_compile`): only the first half happens for this
implementation — the cache silently never helps under compile, but the frozen-stale-value bug does not
manifest, because `_cache` is excluded from `__tensor_flatten__`, so every trace-time reconstruction
(`__tensor_unflatten__`) starts with an empty cache regardless of the eager instance's real state, forcing
the compiled graph to always take the "call the real op" branch. See "Blockers and surprises" below for the
full empirical writeup.

## Compiled path: CSE, not the subclass

Don't fight the tracer — use the mechanism that already exists for this. `fx_graph_cse`
(`torch/_functorch/compile_utils.py`) is invoked from `min_cut_rematerialization_partition`
(`torch/_functorch/partitioners.py`) — the *same* partitioner function that implements AC's recompute-tag
split — and runs on the joint forward+backward graph before it's split. Default-on (`config.cse`). It
hashes `(target, args, kwargs)` per node and merges matches; custom ops are eligible, no type-based
exclusion.

Requirements for two calls to actually merge:
- The quantize step must be its own registered `torch.library.custom_op`, not buried inside a fused op's
  body (today it's inlined in `fp8_blockwise_mm`, invisible as its own node). It needs `register_fake`
  (required for any custom op to trace at all), `register_autograd` with a straight-through backward
  (gradient has to flow back to the original high-precision tensor), and `mutates_args=()`.
- Both call sites must hit the identical op target with identical args — including static config like
  `block_size`/`use_ue8m0`, not just the tensor.
- The tensor argument must resolve to the same graph node. This should hold transitively through
  `reshape`/`contiguous`/`detach` sitting in between, since CSE's matching is transitive and those are
  themselves ordinary CSE-eligible ops — reasoned from the mechanism, not confirmed by watching an actual
  traced graph.
- The op must be genuinely deterministic. **Unverified**: check none of these kernels use stochastic
  rounding before relying on this — CSE-merging two calls that were supposed to independently sample
  rounding noise would be a silent correctness bug, not a missed optimization.
- Both calls must land in the same compiled graph region with no intervening graph break.

Not empirically tested: whether AC being active changes any of this. Expectation is no, since CSE runs
before the recompute-tag split on the same joint graph — but this hasn't been run end-to-end.

**Empirically confirmed** (`test_cse_dedupes_sibling_calls_under_compile`): the real op runs once, not
twice, when compiled with the actual "inductor" backend. See "Blockers and surprises" below for how the
graph-level corroboration had to be adjusted, since the originally-planned inspection point doesn't
actually observe CSE's effect.

**Also confirmed for the realistic w1/w3 case** (`test_cse_dedupes_fwd_and_wgrad_across_sibling_calls_under_compile`):
compiling `QuantLinear.apply(h, w1)` and `QuantLinear.apply(h, w3)` directly (no `QuantCacheTensor` involved
at all) collapses both the fwd-layout cast and the wgrad-layout cast to one real call each, with zero graph
breaks under `fullgraph=True` — dynamo traces straight through the custom `autograd.Function`. This
surfaced a genuinely surprising mechanism, not just "CSE merges the two backward calls where they already
were": since `fake_quantize_wgrad(h, 2.0)` depends only on `h` (available at forward time), not on
`grad_output`, `min_cut_rematerialization_partition` relocates its computation into the *forward* graph
entirely as a saved intermediate — the compiled backward graph never calls it at all, it just consumes the
already-computed value. CSE then merges w1's and w3's identical calls the same way regardless of which
graph they end up in. This is a save-vs-recompute decision made by the same partitioner function that
implements activation-checkpointing's split — related to, but not the same claim as, the still-unverified
"whether AC being active changes any of this" note above (this test doesn't wrap anything in
`torch.utils.checkpoint`; it just shows the partitioner making an analogous save/recompute call on a
backward-only computation with no true backward dependency).

## Gating the two paths apart

The wrap factory checks `torch.compiler.is_compiling()` and returns the plain tensor when compiling:

```python
def from_tensor(x):
    if torch.compiler.is_compiling():
        return x
    return QuantCacheTensor(x)
```

The compiled graph never sees the subclass — no `torch._dynamo.disable()`, no graph breaks, no loss of
whatever pointwise fusion Inductor would otherwise do around the op.

## Status

No current call site — SwiGLU/QKV redundancy is solved by fusion, not this. Deliberately a *different*
mechanism from the weight cache (lifetime-bound vs. explicitly-invalidated via `clear_cache()`): the two
are different regimes — weights persist across calls and need an external invalidation signal, activations
don't persist at all — not one design being more "correct" than the other.

## How to run

```bash
uv run pytest src/prime_rl/experimental/quant_ckpt/ -v
```

## Blockers and surprises

All 8 tests pass, but several things along the way didn't go as the original plan (`PLAN.md` at the repo
root) expected. None of these blocked completion; each required a small, empirically-driven adjustment.

**`uv sync` actually resolves on this Mac now.** The environment-setup recipe (and prior project memory)
assumed `uv sync` "cannot resolve at all on macOS" because `pyproject.toml` gates lock resolution to
Linux/darwin-arm64 combos. Checking `uv.lock` directly: `darwin`/`arm64` resolution-markers are present, and
`pyproject.toml`'s base `[project.dependencies]` (the ones that actually matter for `prime_rl` importing)
carry no platform markers at all — only the `gpu` optional-dependency group (`torch`, `vllm`,
`torchtitan`, etc.) is Linux-gated. Concretely, running plain `uv run python -c "..."` (rather than
`uv run --no-sync ...`) silently triggered a full `uv sync`, installing ~168 base packages (openai,
transformers, wandb, datasets, ...) without error — it didn't fail, and it didn't disturb the separately
`uv pip install`'d `torch`/`pytest`. This didn't break anything here, but it means the recipe's "a plain
`uv sync` cannot resolve at all on macOS" framing is now stale; `uv run --no-sync` was used for the rest of
this session to avoid repeated resyncs.

**`prime_rl.__file__` is `None` — expected, not a failure.** `prime_rl` is a namespace package split across
two source trees (`src/prime_rl` and `packages/prime-rl-configs/src/prime_rl`), so it has no `__init__.py`
and no `__file__`. `prime_rl.__path__` correctly lists both trees, confirming the editable installs worked.

**`test_cse_dedupes_sibling_calls_under_compile`'s originally-planned graph corroboration doesn't see
CSE.** The plan called for walking `cnt.graphs[-1].graph.nodes` (from
`CompileCounterWithBackend("inductor")`) to confirm the two `fake_quantize` calls collapsed into one node.
In practice `cnt.graphs` captures the *dynamo-level* graph, produced before AOTAutograd runs — `fx_graph_cse`
is invoked later, inside `min_cut_rematerialization_partition`, on the joint forward+backward graph, which
`CompileCounterWithBackend` never sees. Empirically: the dynamo-level graph still shows two separate
`fake_quantize` nodes even though the runtime call count is 1. To directly observe the post-CSE graph, the
test instead builds a second compile path with `torch._dynamo.backends.common.aot_autograd(fw_compiler=...,
partition_fn=min_cut_rematerialization_partition)` — explicitly passing the same partitioner "inductor" uses
internally, since a generic `aot_autograd()` backend's *default* partitioner does not call `fx_graph_cse` at
all (confirmed by omitting `partition_fn`: 2 nodes and 2 real calls, versus 1 and 1 with it specified). With
that partitioner, the captured forward graph shows exactly one `fake_quantize` node, matching the runtime
call count.

**`cnt.graphs[-1].graph.print_readable()` is a bug in the plan's own test 8 spec** — `torch.fx.Graph` has no
`print_readable`; only `torch.fx.GraphModule` does (`cnt.graphs` stores `GraphModule`s per
`CompileCounterWithBackend`'s source). Fixed to `cnt.graphs[-1].print_readable()`.

**Test 8's originally-specified construction can't reach the "frozen stale value" failure mode at all — and
not for the reason first suspected.** The plan's spec called for `from_tensor(x)` inside the compiled
function, called twice with different real data (all-ones, then all-twos), expecting either a stale result
or a graph break. Empirically: neither happens — both compiled calls return correct results. The *actual*
cause, confirmed by inspecting the traced graph directly, is the `is_compiling()` guard in `from_tensor`
itself: calling `from_tensor` *inside* a compiled region always takes the "already compiling" branch and
returns the bare input tensor unchanged — `QuantCacheTensor.__torch_dispatch__` never runs at all, so there
is no cache-hit/miss branch to bake in anything. The traced graph for this construction is literally
`torch.ops.proto.fake_quantize(l_x_, 2.0)` called directly on the plain input placeholder — no subclass
involved. (An earlier draft of this note attributed the correct-but-cache-free result to "each `from_tensor`
call gets a fresh empty `_cache`" — that's wrong; the subclass is never constructed here at all, so there's
no cache, empty or otherwise, to reason about.) This is exactly the mitigation the design doc's "Gating the
two paths apart" section describes: called as intended (fresh at each call site, forward pass by forward
pass), the guard keeps the subclass out of any compiled graph entirely.

To actually probe `__torch_dispatch__`'s behavior under compile at all, a test has to deliberately route
around this guard — which also means it's testing a misuse scenario (a caller who doesn't call
`from_tensor` fresh at the compile boundary, e.g. one who holds onto an already-wrapped instance across it),
not the intended calling convention. The test does this by pre-warming the cache *eagerly* on a single
`QuantCacheTensor` instance before it ever enters a compiled region, then mutates the underlying `_data` in place (same
identity, same shape/dtype — no recompile triggered) and calls the compiled function again. Even with a
genuinely warm eager cache at first trace time, the observed result is still neither of the plan's two
hypothesized outcomes: no graph break, and no stale value — `r2` correctly reflects the mutated data, and
`_calls` climbs on *every* compiled invocation (2 → 3), showing the real op reruns regardless of the eager
cache's state. Root cause, confirmed by reading `__tensor_unflatten__`: it calls `cls(inner_tensors["_data"])`,
whose `__init__` always sets `self._cache = {}` fresh. Since `_cache` is excluded from
`__tensor_flatten__`'s contract, every trace-time fakification/reconstruction of the subclass discards
whatever the real eager cache held and starts empty — dynamo's traced view of the object can *never* see a
warm cache, so the compiled graph can only ever bake in "always call the real op." This makes claim 1's
correctness-bug half (frozen stale return) structurally unreachable for this exact implementation — a
stronger and more precise result than the plan's "either (a) or (b)" framing, and specifically *because* of
the (deliberate) choice to exclude `_cache` from the flatten contract, not despite it. A subclass that did
flatten `_cache` into its traced state would be exposed to the real bug.

**Minor:** compiling `QuantCacheTensor` triggers a `UserWarning` from AOTAutograd's caching layer
("`QuantCacheTensor` does not implement `_stable_hash_for_caching`"). Harmless for this prototype (no
autograd-cache reuse is being tested across process runs), but worth knowing about if this subclass were
ever used for real.
