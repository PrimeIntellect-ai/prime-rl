# Prototype: activation-quant caching, eager vs. `torch.compile`

## Context

This prototypes a design worked out in a long prior conversation about caching redundant quantization work in low-precision (FP8/MXFP8) matmuls in `prime-rl`'s trainer. That conversation already settled a design for the *weight* side (a module-owned cache invalidated by an explicit `clear_cache()` hooked to the optimizer's post-step — see `/Users/goon/tmp/quant_weight_cache.md` if useful background, though it's not part of this task). The *activation* side is a different problem: activations are fresh every microbatch, so a persistent module-owned cache with explicit invalidation doesn't fit — the natural design is a tensor subclass whose cache dies with the tensor's own lifetime. There is no live call site for this in `prime-rl` today (the motivating case — `gate_proj`/`up_proj` or `q_proj`/`k_proj`/`v_proj` redundantly quantizing the same input) is better solved by fusing those projections into one `Linear`, not by caching. This is a documented pattern for a future case, and — more importantly — the design surfaced two claims about `torch.compile` that were reasoned from reading PyTorch's dynamo/AOTAutograd source but never actually run:

1. A tensor subclass with a mutable, per-instance cache dict, checked live inside `__torch_dispatch__`, is unsound under `torch.compile`: the cache hit/miss branch is a Python-level check against state that varies call to call, and it gets resolved once at trace time and baked into the compiled graph permanently. Either the graph bakes in "always call the real op" (cache silently never helps) or, if a hit happened to occur at trace time, "always return this one frozen tensor" (a correctness bug — later calls with different real data get back a stale value).
2. `torch.compile` doesn't need this subclass at all for the redundant-quantization case: `fx_graph_cse` (`torch/_functorch/compile_utils.py`), invoked from `min_cut_rematerialization_partition` (`torch/_functorch/partitioners.py`) — the same partitioner function that implements activation-checkpointing's save-vs-recompute split — already merges two calls to the identical op with identical arguments, for free, on by default (`config.cse`). This was confirmed by reading `fx_graph_cse`'s actual hash-and-merge logic (`(target, args, kwargs)` per node, custom ops fully eligible, no type-based exclusion) but **not** by actually running it and watching two calls collapse into one.

This task builds a small, self-contained, CPU-only prototype to empirically check both claims, using toy stand-in ops — no real FP8 kernels, no GPU. Getting a definitive, observed answer (not just a source-reading inference) is the actual goal here.

## Where this lives

Everything goes under `src/prime_rl/experimental/quant_ckpt/` (a new subpackage), sequestered from both the main `prime_rl` package's real code paths and from `tests/` (so a default `uv run pytest tests/` CI-style invocation never touches it — it's run explicitly, see "How to run" below):

- `src/prime_rl/experimental/__init__.py` — empty, makes `experimental` an importable subpackage. Check first whether `src/prime_rl/experimental/` already exists (it shouldn't on a fresh `main` checkout, but confirm) before creating it.
- `src/prime_rl/experimental/quant_ckpt/__init__.py` — empty.
- `src/prime_rl/experimental/quant_ckpt/README.md` — copy `/Users/goon/tmp/activation_cache.md` (a design note covering this exact material — motivation, the eager tensor-subclass design, why compile breaks it, the CSE resolution and its requirements, the `is_compiling()` gate, and status) as the basis for this file. Read it, then reproduce it here nearly verbatim, adding one short paragraph at the top: this directory contains an empirical prototype (`quant_cache_tensor.py` + `test_quant_cache_tensor.py`) validating the claims below, since they were originally derived from reading PyTorch source rather than observed directly — link the two claims above to the two groups of tests described below.
- `src/prime_rl/experimental/quant_ckpt/quant_cache_tensor.py` — the subclass implementation, spec below.
- `src/prime_rl/experimental/quant_ckpt/test_quant_cache_tensor.py` — the test file, spec below.

Note that some subpackages under `src/prime_rl/` are namespace packages without `__init__.py` (e.g. `entrypoints/`, `trainer/`) while others have one (`monitors/`, `evals/`, `dashboard/`, `orchestrator/`); include `__init__.py` here to be unambiguous rather than rely on implicit namespace-package resolution.

## Environment setup (do this first, verify before writing any implementation code)

This repo's `pyproject.toml` restricts lock resolution to Linux only (`environments = [...]`), so a plain `uv sync` cannot resolve at all on macOS — this is a pre-existing, known issue, not something to re-diagnose. The fix is two separate steps, since "get `prime_rl` importable" and "get a working torch" have different constraints:

```bash
cd /Users/goon/github/PrimeIntellect-ai/prime-rl-feat-quant-act-ckpt-proto
rm -rf .venv
uv venv --python 3.12
uv pip install -e packages/prime-rl-configs --no-deps
uv pip install -e . --no-deps
uv pip install --group dev
uv pip install torch pytest
```

The first four lines are a known-working recipe (from project memory) for making `prime_rl` and `prime_rl.configs` importable on macOS by skipping dependency resolution entirely (`--no-deps`) rather than trying to resolve the Linux-only lock — this alone does **not** install torch (that recipe was originally written for LSP navigation only, never running code). The last line installs a real, working torch and pytest directly from PyPI (torch ships native macOS wheels there), as its own command outside the project's own locked dependency graph — untested in combination with the `--no-deps` installs above, so verify it before proceeding:

```bash
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import prime_rl; print(prime_rl.__file__)"
```

If either fails, stop and report back rather than working around it with something ad hoc — this is exactly the kind of thing to flag rather than silently patch.

## `quant_cache_tensor.py`

```python
import torch
from typing import Callable, ClassVar
from torch._ops import OpOverload

aten = torch.ops.aten

class QuantCacheTensor(torch.Tensor):
    _cacheable_ops: ClassVar[dict[OpOverload, Callable]] = {}
    _REWRAP_OPS: ClassVar[set[OpOverload]] = {
        aten.reshape.default, aten.view.default, aten.contiguous.default, aten.detach.default,
    }

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls, data.shape, strides=data.stride(), storage_offset=data.storage_offset(),
            dtype=data.dtype, layout=data.layout, device=data.device, requires_grad=data.requires_grad,
        )

    def __init__(self, data: torch.Tensor):
        self._data = data
        self._cache: dict = {}

    def __tensor_flatten__(self):
        return ["_data"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        return cls(inner_tensors["_data"])

    def __repr__(self):
        return f"QuantCacheTensor({self._data!r}, cached_keys={list(self._cache.keys())})"

    @classmethod
    def register_cacheable_op(cls, op: OpOverload, key_fn: Callable):
        cls._cacheable_ops[op] = key_fn

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, QuantCacheTensor) or torch.compiler.is_compiling():
            return x
        return cls(x)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unwrap = lambda a: a._data if isinstance(a, QuantCacheTensor) else a
        self_arg = next((a for a in args if isinstance(a, QuantCacheTensor)), None)

        if func in cls._cacheable_ops and self_arg is not None:
            key = (func, cls._cacheable_ops[func](args, kwargs))
            if key in self_arg._cache:
                return self_arg._cache[key]
            result = func(*map(unwrap, args), **kwargs)
            self_arg._cache[key] = result
            return result

        result = func(*map(unwrap, args), **kwargs)
        if func in cls._REWRAP_OPS and self_arg is not None:
            out = QuantCacheTensor(result)
            out._cache = self_arg._cache  # same dict object, not a copy — this is what lets the
            return out                     # cache survive a reshape/contiguous/detach chain
        return result
```

Two details matter beyond just transcribing this:
- `__tensor_flatten__`/`__tensor_unflatten__` are not optional polish. Without them, `torch.utils._python_dispatch.is_traceable_wrapper_subclass` returns `False` and dynamo will very likely graph-break on this subclass immediately rather than tracing through `__torch_dispatch__` at all — which would prevent the compile test (below) from ever exercising the behavior it's meant to check. `_cache` is deliberately excluded from the flatten contract (it's not a tensor, and it's not meant to survive a compile-triggered reconstruction) — this is expected and part of what's being tested, not a bug to fix.
- In the rewrap branch, `out._cache = self_arg._cache` must be the *same dict object*, not a copy (`dict(self_arg._cache)` would silently break the whole point) — confirm this literally in code review before running tests, since a copy would still pass tests where the reshape happens once, but silently fail to dedupe across two independent reshape chains from the same original tensor.

## `test_quant_cache_tensor.py`

Dummy op and fixtures, module scope:

```python
import torch
import pytest
from torch._dynamo.testing import CompileCounterWithBackend

from prime_rl.experimental.quant_ckpt.quant_cache_tensor import QuantCacheTensor

_calls: dict[str, int] = {}

@torch.library.custom_op("proto::fake_quantize", mutates_args=())
def fake_quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    _calls["fake_quantize"] = _calls.get("fake_quantize", 0) + 1
    return (x * scale).to(torch.float16)

@fake_quantize.register_fake
def _(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=torch.float16)  # meta only — must NOT touch _calls

def _setup_ctx(ctx, inputs, output):
    ctx.input_dtype = inputs[0].dtype

def _backward(ctx, grad_output):
    return grad_output.to(ctx.input_dtype), None  # straight-through estimator

fake_quantize.register_autograd(_backward, setup_context=_setup_ctx)
QuantCacheTensor.register_cacheable_op(torch.ops.proto.fake_quantize.default, key_fn=lambda args, kwargs: args[1])

@pytest.fixture(autouse=True)
def _reset():
    _calls.clear()
    torch._dynamo.reset()
    yield
```

The counter increments only in the real implementation, never in `register_fake` — this is the detail that makes it a trustworthy oracle for "did the real computation actually run," as opposed to counting trace-time meta invocations.

Eager tests (assert on `_calls["fake_quantize"]`):

1. `test_cache_hit_avoids_recompute` — wrap a tensor once via `QuantCacheTensor.from_tensor`, call `torch.ops.proto.fake_quantize(x, 2.0)` twice on the same wrapped instance → count is 1.
2. `test_different_keys_dont_collide` — same wrapped tensor, `scale=2.0` then `scale=3.0` → count is 2.
3. `test_rewrap_preserves_cache_through_reshape_contiguous_detach` — from one `QuantCacheTensor.from_tensor(x)`, build `x.reshape(...).contiguous().detach()` and quantize it; build a *second, independent* `x.reshape(...).contiguous().detach()` chain from the same original wrapped `x` and quantize that too → count is 1 (both chains share the underlying cache dict via the same original instance).
4. `test_unregistered_op_strips_wrapper` — `wrapped + 1` returns a plain `torch.Tensor`, not `QuantCacheTensor` (`isinstance` check).
5. `test_separate_instances_dont_share_cache` — two independently-wrapped tensors, same `scale` → count is 2, not 1 (cache doesn't leak across unrelated instances).

Compile tests:

6. `test_cse_dedupes_sibling_calls_under_compile` — the core claim to verify. A plain function (no `QuantCacheTensor` involved at all) that calls `torch.ops.proto.fake_quantize(x, 2.0)` *twice* on the same plain tensor `x`, sums both results, and is compiled via `torch.compile(fn, backend=CompileCounterWithBackend("inductor"), fullgraph=True)`. Call the compiled function with `x = torch.randn(4, 4, requires_grad=True)` and call `.backward()` on the output (forces the joint forward+backward graph path, since `fx_graph_cse` runs on that joint graph before it's split). Assert `_calls["fake_quantize"] == 1`. Corroborate directly by walking `cnt.graphs[-1].graph.nodes` for `call_function` nodes whose target involves `fake_quantize`, and assert there's exactly one — this is stronger evidence than the call count alone, since it directly shows the graph was rewritten to one node, not two nodes that happen to both execute the cheap path.
7. `test_eager_does_not_dedupe_for_contrast` — the exact same function, run eager (no `torch.compile` at all) → count is 2. This exists specifically so a broken test harness that always reports 1 (e.g. a counter-reset bug) gets caught, not to test anything about compile itself.
8. `test_quant_cache_tensor_under_compile` — **exploratory, write this to observe and report, not to assert one predicted outcome**. Compile a function that does `wrapped = QuantCacheTensor.from_tensor(x); return torch.ops.proto.fake_quantize(wrapped, 2.0)`, using `CompileCounterWithBackend("inductor")` as the backend (not `fullgraph=True` here — allow a graph break to actually happen and be observed rather than turning it into a hard error). Call the compiled function twice, with genuinely different real tensor data each time (e.g. all-ones then all-twos). Compute a reference by calling the *same, uncompiled* function fresh (a new `QuantCacheTensor.from_tensor` wrap each time, so no cross-call cache contamination) on the second input, and compare. Two possible outcomes, both worth capturing precisely in the test and its output: (a) the compiled second call's result differs from the reference — this confirms claim 1 (a stale, frozen result), or (b) dynamo graph-breaks or raises when it hits the subclass, in which case record that instead. Print `cnt.frame_count` and use `cnt.graphs[-1].graph.print_readable()` (if any graph was produced) regardless of outcome, so the actual generated graph is visible in test output for manual inspection. Once the outcome is known, turn this into a concrete `assert` matching what was actually observed (don't leave it as a bare print-and-pass) — and separately, update `/Users/goon/tmp/activation_cache.md`'s "Why this breaks under torch.compile" section with a one-line note on which specific failure mode was empirically confirmed, since that file currently states this as inferred rather than observed.

## How to run

```bash
uv run pytest src/prime_rl/experimental/quant_ckpt/ -v
```

Explicit path, not swept into a `uv run pytest tests/`-scoped CI invocation. (A fully bare `pytest` with no path from repo root would still discover it, since this repo's `pyproject.toml` sets no `testpaths` restriction — "sequestered" here means "not under `tests/`, not part of the normal test-suite convention," not "literally invisible to every possible invocation.")

## Verification

- Environment check (above) passes before any implementation work starts.
- All 5 eager tests pass as specified — these are unconditional, fixed-outcome assertions.
- Test 6 (CSE) passes as specified — this is the one other fixed-outcome assertion; if it fails, that's a real, reportable finding (the central "compile gets this for free" claim would be wrong), not something to work around until it passes.
- Test 7 passes (sanity check on the harness itself).
- Test 8 is written, run, and its actual observed outcome is reported back — plus the corresponding one-line update to `/Users/goon/tmp/activation_cache.md`.
