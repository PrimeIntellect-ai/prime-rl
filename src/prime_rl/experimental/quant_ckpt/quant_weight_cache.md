# Quantized Weight Caching via Modules

## Motivation

Weights don't change until the next optimizer step, but their quantized form gets recomputed on every microbatch — caching it across a gradient-accumulation window avoids that for free.

## The pattern

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class QuantCacheModule(Protocol):
    def clear_cache(self) -> None: ...

def clear_all_quant_caches(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, QuantCacheModule):
            module.clear_cache()

optimizer.register_step_post_hook(lambda *_: clear_all_quant_caches(model))
```

Any module satisfies `QuantCacheModule` just by implementing `clear_cache()` — no inheritance needed. (An ABC mixin is the safer alternative in practice, since a type checker will catch a missing implementation and we avoid accidental method collisions; kept as a Protocol here for simplicity.) Invalidation is exactly "did an optimizer step happen" — no version counters, no identity tracking.

A module using this looks schematically like:

```python
class QuantCachingModule(nn.Module):
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight
        self._cache: dict[str, torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cached_weight_quant = self._cache.get("weight_quant")
        out, weight_quant = some_autograd_fn(x, self.weight, cached_weight_quant)
        if cached_weight_quant is None:
            self._cache["weight_quant"] = weight_quant.detach()
        return out
```

`some_autograd_fn` is any custom op/autograd Function: it accepts an optional pre-computed shortcut, uses it if present, otherwise computes fresh — and either way returns the value used so the module can populate its cache on a miss.

The wrapped op's internal shape is intentionally left unspecified beyond that. The one contract it must satisfy: `ctx.save_for_backward` always saves full-precision tensors regardless of cache hit/miss, backward derives whatever quantized layout(s) it needs independently, and any cached artifact returned as a side-output comes back detached.

## Example: `Float8BlockwiseLinear` (`src/prime_rl/trainer/models/layers/fp8_linear.py`)

Forward quantizes `weight` via `per_block_cast_to_fp8_triton` (`fp8_linear.py:25`); backward's `grad_x` needs it via the transposed `per_block_cast_to_fp8_tp_triton` (`fp8_linear.py:56`) — two different layouts, both invariant across microbatches, both cacheable.

```python
class Float8BlockwiseLinear(nn.Linear):
    def __init__(self, ...):
        ...
        self._quant_cache: dict[str, torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._quant_cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cached_fwd = self._quant_cache.get("fwd")
        out, weight_fwd_fp8 = _fp8_blockwise_mm(x, self.weight, self.block_size, cached_fwd)
        if cached_fwd is None:
            self._quant_cache["fwd"] = weight_fwd_fp8.detach()
        return out
```

(The backward-layout slot mirrors this, populated on its first call in an accumulation window.)

## Prior art

`mxfp8_linear.py`'s `73efbe988` patch already validates the idea, via a monkeypatch on torchao's `mx_mm.forward` keyed by `(id(weight), weight._version)`. This design fixes its flaws: a never-evicted global dict keyed by raw object identity, only the forward/dim0 layout cached, and third-party internals patched instead of a first-class module.

## Status

Design only — not implemented or empirically tested in this repo. Unlike `quant_cache_tensor.py` (the activation-side prototype in this same directory), there is no test file validating this pattern under eager or compiled execution; the `some_autograd_fn`/`clear_cache()` sketch above has not been run.
