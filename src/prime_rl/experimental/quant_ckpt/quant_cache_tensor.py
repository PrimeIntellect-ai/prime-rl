from typing import Callable, ClassVar

import torch
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
