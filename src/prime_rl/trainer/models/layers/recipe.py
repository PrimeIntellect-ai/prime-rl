from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from torch import nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class RecipeData:
    """Structured, recipe-specific data attached to a converted module.

    Subclass per recipe (e.g. to hold kernel preference, scale mode, block size)
    instead of setting ad-hoc attributes directly on the module.
    """


class GEMMRecipe(ABC, Generic[ModuleT]):
    """Shared interface for a low-precision GEMM recipe.

    A recipe converts a high-precision module (an ``nn.Linear``, or the
    ``GroupedExperts`` behind a routed-MoE GEMM) to a low-precision compute path,
    in three steps: ``pre_quant`` (validate/prepare), ``quantize`` (swap the
    compute path), and ``post_quant`` (attach the recipe's ``RecipeData`` to the
    module for introspection). ``name`` identifies the numerical recipe (e.g.
    ``fp8_blockwise``, ``mxfp8``); ``impl`` identifies the kernel implementation
    backing it (e.g. ``deepgemm``, ``torchao``, ``custom``).
    """

    name: str
    impl: str

    def pre_quant(self, mod: ModuleT) -> ModuleT:
        return mod

    @abstractmethod
    def quantize(self, mod: ModuleT) -> tuple[ModuleT, RecipeData | None]:
        """Swap ``mod``'s compute path for this recipe's low-precision implementation."""
        ...

    def post_quant(self, mod: ModuleT, data: RecipeData | None) -> ModuleT:
        if data is not None:
            mod._recipe_data = data
        return mod

    def convert(self, mod: ModuleT) -> ModuleT:
        mod = self.pre_quant(mod)
        mod, data = self.quantize(mod)
        return self.post_quant(mod, data)
