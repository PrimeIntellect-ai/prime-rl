from __future__ import annotations

import re
from abc import abstractmethod

from torch import nn

from prime_rl.trainer.models.layers.recipe import GEMMRecipe
from prime_rl.utils.logger import get_logger


class LinearRecipe(GEMMRecipe[nn.Linear]):
    """A ``GEMMRecipe`` that swaps dense ``nn.Linear`` modules for a low-precision implementation."""

    @abstractmethod
    def is_shape_supported(self, in_features: int, out_features: int) -> bool: ...


def replace_all_linear_with_low_precision_linear(
    model: nn.Module, recipe: LinearRecipe, ignore_modules: list[str]
) -> None:
    """Generic replace linear. Replaces nn.Linear in a module with the specified recipe's linear implementation. Skips linears which are either on the ignore list or do not fit the requirements."""
    logger = get_logger()
    logger.info(f"Replacing linear layers with {recipe.name} ({recipe.impl}) linear layers (ignore={ignore_modules})")
    replaced_modules: list[str] = []
    skipped_modules: list[str] = []
    skipped_unaligned: list[str] = []
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if not isinstance(module, nn.Linear):
            continue
        if any(re.search(pattern, name) for pattern in ignore_modules):
            skipped_modules.append(name)
            continue
        if not recipe.is_shape_supported(module.in_features, module.out_features):
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, recipe.convert(module))
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with {recipe.name} linear "
        f"(skipped {len(skipped_modules)} by name, {len(skipped_unaligned)} by shape); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
