from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn

from prime_rl.utils.logger import get_logger


class LinearRecipe(ABC):
    name: str

    @abstractmethod
    def linear_cls(self) -> type[nn.Linear]: ...

    @abstractmethod
    def is_linear_shape_supported(self, in_features: int, out_features: int) -> bool: ...

    @abstractmethod
    def convert_linear(self, mod: nn.Linear) -> nn.Linear: ...

    """Convert a high-precision linear to this recipe's low-precision linear."""


@dataclass(frozen=True)
class PreparedWeights:
    pass


@dataclass(frozen=True)
class PreparedActivations:
    pass


class MoEExpertKernel(ABC):
    """Computes fully routed MoE. Each backend can decide what pre/post processing needs to happen."""
    name: str

    @abstractmethod
    def preprocess_weights(self, w1: Tensor, w2: Tensor, w3: Tensor | None) -> PreparedWeights: ...

    @abstractmethod
    def preprocess_activations(self, x: Tensor, num_tokens_per_expert: Tensor) -> PreparedActivations: ...

    @abstractmethod
    def compute(self, weights: PreparedWeights, activations: PreparedActivations) -> Tensor: ...

    @abstractmethod
    def postprocess_activations(self, out: Tensor, activations: PreparedActivations) -> Tensor: ...


def replace_all_linear_with_low_precision_linear(
    model: nn.Module, recipe: LinearRecipe, ignore_modules: list[str]
) -> None:
    """Generic replace linear. Replaces nn.Linear in a module with the specified recipe's linear implementation. Skips linears which are either on the ignore list or do not fit the requirements."""
    logger = get_logger()
    logger.info(f"Replacing linear layers with {recipe.name} linear layers (ignore={ignore_modules})")
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
        if not recipe.is_linear_shape_supported(module.in_features, module.out_features):
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, recipe.convert_linear(module))
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with {recipe.name} linear "
        f"(skipped {len(skipped_modules)} by name, {len(skipped_unaligned)} by shape); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
