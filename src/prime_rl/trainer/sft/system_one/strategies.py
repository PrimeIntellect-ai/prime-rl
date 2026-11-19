"""The System One strategy of each decision model, by architecture name."""

from typing import TYPE_CHECKING

from transformers import PretrainedConfig

from prime_rl.trainer.sft.system_one.clef_flash import ClefFlashStrategy
from prime_rl.trainer.sft.system_one.data import SystemOneStrategy

if TYPE_CHECKING:
    from prime_rl.trainer.model import ExtraConfigSelection

SYSTEM_ONE_STRATEGIES: dict[str, type[SystemOneStrategy]] = {"ClefFlashForJointSchema": ClefFlashStrategy}
"""Architecture name -> strategy."""


def get_system_one_strategy_cls(
    model_config: PretrainedConfig, selection: "ExtraConfigSelection | None" = None
) -> type[SystemOneStrategy]:
    """The strategy of the model's first architecture. A class selected by an extra config file is looked up by its
    name, as config.json keeps the stock architecture."""
    architectures = [selection.cls.__name__] if selection is not None else model_config.architectures or []
    if architectures and architectures[0] in SYSTEM_ONE_STRATEGIES:
        return SYSTEM_ONE_STRATEGIES[architectures[0]]
    raise ValueError(
        f"System One data trains a decision model, an architecture with a System One strategy, but the model's "
        f"architectures {architectures} do not start with one. Strategies are registered for "
        f"{sorted(SYSTEM_ONE_STRATEGIES)} in SYSTEM_ONE_STRATEGIES."
    )
