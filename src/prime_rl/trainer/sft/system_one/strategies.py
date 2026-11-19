"""The System One strategy of each decision model, by architecture name, like vLLM's ``READ_STRATEGIES``."""

from transformers import PretrainedConfig

from prime_rl.trainer.sft.system_one.data import SystemOneStrategy

SYSTEM_ONE_STRATEGIES: dict[str, type[SystemOneStrategy]] = {}
"""Architecture name -> strategy."""


def get_system_one_strategy_cls(model_config: PretrainedConfig) -> type[SystemOneStrategy]:
    """The strategy of the model's architecture, the first entry of ``architectures``, which vLLM also serves."""
    architectures = list(getattr(model_config, "architectures", None) or [])
    if architectures and architectures[0] in SYSTEM_ONE_STRATEGIES:
        return SYSTEM_ONE_STRATEGIES[architectures[0]]
    raise ValueError(
        f"System One data trains a decision model, an architecture with a System One strategy, but the model's "
        f"architectures {architectures} do not start with one. Strategies are registered for "
        f"{sorted(SYSTEM_ONE_STRATEGIES)} in SYSTEM_ONE_STRATEGIES (src/prime_rl/trainer/sft/system_one/strategies.py)."
    )
