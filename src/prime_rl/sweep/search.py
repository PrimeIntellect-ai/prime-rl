import hashlib
import json
import math
import random
from itertools import product
from typing import Any

from prime_rl.configs.sweep import (
    ChoiceParameterConfig,
    IntUniformParameterConfig,
    LogUniformParameterConfig,
    SweepParameterConfig,
    UniformParameterConfig,
)
from prime_rl.sweep.materialize import Trial, trial_label


def parameters_hash(parameters: dict[str, Any]) -> str:
    """Stable 8-character hash of a trial's flat override dict.

    Sorted-key JSON keeps the hash stable across dict insertion order. Trial IDs
    use this as a suffix so identity survives deletion and regeneration.
    """
    serialized = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.blake2b(serialized.encode(), digest_size=4).hexdigest()


def _make_trial(idx: int, parameters: dict[str, Any]) -> Trial:
    trial_id = f"{idx:04d}-{parameters_hash(parameters)}"
    label = trial_label(parameters) or trial_id
    return Trial(id=trial_id, label=label, parameters=parameters)


def expand_grid(parameters: dict[str, SweepParameterConfig]) -> list[Trial]:
    paths = list(parameters.keys())
    value_lists: list[list[Any]] = []
    for path in paths:
        config = parameters[path]
        if not isinstance(config, ChoiceParameterConfig):
            raise ValueError(
                f"Grid strategy requires choice (values=...) parameters; '{path}' uses distribution "
                f"'{config.distribution}'."
            )
        value_lists.append(config.values)

    trials = []
    for idx, values in enumerate(product(*value_lists)):
        trials.append(_make_trial(idx, dict(zip(paths, values))))
    return trials


def _sample_parameter(config: SweepParameterConfig, rng: random.Random) -> Any:
    if isinstance(config, ChoiceParameterConfig):
        return rng.choice(config.values)
    if isinstance(config, UniformParameterConfig):
        return rng.uniform(config.min, config.max)
    if isinstance(config, LogUniformParameterConfig):
        return math.exp(rng.uniform(math.log(config.min), math.log(config.max)))
    if isinstance(config, IntUniformParameterConfig):
        steps = (config.max - config.min) // config.step
        return config.min + rng.randint(0, steps) * config.step
    raise ValueError(f"Unsupported parameter type: {type(config)!r}")


def sample_random(
    parameters: dict[str, SweepParameterConfig],
    num_trials: int,
    seed: int | None = None,
) -> list[Trial]:
    """Draw ``num_trials`` independent samples using a seeded RNG.

    The seed is the only source of randomness, so a study can be replayed
    exactly by recording it in the manifest.
    """
    rng = random.Random(seed)
    trials = []
    for idx in range(num_trials):
        sampled = {path: _sample_parameter(config, rng) for path, config in parameters.items()}
        trials.append(_make_trial(idx, sampled))
    return trials
