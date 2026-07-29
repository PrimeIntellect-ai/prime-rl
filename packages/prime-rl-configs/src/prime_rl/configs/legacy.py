"""Translation of pre-``algo`` orchestrator config keys into the current shape.

Hosted training's control plane emits the orchestrator config that predates the
algorithm abstraction: a top-level ``training_mode`` with a sibling ``[teacher]``
block, per-env ``advantage``, and the policy client at ``[client]``. This module
maps that shape onto ``[algo]`` and ``[model.client]`` at validation time so a
control plane that hasn't been updated yet keeps working against a current image.

Temporary. Delete this module and its ``OrchestratorConfig`` validator once every
caller emits ``algo`` directly.
"""

import warnings
from typing import Any

# `training_mode` named the shape of the whole run; `algo.type` names it now.
_TRAINING_MODE_TO_ALGO_TYPE = {"rl": "grpo", "opd": "opd", "sft": "sft"}

# The `default` advantage was GRPO's group-mean baseline; `custom` has no successor.
_ADVANTAGE_TYPE_TO_ALGO_TYPE = {"default": "grpo"}


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"'{old}' is deprecated, use '{new}' instead. Auto-translating for now, but this will be removed in a future release.",
        FutureWarning,
        stacklevel=2,
    )


def _frozen_model(teacher: dict[str, Any]) -> dict[str, Any]:
    """A legacy ``[teacher]`` block (a model plus its client) as a frozen model reference."""
    model = teacher.get("model") or {}
    name = model.get("name")
    if not name:
        raise ValueError("legacy 'teacher' needs 'teacher.model.name' — the served model name of the teacher endpoint")
    unknown = set(teacher) - {"model", "client"}
    if unknown:
        raise ValueError(f"legacy 'teacher' only carried 'model' and 'client', got {sorted(unknown)}")
    return {"name": name, **(teacher.get("client") or {})}


def _migrate_algo(data: dict[str, Any]) -> None:
    """``training_mode`` + ``[teacher]`` -> ``[algo]``."""
    training_mode = data.pop("training_mode", None)
    teacher = data.pop("teacher", None)

    if training_mode is None and teacher is None:
        return
    if "algo" in data:
        raise ValueError(
            "'algo' is set alongside the legacy 'training_mode'/'teacher' keys. Drop the legacy keys — "
            "the translation would be ambiguous."
        )
    if training_mode is None:
        raise ValueError("legacy 'teacher' requires 'training_mode' — a teacher only has a role under 'opd' or 'sft'")

    _deprecated("training_mode", "algo.type")
    if training_mode not in _TRAINING_MODE_TO_ALGO_TYPE:
        raise ValueError(
            f"unknown training_mode {training_mode!r}, expected one of {sorted(_TRAINING_MODE_TO_ALGO_TYPE)}"
        )
    algo_type = _TRAINING_MODE_TO_ALGO_TYPE[training_mode]
    algo: dict[str, Any] = {"type": algo_type}

    if teacher is not None:
        _deprecated("teacher", "algo.teacher (opd) / algo.sampling.source (sft)")
        frozen_model = _frozen_model(teacher)
        if algo_type == "opd":
            # OPD keeps sampling on the policy; the teacher only scores it.
            algo["teacher"] = frozen_model
        elif algo_type == "sft":
            # SFT sampled *from* the teacher, which is now a sampling source.
            algo["sampling"] = {"source": frozen_model}
        else:
            raise ValueError("'teacher' is not supported under training_mode = 'rl' — rl samples from the policy only")
    elif algo_type == "opd":
        raise ValueError("training_mode = 'opd' requires a 'teacher' to score the policy's rollouts")

    data["algo"] = algo


def _migrate_advantage(env: dict[str, Any]) -> None:
    """A train env's ``advantage`` -> its ``algo``."""
    advantage = env.pop("advantage", None)
    if advantage is None:
        return
    if "algo" in env:
        raise ValueError("'algo' is set alongside the legacy 'advantage' key. Drop 'advantage'.")
    if not isinstance(advantage, dict):
        raise ValueError(f"'advantage' must be a table, got {type(advantage).__name__}")

    _deprecated("train.env.advantage", "train.env.algo")
    advantage = dict(advantage)
    advantage_type = advantage.pop("type", "default")
    if advantage_type not in _ADVANTAGE_TYPE_TO_ALGO_TYPE:
        raise ValueError(
            f"advantage.type = {advantage_type!r} has no algorithm equivalent. A custom advantage function is now a "
            "custom loss — see 'algo' for the supported algorithms."
        )

    # Both reward-shaping knobs were redefined rather than renamed, so translating
    # them would silently change training dynamics. Make the caller restate them.
    if advantage.pop("length_penalty", None) is not None:
        raise ValueError(
            "advantage.length_penalty was redefined: it now normalizes by the group's max output tokens rather than "
            "orchestrator.seq_len, splits into 'num_output_tokens_weight' / 'num_input_tokens_weight' / "
            "'num_turns_weight', and drops 'gate_by_correctness'. Set 'algo.length_penalty' explicitly."
        )
    if advantage.pop("length_weighted_baseline", None):
        raise ValueError("advantage.length_weighted_baseline was removed; the GRPO baseline is the plain group mean.")

    env["algo"] = {"type": _ADVANTAGE_TYPE_TO_ALGO_TYPE[advantage_type], **advantage}


def _migrate_multiplex(env: dict[str, Any], path: str) -> None:
    """An env's ``multiplex`` -> ``interception.multiplex``, which owns rollouts-per-server now.

    Not the identically named ``pool.multiplex``, which sizes env-server workers instead.
    """
    multiplex = env.pop("multiplex", None)
    if multiplex is None:
        return

    _deprecated(f"{path}.multiplex", f"{path}.interception.multiplex")
    interception = env.setdefault("interception", {})
    if not isinstance(interception, dict):
        raise ValueError(f"'{path}.interception' must be a table, got {type(interception).__name__}")
    if interception.setdefault("type", "elastic") != "elastic":
        raise ValueError(
            f"'{path}.multiplex' only applies to the elastic interception pool, but "
            f"'{path}.interception.type' is {interception['type']!r}"
        )
    interception.setdefault("multiplex", multiplex)


def _drop_max_retries(scope: dict[str, Any], path: str) -> None:
    """``max_retries`` never had an effect and is gone; per-env retries live on the env itself."""
    if scope.pop("max_retries", None) is not None:
        warnings.warn(
            f"'{path}.max_retries' is deprecated and had no effect. Ignoring it for now, but this will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )


def _migrate_envs(data: dict[str, Any]) -> None:
    for scope, path in (("train", "train"), ("eval", "eval")):
        group = data.get(scope)
        if not isinstance(group, dict):
            continue
        _drop_max_retries(group, path)
        envs = group.get("env")
        if not isinstance(envs, list):
            continue
        for index, env in enumerate(envs):
            if not isinstance(env, dict):
                continue
            _drop_max_retries(env, f"{path}.env[{index}]")
            _migrate_multiplex(env, f"{path}.env[{index}]")
            if scope == "train":
                _migrate_advantage(env)


def _migrate_client(data: dict[str, Any]) -> None:
    """Top-level ``[client]`` -> ``[model.client]``, the policy's own client."""
    client = data.pop("client", None)
    if client is None:
        return
    if not isinstance(client, dict):
        raise ValueError(f"'client' must be a table, got {type(client).__name__}")

    _deprecated("client", "model.client")
    model = data.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError(f"'model' must be a table, got {type(model).__name__}")
    canonical = model.setdefault("client", {})
    if not isinstance(canonical, dict):
        raise ValueError(f"'model.client' must be a table, got {type(canonical).__name__}")
    for key, value in client.items():
        canonical.setdefault(key, value)


def migrate_legacy_orchestrator_config(data: Any) -> Any:
    """Translate the pre-``algo`` orchestrator config keys in ``data``, in place.

    Runs as an ``OrchestratorConfig`` ``mode="before"`` validator, so it sees the
    merged TOML and CLI payload and covers ``model_validate`` callers (the hosted
    config validator) too. Legacy envs may still arrive under the deprecated
    top-level ``[[env]]``, so both env locations are migrated here rather than
    depending on validator ordering.
    """
    if not isinstance(data, dict):
        return data
    _migrate_algo(data)
    _migrate_client(data)
    _migrate_envs(data)
    for index, env in enumerate(data.get("env") or []):
        if isinstance(env, dict):
            _drop_max_retries(env, f"env[{index}]")
            _migrate_multiplex(env, f"env[{index}]")
            _migrate_advantage(env)
    return data
