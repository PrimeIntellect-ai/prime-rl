"""Translation of legacy config keys into the current shape.

Hosted training's control plane and deployment charts emit older generations of
config keys. The pre-``algo`` run shape: a top-level ``training_mode`` with a
sibling ``[teacher]`` block, per-env ``advantage``, and the policy client at
``[client]``. The pre-0.3.0 flat env shape: ``[[train.env]]`` / ``[[eval.env]]``
entries whose verifiers knobs (``taskset``, ``harness``, ``pool``, ``timeout``,
token caps, the v0 ``id``/``args``) sat flat on the entry instead of composing
the ``env``/``serve``/``legacy`` blocks. And the pre-vllm-block inference shape:
``[model]``/``[parallel]`` blocks, flat engine args, and a ``vllm_extra`` dict
instead of the ``vllm`` pass-through block. This module maps all of it onto the
current shape at validation time so callers that haven't been updated yet keep
working against a current image.

Split by scope, because an env block is validated through two different roots:
the orchestrator's ``[[train.source]]`` / ``[[eval.source]]`` entries *and* the
standalone env server's config. Source-scoped keys are migrated on the source
``EnvConfig`` itself; the env server re-homes its old nested ``[env]`` block
through the same translation.

Temporary. Delete this module and its validators once every caller emits the
current shape directly.
"""

import json
import warnings
from typing import Any

# `training_mode` named the shape of the whole run; `algo.type` names it now.
_TRAINING_MODE_TO_ALGO_TYPE = {"rl": "grpo", "opd": "opd", "sft": "sft"}

# The `default` advantage was GRPO's group-mean baseline; `custom` has no successor.
_ADVANTAGE_TYPE_TO_ALGO_TYPE = {"default": "grpo"}

# Flat per-run caps that moved onto the env's agent seat.
_FLAT_AGENT_KEYS = ("max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens")

# Inference engine args that sat flat on the old InferenceConfig root and moved
# under the ``vllm`` block (same name there).
_FLAT_INFERENCE_VLLM_KEYS = (
    "enable_lora",
    "max_loras",
    "max_cpu_loras",
    "max_lora_rank",
    "lora_target_modules",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "quantization",
    "api_server_count",
    "data_parallel_size_local",
    "data_parallel_rpc_port",
    "seed",
    "enable_expert_parallel",
    "enable_eplb",
    "enable_dbo",
    "enable_return_routed_experts",
)

# Keys that only ever existed on the flat pre-0.3.0 env shape — any of them marks an
# old-shaped block. ``id``/``taskset``/``timeout``/``retries`` are ambiguous (the
# composed verifiers env block has them too), so they don't qualify on their own.
_FLAT_ENV_MARKERS = (
    "name",
    "harness",
    "pool",
    "num_workers",
    "address",
    "multiplex",
    "max_retries",
    "args",
    "extra_env_kwargs",
    "ratio",
    *_FLAT_AGENT_KEYS,
)


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

    _deprecated("advantage", "algo")
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


def _drop_max_retries(scope: dict[str, Any], path: str) -> None:
    """``max_retries`` never had an effect and is gone; per-env retries live on the env itself."""
    if scope.pop("max_retries", None) is not None:
        warnings.warn(
            f"'{path}.max_retries' is deprecated and had no effect. Ignoring it for now, but this will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )


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


def _migrate_max_tokens(sampling: Any, path: str) -> None:
    """A sampling table's ``max_tokens`` -> ``max_completion_tokens`` (which wins when both are set)."""
    if isinstance(sampling, dict) and "max_tokens" in sampling:
        _deprecated(f"{path}.max_tokens", f"{path}.max_completion_tokens")
        sampling.setdefault("max_completion_tokens", sampling.pop("max_tokens"))


def _migrate_rollouts_per_example(scope: dict[str, Any], path: str) -> None:
    """``rollouts_per_example`` -> ``group_size`` (which wins when both are set)."""
    if "rollouts_per_example" in scope:
        _deprecated(f"{path}.rollouts_per_example", f"{path}.group_size")
        scope.setdefault("group_size", scope.pop("rollouts_per_example"))


def _sub_table(data: dict[str, Any], block: str) -> dict[str, Any]:
    sub = data.setdefault(block, {})
    if not isinstance(sub, dict):
        raise ValueError(f"'{block}' must be a table, got {type(sub).__name__}")
    return sub


def _migrate_multiplex(data: dict[str, Any]) -> None:
    """An env's flat ``multiplex`` -> ``env.interception.multiplex``, which owns rollouts-per-server.

    Not the identically named ``serve.pool.multiplex``, which sizes env-server workers instead.
    """
    multiplex = data.pop("multiplex", None)
    if multiplex is None:
        return

    _deprecated("multiplex", "env.interception.multiplex")
    interception = _sub_table(_sub_table(data, "env"), "interception")
    if interception.setdefault("type", "elastic") != "elastic":
        raise ValueError(
            "'multiplex' only applies to the elastic interception pool, but "
            f"'interception.type' is {interception['type']!r}"
        )
    interception.setdefault("multiplex", multiplex)


def _migrate_flat_env_keys(data: dict[str, Any]) -> None:
    """Move the flat pre-0.3.0 env keys into the composed blocks, in place: what runs
    onto ``env`` (per-run caps onto its ``agent`` seat), hosting onto ``serve``, the
    classic v0 fields onto ``legacy``. An explicitly set composed key wins."""
    if "taskset" in data:
        _deprecated("taskset", "env.taskset")
        _sub_table(data, "env").setdefault("taskset", data.pop("taskset"))
    if "harness" in data:
        _deprecated("harness", "env.agent.harness")
        harness = data.pop("harness")
        agent = _sub_table(_sub_table(data, "env"), "agent")
        # The runtime moved off the harness onto the agent seat that provisions it.
        if isinstance(harness, dict) and "runtime" in harness:
            agent.setdefault("runtime", harness.pop("runtime"))
        agent.setdefault("harness", harness)
    for key in _FLAT_AGENT_KEYS:
        if key in data:
            _deprecated(key, f"env.agent.{key}")
            _sub_table(_sub_table(data, "env"), "agent").setdefault(key, data.pop(key))
    if "timeout" in data:
        # The flat timeout's stages (setup/rollout/finalize/scoring) bound the agent's
        # run; the composed env-level timeout has different fields (episode/finalize).
        _deprecated("timeout", "env.agent.timeout")
        _sub_table(_sub_table(data, "env"), "agent").setdefault("timeout", data.pop("timeout"))
    if "retries" in data:
        _deprecated("retries", "env.retries")
        _sub_table(data, "env").setdefault("retries", data.pop("retries"))
    _migrate_multiplex(data)
    if "pool" in data:
        _deprecated("pool", "serve.pool")
        _sub_table(data, "serve").setdefault("pool", data.pop("pool"))
    if "num_workers" in data:
        # An int becomes a fixed static pool, "auto" falls through to the default
        # elastic pool. An explicit pool always wins.
        num_workers = data.pop("num_workers")
        _deprecated("num_workers", "serve.pool")
        serve = _sub_table(data, "serve")
        if "pool" not in serve and num_workers != "auto":
            serve["pool"] = {"type": "static", "num_workers": num_workers}
    if "address" in data:
        _deprecated("address", "serve.address")
        _sub_table(data, "serve").setdefault("address", data.pop("address"))
    if "args" in data:
        _deprecated("args", "legacy.args")
        _sub_table(data, "legacy").setdefault("args", data.pop("args"))
    if "extra_env_kwargs" in data:
        _deprecated("extra_env_kwargs", "legacy.extra_env_kwargs")
        _sub_table(data, "legacy").setdefault("extra_env_kwargs", data.pop("extra_env_kwargs"))
    if "id" in data:
        # A flat `id` was the classic (v0) env id, decorative next to a v1 taskset
        # (the taskset won); the composed shape spells a v1 env pairing `env.id`.
        env_id = data.pop("id")
        env = data.get("env")
        taskset = env.get("taskset") if isinstance(env, dict) else None
        taskset_id = taskset.get("id") if isinstance(taskset, dict) else getattr(taskset, "id", None)
        if taskset_id:
            warnings.warn(
                f"'id' = {env_id!r} is ignored next to the v1 taskset {taskset_id!r} — the taskset wins, "
                "as it always has. Drop the 'id' key.",
                FutureWarning,
                stacklevel=2,
            )
        else:
            _deprecated("id", "legacy.id")
            _sub_table(data, "legacy").setdefault("id", env_id)


def migrate_legacy_orchestrator_config(data: Any) -> Any:
    """Translate the run-scoped legacy keys in an orchestrator config, in place.

    Runs as an ``OrchestratorConfig`` ``mode="before"`` validator, so it sees the
    merged TOML and CLI payload and covers ``model_validate`` callers (the hosted
    config validator) too. Source-scoped keys are handled by the source configs' own
    validators, which also reach the standalone env server.
    """
    if not isinstance(data, dict):
        return data
    _migrate_algo(data)
    _migrate_client(data)
    if "env" in data:
        _deprecated("env", "train.source")
        train = data.setdefault("train", {})
        if isinstance(train, dict):
            train.setdefault("source", data.pop("env"))
    if "sampling" in data:
        _deprecated("sampling", "train.sampling")
        train = data.setdefault("train", {})
        if isinstance(train, dict):
            train.setdefault("sampling", data.pop("sampling"))
    if "max_inflight_rollouts" in data:
        _deprecated("max_inflight_rollouts", "max_inflight_episodes")
        data.setdefault("max_inflight_episodes", data.pop("max_inflight_rollouts"))
    _migrate_rollouts_per_example(data, "orchestrator")
    for scope in ("train", "eval"):
        group = data.get(scope)
        if isinstance(group, dict):
            _drop_max_retries(group, scope)
            if "env" in group:
                _deprecated(f"{scope}.env", f"{scope}.source")
                group.setdefault("source", group.pop("env"))
            _migrate_max_tokens(group.get("sampling"), f"{scope}.sampling")
            if scope == "eval":
                _migrate_rollouts_per_example(group, "eval")
    return data


def migrate_legacy_env_config(data: Any) -> Any:
    """Translate the legacy keys on a single source entry, in place.

    Runs as the source ``EnvConfig`` ``mode="before"`` validator (before the ``env``
    field is narrowed), so it covers the orchestrator's ``[[train.source]]`` /
    ``[[eval.source]]`` entries; the standalone env server routes its old nested
    ``[env]`` block through it via ``migrate_legacy_env_server_config``.
    """
    if not isinstance(data, dict):
        return data
    _drop_max_retries(data, "env")
    _migrate_rollouts_per_example(data, "env")
    _migrate_max_tokens(data.get("sampling"), "env.sampling")
    _migrate_flat_env_keys(data)
    return data


def migrate_legacy_env_server_config(data: Any) -> Any:
    """Re-home the env server's old source-shaped ``[env]`` block, in place.

    The old entrypoint nested everything under ``[env]``; now ``[env]`` is the
    verifiers env block with ``[serve]`` and ``[legacy]`` as siblings. A block is
    old-shaped when it carries a key only the flat shape had, or a bare v0 ``id``
    with no v1 taskset — hosted's control plane never pairs a v1 env by id.
    """
    if not isinstance(data, dict):
        return data
    env = data.get("env")
    if not isinstance(env, dict):
        return data
    taskset = env.get("taskset")
    taskset_id = taskset.get("id") if isinstance(taskset, dict) else None
    bare_v0_id = "id" in env and not taskset_id and "agent" not in env and "max_concurrent_agents" not in env
    if not (any(key in env for key in _FLAT_ENV_MARKERS) or bare_v0_id):
        return data
    migrate_legacy_env_config(env)
    # Orchestration-only labels the old env server parsed and ignored.
    env.pop("name", None)
    env.pop("ratio", None)
    for block in ("serve", "legacy"):
        sub = env.pop(block, None)
        if isinstance(sub, dict):
            root = data.setdefault(block, {})
            if isinstance(root, dict):
                for key, value in sub.items():
                    root.setdefault(key, value)
    inner = env.pop("env", None)
    if isinstance(inner, dict):
        for key, value in inner.items():
            env.setdefault(key, value)
    return data


def migrate_legacy_inference_config(data: Any) -> Any:
    """Translate the flat pre-vllm-block inference shape, in place.

    Hosted's deployment charts still start the inference server with the old
    spellings: a ``[model]`` block (``--model.name``, ``--model.max-model-len``, …),
    ``[parallel]`` (``tp``/``dp``), flat engine args (``--enable-lora``,
    ``--gpu-memory-utilization``, …), and a ``--vllm-extra`` JSON dict. All of it
    lands on the ``vllm`` block, which forwards unknown keys to vLLM verbatim.
    An explicitly set ``vllm.*`` key wins over a translated one; ``vllm_extra``
    overrides, matching its old apply-after-config semantics.
    """
    if not isinstance(data, dict):
        return data
    translated: dict[str, Any] = {}
    model = data.pop("model", None)
    if model is not None:
        _deprecated("model", "vllm")
        if isinstance(model, dict):
            for key, value in model.items():
                translated["model" if key == "name" else key] = value
        else:
            translated["model"] = model
    parallel = data.pop("parallel", None)
    if parallel is not None:
        _deprecated("parallel", "vllm.tensor_parallel_size / vllm.data_parallel_size")
        if not isinstance(parallel, dict):
            raise ValueError(f"'parallel' must be a table, got {type(parallel).__name__}")
        parallel = dict(parallel)
        if "tp" in parallel:
            translated["tensor_parallel_size"] = parallel.pop("tp")
        if "dp" in parallel:
            translated["data_parallel_size"] = parallel.pop("dp")
        if parallel:
            raise ValueError(f"'parallel' only carried 'tp' and 'dp', got {sorted(parallel)}")
    for key in _FLAT_INFERENCE_VLLM_KEYS:
        if key in data:
            _deprecated(key, f"vllm.{key}")
            translated[key] = data.pop(key)
    extra = data.pop("vllm_extra", None)
    if isinstance(extra, str):
        extra = json.loads(extra)
    if extra is not None:
        _deprecated("vllm_extra", "vllm.<arg>")
        if not isinstance(extra, dict):
            raise ValueError(f"'vllm_extra' must be a table, got {type(extra).__name__}")
    if translated or extra:
        vllm = _sub_table(data, "vllm")
        for key, value in translated.items():
            vllm.setdefault(key, value)
        if extra:
            vllm.update(extra)
    return data


def migrate_legacy_train_env_config(data: Any) -> Any:
    """Translate a train source's ``advantage``, in place.

    Separate from ``migrate_legacy_env_config`` because ``algo`` only exists on
    train sources — translating an ``advantage`` on an eval source or on the env
    server would just trade one rejected key for another.
    """
    if not isinstance(data, dict):
        return data
    _migrate_advantage(data)
    return data
