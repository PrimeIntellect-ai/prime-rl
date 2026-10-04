from __future__ import annotations

from typing import Any


def propagate_shared_fields(data: Any) -> Any:
    """Propagate ``RLConfig``'s shared top-level fields into the matching sub-config
    dicts before sub-configs are constructed, so each sub-config's ``mode="after"``
    validators see the resolved values at construction time.

    Behaviour:
      - **Fill-if-absent**: an explicit sub-config value is never overwritten.
        The shared block acts as a default, not a stomper.
      - **Consistency mutex**: setting the same field at both the shared and
        sub-config level only raises when the values *disagree*. Matching
        values are accepted as a harmless no-op so that the materialized
        config (``model_dump`` writes both copies) round-trips through re-load.
        The original footgun the mutex was designed to catch — a sub-config
        value silently winning over a later CLI shared override — is still
        caught because that scenario produces *different* values.
    """
    if not isinstance(data, dict):
        return data

    def get(path: str) -> Any | None:
        node: Any = data
        for p in path.split("."):
            if not isinstance(node, dict) or p not in node:
                return None
            node = node[p]
        return node

    def fill(path: str, value: Any) -> None:
        parts = path.split(".")
        if parts[0] not in data or not isinstance(data[parts[0]], dict):
            return
        node = data
        for p in parts[:-1]:
            if not isinstance(node, dict):
                return
            node = node.setdefault(p, {})
        if isinstance(node, dict) and parts[-1] not in node:
            node[parts[-1]] = value

    conflicts: list[tuple[str, str]] = []

    def propagate(shared_path: str, *targets: str) -> None:
        """Verbatim shared → targets. Records *disagreeing* overlap into
        ``conflicts`` and fills each target if the shared value is set. Matching
        values are silently accepted so the materialized config round-trips
        through re-load.
        """
        value = get(shared_path)
        if value is None:
            return
        for target in targets:
            sub_value = get(target)
            if sub_value is not None and sub_value != value:
                conflicts.append((shared_path, target))
        for target in targets:
            fill(target, value)

    # [model] → trainer / orchestrator / inference.
    propagate(
        "model.name",
        "trainer.model.name",
        "inference.vllm.model",
        "orchestrator.model.name",
    )
    propagate(
        "model.vlm",
        "trainer.model.vlm",
        "orchestrator.model.vlm",
    )

    # [log]
    propagate("log.level", "trainer.log.level", "orchestrator.log.level", "inference.log.level")
    propagate(
        "log.json_logging",
        "trainer.log.json_logging",
        "orchestrator.log.json_logging",
        "inference.log.json_logging",
    )

    # [ckpt] leaves. (Bare empty ``[ckpt]`` block enablement is at the end.)
    # ``orchestrator.ckpt`` has no ``output_dir`` field — trainer-only.
    propagate("ckpt.output_dir", "trainer.ckpt.output_dir")
    propagate("ckpt.interval", "trainer.ckpt.interval", "orchestrator.ckpt.interval")
    propagate("ckpt.keep_last", "trainer.ckpt.keep_last", "orchestrator.ckpt.keep_last")
    propagate("ckpt.keep_interval", "trainer.ckpt.keep_interval", "orchestrator.ckpt.keep_interval")

    # [monitors.wandb] leaves. (Bare empty ``[monitors.wandb]`` block enablement is at the end.)
    # ``monitors.wandb.name`` flows verbatim to both sub-configs — shared W&B mode is
    # always on for the rl entrypoint, so the legacy ``-trainer`` /
    # ``-orchestrator`` suffix split is gone.
    for leaf in ("project", "entity", "name", "group", "tags", "offline"):
        propagate(
            f"monitors.wandb.{leaf}",
            f"trainer.monitors.wandb.{leaf}",
            f"orchestrator.monitors.wandb.{leaf}",
        )

    # [monitors.file] leaf. (Bare empty ``[monitors.file]`` block enablement is at the end.)
    propagate("monitors.file.path", "trainer.monitors.file.path", "orchestrator.monitors.file.path")

    # [monitors.prime] leaf → orchestrator only (the RL trainer has no platform
    # integration; the SFT trainer does, but it parses SFTConfig directly).
    propagate("monitors.prime.name", "orchestrator.monitors.prime.name")

    # [tokenizer]. ``chat_template`` also flows to ``inference.vllm`` (vLLM's
    # ``--chat-template``); ``name`` and ``trust_remote_code`` can legitimately
    # differ between sub-configs (auto-derived from model names, which may
    # differ for FP8-quantized inference variants).
    propagate("tokenizer.name", "trainer.tokenizer.name", "orchestrator.tokenizer.name")
    propagate(
        "tokenizer.trust_remote_code",
        "trainer.tokenizer.trust_remote_code",
        "orchestrator.tokenizer.trust_remote_code",
    )
    propagate(
        "tokenizer.chat_template",
        "trainer.tokenizer.chat_template",
        "orchestrator.tokenizer.chat_template",
        "inference.vllm.chat_template",
    )

    propagate("resume", "trainer.resume", "orchestrator.resume")

    # [rollout_transport] → both sub-configs (host is launcher-injected for zmq multi-node).
    propagate("rollout_transport", "trainer.rollout_transport", "orchestrator.rollout_transport")

    # Top-level scalars.
    propagate("max_steps", "trainer.max_steps", "orchestrator.max_steps")
    propagate("seq_len", "trainer.model.seq_len", "orchestrator.seq_len")

    # [slurm] → inference: a multi-node RL run drives its inference deployment under
    # the same SLURM allocation, so the nested inference inherits [slurm]. This is
    # what lets the nested InferenceConfig's multi-node / disaggregated SLURM check
    # pass (the per-rank inference.toml drops slurm, so each rank still runs locally).
    propagate("slurm", "inference.slurm")

    # Cascade trainer.tokenizer.chat_template → inference.vllm.chat_template
    # (vLLM ``--chat-template``). Read trainer's value *after* the shared
    # propagation above so we cover both:
    #   - shared ``[tokenizer] chat_template`` (already filled all three above,
    #     this re-fill is a no-op via fill-if-absent), and
    #   - ``[trainer.tokenizer] chat_template`` set directly without shared
    #     (only path that reaches inference; ``RLConfig.validate_shared_configs``
    #     would otherwise complain about the missing inference value).
    trainer_chat_template = get("trainer.tokenizer.chat_template")
    if trainer_chat_template is not None:
        fill("inference.vllm.chat_template", trainer_chat_template)

    # Bare ``[ckpt]`` / ``[monitors.wandb]`` block: presence-only signal that
    # enables the section with defaults on both sub-configs. Necessary because
    # ``trainer.ckpt`` / ``orchestrator.ckpt`` are Optional[None] by default —
    # without this an empty shared block would be a no-op. Leaf-level conflicts
    # (e.g. shared ``ckpt.interval`` vs ``trainer.ckpt.interval``) are already
    # caught above; the bare block is exempt because ``[ckpt]`` +
    # ``[trainer.ckpt] keep_last = 3`` is a legitimate "enable + customise per
    # side" pattern. CLI ``--no-ckpt`` / ``--no-monitors.wandb`` land as the
    # *string* ``"None"`` until ``BaseConfig``'s parent-class validator converts
    # it, which happens after this one — the disable must propagate too, since
    # the monitor sub-configs default to enabled.
    presence_targets = {
        "ckpt": ("trainer", "orchestrator"),
        "monitors.wandb": ("trainer", "orchestrator"),
        "monitors.file": ("trainer", "orchestrator"),
        "monitors.prime": ("orchestrator",),
    }
    for key, targets in presence_targets.items():
        value = get(key)
        if isinstance(value, dict):
            for target in targets:
                fill(f"{target}.{key}", {})
        elif value == "None":
            for target in targets:
                fill(f"{target}.{key}", "None")

    if conflicts:
        lines = [
            "Shared config conflicts with matching sub-config field(s). Pick one place "
            "to set each value — duplicating it is ambiguous and the sub-config "
            "would silently shadow any later shared-level override (e.g. on the CLI):",
        ]
        for shared, sub in conflicts:
            lines.append(f"  - [{shared!r}] is set, but [{sub!r}] is also set")
        raise ValueError("\n".join(lines))

    return data
