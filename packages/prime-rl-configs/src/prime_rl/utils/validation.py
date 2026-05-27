from __future__ import annotations

from typing import Any, Optional

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig


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
      - **Aliased sub-paths**: ``orchestrator.model.*`` is checked against its
        ``orchestrator.student.model.*`` alias (and vice versa), so the
        conflict fires regardless of which spelling the user wrote.
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

    def propagate(shared_path: str, *targets: str, aliases: tuple[str, ...] = ()) -> None:
        """Verbatim shared → targets. Records *disagreeing* overlap (incl. alias
        spellings) into ``conflicts`` and fills each target if the shared value
        is set. Matching values are silently accepted so the materialized
        config round-trips through re-load.
        """
        value = get(shared_path)
        if value is None:
            return
        for sub in (*targets, *aliases):
            sub_value = get(sub)
            if sub_value is not None and sub_value != value:
                conflicts.append((shared_path, sub))
        for target in targets:
            fill(target, value)

    # [model] → trainer / orchestrator (student, via AliasChoices) / inference.
    propagate(
        "model.name",
        "trainer.model.name",
        "inference.model.name",
        "orchestrator.model.name",
        aliases=("orchestrator.student.model.name",),
    )
    propagate(
        "model.vlm",
        "trainer.model.vlm",
        "inference.model.vlm",
        "orchestrator.model.vlm",
        aliases=("orchestrator.student.model.vlm",),
    )

    # [log]
    propagate("log.level", "trainer.log.level", "orchestrator.log.level")
    propagate("log.json_logging", "trainer.log.json_logging", "orchestrator.log.json_logging")

    # [ckpt] leaves. (Bare empty ``[ckpt]`` block enablement is at the end.)
    # ``orchestrator.ckpt`` has no ``output_dir`` field — trainer-only.
    propagate("ckpt.output_dir", "trainer.ckpt.output_dir")
    propagate("ckpt.interval", "trainer.ckpt.interval", "orchestrator.ckpt.interval")
    propagate("ckpt.resume_step", "trainer.ckpt.resume_step", "orchestrator.ckpt.resume_step")
    propagate("ckpt.keep_last", "trainer.ckpt.keep_last", "orchestrator.ckpt.keep_last")
    propagate("ckpt.keep_interval", "trainer.ckpt.keep_interval", "orchestrator.ckpt.keep_interval")

    # [wandb] leaves. (Bare empty ``[wandb]`` block enablement is at the end.)
    propagate("wandb.project", "trainer.wandb.project", "orchestrator.wandb.project")
    propagate("wandb.entity", "trainer.wandb.entity", "orchestrator.wandb.entity")
    propagate("wandb.group", "trainer.wandb.group", "orchestrator.wandb.group")
    propagate("wandb.tags", "trainer.wandb.tags", "orchestrator.wandb.tags")
    propagate("wandb.offline", "trainer.wandb.offline", "orchestrator.wandb.offline")

    # wandb.name: in non-shared mode the two sub-configs get distinct
    # ``-trainer`` / ``-orchestrator`` suffixes so the W&B runs are
    # distinguishable. ``OrchestratorConfig.auto_setup_prime_monitor_run_name``
    # then defaults prime_monitor.run_name to the (unsuffixed) value.
    # Conflicts are reported against the *transformed* sub-config value, not
    # the raw shared name, so the materialized config round-trips cleanly.
    wandb_name = get("wandb.name")
    if wandb_name is not None:
        non_shared = get("wandb.shared") is False
        expected = {
            "trainer.wandb.name": f"{wandb_name}-trainer" if non_shared else wandb_name,
            "orchestrator.wandb.name": f"{wandb_name}-orchestrator" if non_shared else wandb_name,
        }
        for sub, expected_value in expected.items():
            sub_value = get(sub)
            if sub_value is not None and sub_value != expected_value:
                conflicts.append(("wandb.name", sub))
            fill(sub, expected_value)

    # [tokenizer]. ``chat_template`` also flows to ``inference.model`` (vLLM's
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
        "inference.model.chat_template",
    )

    # Top-level scalars.
    propagate("max_steps", "trainer.max_steps", "orchestrator.max_steps")
    propagate("seq_len", "trainer.model.seq_len", "orchestrator.seq_len")

    # output_dir: orchestrator gets a ``/run_default`` subdir so trainer +
    # orchestrator nest under the same experiment root without colliding.
    # Conflicts are reported against the *transformed* sub-config value, not
    # the raw shared path, so the materialized config round-trips cleanly.
    output_dir = get("output_dir")
    if output_dir is not None:
        expected = {
            "trainer.output_dir": output_dir,
            "orchestrator.output_dir": f"{output_dir}/run_default",
        }
        for sub, expected_value in expected.items():
            sub_value = get(sub)
            if sub_value is not None and sub_value != expected_value:
                conflicts.append(("output_dir", sub))
            fill(sub, expected_value)

    # Cascade trainer.tokenizer.chat_template → inference.model.chat_template
    # (vLLM ``--chat-template``). Read trainer's value *after* the shared
    # propagation above so we cover both:
    #   - shared ``[tokenizer] chat_template`` (already filled all three above,
    #     this re-fill is a no-op via fill-if-absent), and
    #   - ``[trainer.tokenizer] chat_template`` set directly without shared
    #     (only path that reaches inference; ``validate_shared_tokenizer``
    #     would otherwise complain about the missing inference value).
    trainer_chat_template = get("trainer.tokenizer.chat_template")
    if trainer_chat_template is not None:
        fill("inference.model.chat_template", trainer_chat_template)

    # Bare ``[ckpt]`` / ``[wandb]`` block: presence-only signal that enables
    # the section with defaults on both sub-configs. Necessary because
    # ``trainer.ckpt`` / ``orchestrator.ckpt`` are Optional[None] by default —
    # without this an empty shared block would be a no-op. Leaf-level
    # conflicts (e.g. shared ``ckpt.interval`` vs ``trainer.ckpt.interval``)
    # are already caught above; the bare block is exempt because
    # ``[ckpt]`` + ``[trainer.ckpt] keep_last = 3`` is a legitimate
    # "enable + customise per side" pattern.
    for key in ("ckpt", "wandb"):
        if get(key) is not None:
            fill(f"trainer.{key}", {})
            fill(f"orchestrator.{key}", {})

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


def validate_shared_ckpt_config(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.ckpt and not orchestrator.ckpt:
        raise ValueError(
            "Trainer checkpoint config is specified, but orchestrator checkpoint config is not. Please setup checkpointing on both for checkpointing to work properly."
        )
    if orchestrator.ckpt and not trainer.ckpt:
        raise ValueError(
            "Orchestrator checkpoint config is specified, but trainer checkpoint config is not. Please setup checkpointing on both for checkpointing to work properly."
        )
    if trainer.ckpt and orchestrator.ckpt and trainer.ckpt.interval != orchestrator.ckpt.interval:
        raise ValueError(
            f"Trainer checkpoint interval ({trainer.ckpt.interval}) and orchestrator checkpoint interval ({orchestrator.ckpt.interval}) are not the same. Please specify the same checkpoint interval for both."
        )
    if trainer.ckpt and orchestrator.ckpt and trainer.ckpt.resume_step != orchestrator.ckpt.resume_step:
        raise ValueError(
            f"Trainer checkpoint resume step ({trainer.ckpt.resume_step}) and orchestrator checkpoint resume step ({orchestrator.ckpt.resume_step}) are not the same. Please specify the same checkpoint resume step for both."
        )


def validate_shared_model_name(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
    inference: Optional[InferenceConfig] = None,
) -> None:
    # Orchestrator must match inference (it queries the inference server)
    if inference is not None:
        if inference.model.name != orchestrator.student.model.name:
            raise ValueError(
                f"Inference model name ({inference.model.name}) and orchestrator model name ({orchestrator.student.model.name}) are not the same. "
                "The orchestrator queries the inference server and must use the same model name."
            )
        return

    if trainer.model.name.startswith("Jackmin108/"):  # The TT MoE models will have a different name on the orchestrator
        return
    if trainer.model.name != orchestrator.student.model.name:
        raise ValueError(
            f"Trainer model name ({trainer.model.name}) and orchestrator model name ({orchestrator.student.model.name}) are not the same. Please specify the same model name for both."
        )


def validate_shared_output_dir(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.output_dir != orchestrator.output_dir.parent:
        raise ValueError(
            f"Trainer outputs directory ({trainer.output_dir}) and orchestrator outputs directory parent ({orchestrator.output_dir.parent}) are not the same. Please specify the same outputs directory for both."
        )


def validate_shared_wandb_config(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.wandb and not orchestrator.wandb:
        raise ValueError(
            "Trainer W&B config is specified, but orchestrator W&B config is not. "
            "This means only trainer metrics will be logged. Please specify [orchestrator.wandb] to log orchestrator metrics as well, "
            "or use [wandb] to configure both at once."
        )
    if orchestrator.wandb and not trainer.wandb:
        raise ValueError(
            "Orchestrator W&B config is specified, but trainer W&B config is not. "
            "This means only orchestrator metrics will be logged. Please specify [trainer.wandb] to log trainer metrics as well, "
            "or use [wandb] to configure both at once."
        )
    if trainer.wandb and orchestrator.wandb:
        if trainer.wandb.project != orchestrator.wandb.project:
            raise ValueError(
                f"Trainer W&B project ({trainer.wandb.project}) and orchestrator W&B project ({orchestrator.wandb.project}) are not the same. Please specify the same W&B project for both."
            )


def validate_shared_max_steps(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.max_steps != orchestrator.max_steps:
        raise ValueError(
            f"Trainer max steps ({trainer.max_steps}) and orchestrator max steps ({orchestrator.max_steps}) are not the same. Please specify the same max steps for both."
        )


def validate_shared_seq_len(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.model.seq_len < orchestrator.seq_len:
        raise ValueError(
            f"Trainer model seq_len ({trainer.model.seq_len}) must be >= orchestrator seq_len ({orchestrator.seq_len}). "
            f"The trainer needs to be able to handle sequences at least as long as those produced by the orchestrator."
        )


def validate_shared_tokenizer(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
    inference: Optional[InferenceConfig] = None,
) -> None:
    # Validate chat_template is consistent across all components.
    # We only check chat_template (not name/trust_remote_code) because those
    # are auto-derived from model names which may legitimately differ (e.g.
    # when inference uses an FP8 quantized variant of the same model).
    if trainer.tokenizer.chat_template != orchestrator.tokenizer.chat_template:
        raise ValueError(
            f"Trainer chat_template ({trainer.tokenizer.chat_template!r}) and orchestrator "
            f"chat_template ({orchestrator.tokenizer.chat_template!r}) do not match. "
            f"Use the shared [tokenizer] config to set chat_template for both."
        )
    if inference is not None:
        if trainer.tokenizer.chat_template != inference.model.chat_template:
            raise ValueError(
                f"Inference chat_template ({inference.model.chat_template!r}) does not match "
                f"the shared tokenizer chat_template ({trainer.tokenizer.chat_template!r}). "
                f"Use the shared [tokenizer] config to set chat_template for all components."
            )


def validate_shared_weight_broadcast(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
    inference: Optional[InferenceConfig] = None,
) -> None:
    if (
        inference
        and trainer.weight_broadcast.type != orchestrator.weight_broadcast.type != inference.weight_broadcast.type
    ):
        raise ValueError(
            f"Inference weight broadcast type ({inference.weight_broadcast.type}) and orchestrator weight broadcast type ({orchestrator.weight_broadcast.type}) are not the same. Please specify the same weight broadcast type for both."
        )
    elif trainer.weight_broadcast.type != orchestrator.weight_broadcast.type:
        raise ValueError(
            f"Trainer weight broadcast type ({trainer.weight_broadcast.type}) and orchestrator weight broadcast type ({orchestrator.weight_broadcast.type}) are not the same. Please specify the same weight broadcast type for both."
        )
