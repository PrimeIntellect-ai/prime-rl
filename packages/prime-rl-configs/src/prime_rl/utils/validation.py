from __future__ import annotations

from typing import Any, Optional

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig


# Per-field shared ↔ sub-config mappings used by ``validate_no_shared_field_conflicts``.
# Each entry: "shared.dotted.path" -> [list of sub-config paths it propagates to].
#
# Setting both a shared field and any of its targets is a config conflict — the
# sub-config silently wins under fill-if-absent semantics, which is exactly the
# silent-no-op class of bugs from #2430. The mutex check below raises instead.
#
# Where a sub-config field is reachable via multiple keys (alias), all variants
# are listed so the check catches the conflict regardless of which form the user
# wrote (e.g. ``[orchestrator.model]`` and ``[orchestrator.student.model]`` are
# the same field, exposed via ``AliasChoices("student", "model")``).
_SHARED_TO_SUB_FIELDS: dict[str, tuple[str, ...]] = {
    # [model] → trainer / orchestrator (student) / inference
    "model.name": (
        "trainer.model.name",
        "inference.model.name",
        "orchestrator.model.name",
        "orchestrator.student.model.name",
    ),
    "model.vlm": (
        "trainer.model.vlm",
        "inference.model.vlm",
        "orchestrator.model.vlm",
        "orchestrator.student.model.vlm",
    ),
    # [log] → trainer / orchestrator
    "log.level": ("trainer.log.level", "orchestrator.log.level"),
    "log.json_logging": ("trainer.log.json_logging", "orchestrator.log.json_logging"),
    # [ckpt] (field-level only; presence of an empty shared [ckpt] block is
    # explicitly allowed — it just enables ckpt on both sub-configs).
    "ckpt.output_dir": ("trainer.ckpt.output_dir",),  # orchestrator.ckpt has no output_dir
    "ckpt.interval": ("trainer.ckpt.interval", "orchestrator.ckpt.interval"),
    "ckpt.resume_step": ("trainer.ckpt.resume_step", "orchestrator.ckpt.resume_step"),
    "ckpt.keep_last": ("trainer.ckpt.keep_last", "orchestrator.ckpt.keep_last"),
    "ckpt.keep_interval": ("trainer.ckpt.keep_interval", "orchestrator.ckpt.keep_interval"),
    # [wandb] (same: empty shared [wandb] block is allowed, only field-level
    # collisions are forbidden).
    "wandb.project": ("trainer.wandb.project", "orchestrator.wandb.project"),
    "wandb.offline": ("trainer.wandb.offline", "orchestrator.wandb.offline"),
    "wandb.name": ("trainer.wandb.name", "orchestrator.wandb.name"),
    # [tokenizer]
    "tokenizer.name": ("trainer.tokenizer.name", "orchestrator.tokenizer.name"),
    "tokenizer.trust_remote_code": (
        "trainer.tokenizer.trust_remote_code",
        "orchestrator.tokenizer.trust_remote_code",
    ),
    "tokenizer.chat_template": (
        "trainer.tokenizer.chat_template",
        "orchestrator.tokenizer.chat_template",
    ),
    # Top-level scalars
    "max_steps": ("trainer.max_steps", "orchestrator.max_steps"),
    "max_async_level": ("trainer.max_async_level", "orchestrator.max_async_level"),
    "output_dir": ("trainer.output_dir", "orchestrator.output_dir"),
    "seq_len": ("trainer.model.seq_len", "orchestrator.seq_len"),
}


def _get_dotted(data: Any, path: str) -> Any:
    """Walk a raw config dict by dotted path; return None if any segment is absent."""
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def validate_no_shared_field_conflicts(data: Any) -> None:
    """Raise if any shared top-level field is also set on a matching sub-config field.

    Shared fields (``[model] name``, ``seq_len``, ``[ckpt] interval``, etc.)
    propagate down to sub-configs via fill-if-absent semantics in
    ``RLConfig.auto_setup_shared_configs``. If the user writes the same field
    in both places, the sub-config silently wins — and any later CLI override
    of the shared field is also silently shadowed. That is the bug from #2430.

    This check forbids the overlap entirely: pick one place to express each
    setting. Operates on the raw input dict before sub-configs are built so it
    can tell user-set entries apart from validator-filled ones.

    Aliased sub-config paths are checked under all their accepted keys (e.g.
    both ``orchestrator.model.name`` and ``orchestrator.student.model.name``).
    """
    if not isinstance(data, dict):
        return
    conflicts: list[tuple[str, str]] = []
    for shared_path, sub_paths in _SHARED_TO_SUB_FIELDS.items():
        if _get_dotted(data, shared_path) is None:
            continue
        for sub_path in sub_paths:
            if _get_dotted(data, sub_path) is not None:
                conflicts.append((shared_path, sub_path))
    if not conflicts:
        return
    lines = [
        "Shared config conflicts with matching sub-config field(s). Pick one place "
        "to set each value — duplicating it is ambiguous and the sub-config "
        "would silently shadow any later shared-level override (e.g. on the CLI):",
    ]
    for shared, sub in conflicts:
        lines.append(f"  - [{shared!r}] is set, but [{sub!r}] is also set")
    raise ValueError("\n".join(lines))


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


def validate_shared_max_async_level(
    trainer: TrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.max_async_level != orchestrator.max_async_level:
        raise ValueError(
            f"Trainer max async level ({trainer.max_async_level}) and orchestrator max async level ({orchestrator.max_async_level}) are not the same. Please specify the same max async level for both."
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
