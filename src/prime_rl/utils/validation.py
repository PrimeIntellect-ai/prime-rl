from __future__ import annotations

from typing import Optional

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.rl.config import RLTrainerConfig


def validate_shared_ckpt_config(
    trainer: RLTrainerConfig,
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
    trainer: RLTrainerConfig,
    orchestrator: OrchestratorConfig,
    inference: Optional[InferenceConfig] = None,
) -> None:
    if trainer.model.name.startswith("Jackmin108/"):  # The TT MoE models will have a different name on the orchestrator
        return
    if trainer.model.name != orchestrator.model.name:
        raise ValueError(
            f"Trainer model name ({trainer.model.name}) and orchestrator model name ({orchestrator.model.name}) are not the same. Please specify the same model name for both."
        )

    if inference and inference.model.name != orchestrator.model.name:
        raise ValueError(
            f"Inference model name ({inference.model.name}) and orchestrator model name ({orchestrator.model.name}. Please specify the same model name for both."
        )


def validate_shared_max_model_len(
    orchestrator: OrchestratorConfig,
    inference: Optional[InferenceConfig] = None,
) -> None:
    if inference and inference.model.max_model_len and orchestrator.seq_len != inference.model.max_model_len:
        raise ValueError(
            f"Orchestrator sequence length ({orchestrator.seq_len}) and inference model max model length ({inference.model.max_model_len}) are not the same. Please specify the same max model length for both."
        )


def validate_shared_output_dir(
    trainer: RLTrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.output_dir != orchestrator.output_dir:
        raise ValueError(
            f"Trainer outputs directory ({trainer.output_dir}) and orchestrator outputs directory ({orchestrator.output_dir}) are not the same. Please specify the same outputs directory for both."
        )


def validate_shared_wandb_config(
    trainer: RLTrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.wandb and orchestrator.wandb:
        if trainer.wandb.project != orchestrator.wandb.project:
            raise ValueError(
                f"Trainer W&B project ({trainer.wandb.project}) and orchestrator W&B project ({orchestrator.wandb.project}) are not the same. Please specify the same W&B project for both."
            )


def validate_shared_max_steps(
    trainer: RLTrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.max_steps != orchestrator.max_steps:
        raise ValueError(
            f"Trainer max steps ({trainer.max_steps}) and orchestrator max steps ({orchestrator.max_steps}) are not the same. Please specify the same max steps for both."
        )


def validate_shared_async_level(
    trainer: RLTrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    if trainer.async_level != orchestrator.async_level:
        raise ValueError(
            f"Trainer async level ({trainer.async_level}) and orchestrator async level ({orchestrator.async_level}) are not the same. Please specify the same async level for both."
        )
