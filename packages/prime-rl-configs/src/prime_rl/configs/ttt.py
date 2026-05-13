from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig


class TTTLoRAConfig(BaseConfig):
    """LoRA parameters for test-time-training adapters."""

    rank: Annotated[
        Literal[8, 16, 32, 64, 128, 256, 320, 512],
        Field(description="Rank for the per-rollout prompt and completion TTT LoRAs."),
    ] = 8

    alpha: Annotated[
        int,
        Field(gt=0, description="LoRA alpha for the per-rollout prompt and completion TTT LoRAs."),
    ] = 16

    dropout: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Dropout used while training TTT LoRAs."),
    ] = 0.0

    target_modules: Annotated[
        list[str],
        Field(description="Module patterns targeted by the TTT LoRAs."),
    ] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class TTTOptimizerConfig(BaseConfig):
    """Optimizer parameters for online TTT adapter updates."""

    type: Annotated[
        Literal["adamw"],
        Field(description="Optimizer used for TTT adapter updates."),
    ] = "adamw"

    lr: Annotated[
        float,
        Field(gt=0.0, description="Learning rate for TTT adapter updates."),
    ] = 1e-4

    weight_decay: Annotated[
        float,
        Field(ge=0.0, description="Weight decay for TTT adapter updates."),
    ] = 0.0

    max_grad_norm: Annotated[
        float | None,
        Field(gt=0.0, description="Optional gradient clipping norm for TTT adapter updates."),
    ] = 1.0

    steps_per_update: Annotated[
        int,
        Field(ge=1, description="Optimizer steps to take for each TTT update window."),
    ] = 1


class TTTConfig(BaseConfig):
    """Configuration for per-rollout test-time training during RL rollouts."""

    enabled: Annotated[
        bool,
        Field(description="Enable TTT-aware rollout/training plumbing."),
    ] = False

    window_seq_len: Annotated[
        int,
        Field(gt=0, description="Physical attention window used by trainer and inference."),
    ] = 8192

    total_seq_len: Annotated[
        int,
        Field(gt=0, description="Logical rollout token budget retained for TTT replay."),
    ] = 32768

    update_every_turns: Annotated[
        int | None,
        Field(gt=0, description="Train TTT LoRAs every N environment turns."),
    ] = 1

    update_every_tokens: Annotated[
        int | None,
        Field(gt=0, description="Train TTT LoRAs every N new tokens instead of turns."),
    ] = None

    overlap_turns: Annotated[
        int | None,
        Field(ge=0, description="Minimum already-trained turn overlap retained in the physical window."),
    ] = 2

    overlap_tokens: Annotated[
        int | None,
        Field(ge=0, description="Minimum already-trained token overlap retained in the physical window."),
    ] = None

    adapter_scope: Annotated[
        Literal["rollout"],
        Field(description="Scope for TTT adapters. Only per-rollout adapters are currently represented."),
    ] = "rollout"

    train_prompt_lora: Annotated[
        bool,
        Field(description="Train Phi_p on prompt/environment tokens."),
    ] = True

    train_completion_lora: Annotated[
        bool,
        Field(description="Train Phi_c on model completion tokens."),
    ] = True

    replay_policy: Annotated[
        Literal["turn_snapshots"],
        Field(description="How RL replay should recover active TTT adapter versions."),
    ] = "turn_snapshots"

    merge_prompt_lora: Annotated[
        Literal["after_trainer_step", "never"],
        Field(description="Whether final Phi_p adapters are merged into Theta after an RL optimizer step."),
    ] = "never"

    lora: Annotated[
        TTTLoRAConfig,
        Field(description="LoRA adapter configuration for TTT."),
    ] = TTTLoRAConfig()

    optim: Annotated[
        TTTOptimizerConfig,
        Field(description="Optimizer configuration for TTT adapter updates."),
    ] = TTTOptimizerConfig()

    @model_validator(mode="after")
    def validate_ttt_shape(self):
        if not self.enabled:
            return self
        if self.total_seq_len < self.window_seq_len:
            raise ValueError(
                f"TTT total_seq_len ({self.total_seq_len}) must be >= window_seq_len ({self.window_seq_len})."
            )
        if (self.update_every_turns is None) == (self.update_every_tokens is None):
            raise ValueError("Set exactly one of update_every_turns or update_every_tokens for TTT.")
        if (self.overlap_turns is None) == (self.overlap_tokens is None):
            raise ValueError("Set exactly one of overlap_turns or overlap_tokens for TTT.")
        if not self.train_prompt_lora and not self.train_completion_lora:
            raise ValueError("TTT must train at least one of prompt or completion LoRA.")
        return self
