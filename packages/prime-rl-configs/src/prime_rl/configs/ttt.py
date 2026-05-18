from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from prime_rl.utils.config import BaseConfig


class TTTLoRAConfig(BaseConfig):
    """LoRA parameters for test-time-training adapters."""

    rank: Annotated[
        Literal[8, 16, 32, 64, 128, 256, 320, 512],
        Field(description="Rank for the per-rollout online TTT LoRA."),
    ] = 8

    alpha: Annotated[
        int,
        Field(gt=0, description="LoRA alpha for the per-rollout online TTT LoRA."),
    ] = 16

    dropout: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Dropout used while training TTT LoRAs."),
    ] = 0.0

    target_modules: Annotated[
        list[str],
        Field(description="Module patterns targeted by the TTT LoRAs. Use 'auto' for the default transformer set."),
    ] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @field_validator("target_modules", mode="before")
    @classmethod
    def resolve_auto_target_modules(cls, value):
        if value == "auto":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return value


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


class TTTLearnerConfig(BaseConfig):
    """Runtime settings for the online TTT learner service."""

    host: Annotated[
        str,
        Field(description="Host the local TTT learner listens on."),
    ] = "127.0.0.1"

    port: Annotated[
        int,
        Field(gt=0, lt=65536, description="Port the local TTT learner listens on."),
    ] = 9009

    base_url: Annotated[
        str | None,
        Field(description="Optional explicit TTT learner base URL used by Verifiers clients."),
    ] = None

    adapter_dir: Annotated[
        Path | None,
        Field(description="Directory where materialized per-turn LoRA adapters are written."),
    ] = None

    device: Annotated[
        str,
        Field(description="Torch device used by the learner model replica."),
    ] = "cuda"

    dtype: Annotated[
        Literal["bfloat16", "float16", "float32"],
        Field(description="Torch dtype used by the learner model replica."),
    ] = "bfloat16"

    max_concurrent_sessions: Annotated[
        int,
        Field(gt=0, description="Maximum active rollout-local LoRA sessions retained by the learner."),
    ] = 64

    request_timeout_s: Annotated[
        float,
        Field(gt=0, description="HTTP request timeout for rollout-to-learner calls."),
    ] = 120.0

    load_adapters_into_vllm: Annotated[
        bool,
        Field(description="Whether the learner should call vLLM /load_lora_adapter after materialization."),
    ] = True

    unload_vllm_adapters: Annotated[
        bool,
        Field(description="Unload per-turn TTT LoRAs from vLLM as soon as their generation turn is complete."),
    ] = True

    session_offload: Annotated[
        Literal["none", "cpu_after_request"],
        Field(description="Where inactive rollout-local LoRA tensors and optimizer state live between learner calls."),
    ] = "cpu_after_request"

    delete_consumed_adapters: Annotated[
        bool,
        Field(description="Delete materialized TTT adapter directories after trainer replay and optional merge."),
    ] = True

    trainer_cache_device_tensors: Annotated[
        bool,
        Field(description="Keep GPU copies of replay adapters cached in trainer between active replay contexts."),
    ] = False

    vllm_admin_base_urls: Annotated[
        list[str],
        Field(description="vLLM admin base URLs that should receive materialized adapters."),
    ] = []

    snapshot_retention: Annotated[
        Literal["until_trainer_ack", "until_session_finish"],
        Field(description="How long learner snapshot metadata is retained."),
    ] = "until_trainer_ack"

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or f"http://{self.host}:{self.port}"


class TTTConfig(BaseConfig):
    """Configuration for per-rollout test-time training during RL rollouts."""

    enabled: Annotated[
        bool,
        Field(description="Enable TTT-aware rollout/training plumbing."),
    ] = False

    mode: Annotated[
        Literal["sliding_window_only", "online_lora"],
        Field(description="TTT implementation mode."),
    ] = "sliding_window_only"

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
        Field(gt=0, description="Deprecated. Online TTT now updates by tokens."),
    ] = None

    update_every_tokens: Annotated[
        int,
        Field(ge=2, description="Run one online TTT LoRA optimizer update per N exact rollout tokens."),
    ] = 1024

    overlap_turns: Annotated[
        int | None,
        Field(ge=0, description="Reserved for future overlap-aware window selection; currently not applied."),
    ] = 2

    overlap_tokens: Annotated[
        int | None,
        Field(ge=0, description="Reserved for future overlap-aware window selection; currently not applied."),
    ] = None

    adapter_scope: Annotated[
        Literal["rollout"],
        Field(description="Scope for TTT adapters. Only per-rollout adapters are currently represented."),
    ] = "rollout"

    replay_policy: Annotated[
        Literal["turn_snapshots"],
        Field(description="How RL replay should recover active TTT adapter versions."),
    ] = "turn_snapshots"

    theta_update_policy: Annotated[
        Literal["always_newest"],
        Field(description="How learner/inference bases track PipelineRL policy updates."),
    ] = "always_newest"

    keep_rollout_loras_across_theta_updates: Annotated[
        bool,
        Field(description="Keep active rollout LoRAs unchanged when Theta is refreshed."),
    ] = True

    require_exact_token_ids: Annotated[
        bool,
        Field(description="Fail TTT rollouts instead of using fallback trajectory retokenization."),
    ] = True

    cache_salt_includes_adapter: Annotated[
        bool,
        Field(description="Include active adapter version in vLLM cache salt."),
    ] = True

    lora: Annotated[
        TTTLoRAConfig,
        Field(description="LoRA adapter configuration for TTT."),
    ] = TTTLoRAConfig()

    optim: Annotated[
        TTTOptimizerConfig,
        Field(description="Optimizer configuration for TTT adapter updates."),
    ] = TTTOptimizerConfig()

    learner: Annotated[
        TTTLearnerConfig,
        Field(description="Online TTT learner service settings."),
    ] = TTTLearnerConfig()

    @model_validator(mode="after")
    def validate_ttt_shape(self):
        if not self.enabled:
            return self
        if self.total_seq_len < self.window_seq_len:
            raise ValueError(
                f"TTT total_seq_len ({self.total_seq_len}) must be >= window_seq_len ({self.window_seq_len})."
            )
        if self.update_every_turns is not None:
            raise ValueError("TTT update_every_turns is no longer supported; set update_every_tokens instead.")
        if self.mode == "online_lora" and not self.keep_rollout_loras_across_theta_updates:
            raise ValueError("online_lora TTT requires keep_rollout_loras_across_theta_updates=true.")
        return self


class ToolOutputTrainingConfig(BaseConfig):
    """Permanent trainer-side SFT on rendered tool-output content tokens."""

    enabled: Annotated[
        bool,
        Field(description="Enable auxiliary trainer-side SFT on prompt-side tool-output content tokens."),
    ] = False

    weight: Annotated[
        float,
        Field(ge=0.0, description="Scalar weight for the auxiliary tool-output NLL."),
    ] = 1.0

    tool_names: Annotated[
        list[str] | None,
        Field(description="Optional allowlist of tool names to train on. None trains on all tool outputs."),
    ] = None

    content_only: Annotated[
        bool,
        Field(description="Use renderer content_mask so role/control/separator tokens are excluded."),
    ] = True

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        names = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError("tool_names must contain non-empty strings.")
            names.append(item)
        return names
